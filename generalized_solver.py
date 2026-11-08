import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from polyharmonic_conv import (
    PolyConvClassifier,
    default_channels,
    parse_padding,
    kernel_from_constellation,
)


def parse_csv_ints(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_model(args):
    if args.channels:
        channels = parse_csv_ints(args.channels)
        if args.layers is not None and len(channels) != args.layers:
            raise ValueError("--layers must equal len(--channels)")
        layers = len(channels)
    else:
        layers = args.layers if args.layers is not None else 4
        channels = default_channels(layers)

    padding = parse_padding(args.padding, args.patch_size)
    head_dims = parse_csv_ints(args.head_dims)
    strides = parse_csv_ints(args.strides) if args.strides else None

    return PolyConvClassifier(
        conv_channels=channels,
        patch_size=args.patch_size,
        stride=args.stride,
        strides=strides,
        padding=padding,
        head_dims=head_dims,
    )


class GeneralizedPolySolver:
    """
    A linear-solver (Gauss-Newton-like) optimizer for a polyharmonic network
    with shared Y in convolutional layers.

    For each layer l:
        A_l = H_l Y_l
        X_l = tanh(A_l) or A_l
    Given g_l = dL/dA_l, for each output channel t solve

        (H_l^T H_l + alpha I) dY_l[:, t] = - H_l^T g_l[:, t]

    and update Y_l. No SGD is used; the update is a linear solve.
    """
    def __init__(self, model, alpha=1e-2, step=1.0):
        self.model = model
        self.alpha = alpha
        self.step = step

    @staticmethod
    def _conv_rows(layer, x):
        B, C_in, H, W = x.shape

        if layer.pad > 0:
            x = F.pad(x, (layer.pad, layer.pad, layer.pad, layer.pad))

        Hout, Wout = layer.output_shape(H, W)
        L = Hout * Wout

        patches = F.unfold(x, kernel_size=layer.p, stride=layer.s)
        patches = patches.transpose(1, 2).reshape(B * L, layer.n)

        K = kernel_from_constellation(patches, layer.b, layer.c)
        H_rows = K @ layer.U  # [B*L, k]
        return H_rows, (B, Hout, Wout, layer.out_channels)

    @staticmethod
    def _head_rows(layer, x):
        K = kernel_from_constellation(x, layer.b, layer.c)
        return K @ layer.U  # [B, k]

    def forward_cache(self, x):
        """
        Returns final logits and a list of (layer, H_rows, A_l, shape)
        for each polyharmonic layer.
        """
        cache = []

        for layer in self.model.conv_layers:
            H_rows, (B, Hout, Wout, m) = self._conv_rows(layer, x)
            A = H_rows @ layer.Y                    # [B*L, m]
            cache.append((layer, H_rows, A, (B, Hout, Wout, m)))
            y = A.reshape(B, Hout, Wout, m).permute(0, 3, 1, 2)
            if layer.use_tanh:
                y = torch.tanh(y)
            x = y

        # same global average pooling as in PolyConvClassifier.forward
        x = x.mean(dim=(2, 3))

        for layer in self.model.head_layers:
            H_rows = self._head_rows(layer, x)      # [B, k]
            A = H_rows @ layer.Y                    # [B, m]
            cache.append((layer, H_rows, A, (x.shape[0], layer.out_dim)))
            y = torch.tanh(A) if layer.use_tanh else A
            x = y

        return x, cache

    def step_update(self, x, y_target, loss_fn):
        logits, cache = self.forward_cache(x)
        loss = loss_fn(logits, y_target)

        # Gradients of loss w.r.t. pre-activations A_l.
        A_list = [c[2] for c in cache]
        grads = torch.autograd.grad(loss, A_list, retain_graph=True, create_graph=False)

        with torch.no_grad():
            for (layer, H_rows, A, shape), g in zip(cache, grads):
                # g has shape [N, m]; update each output channel separately.
                m = layer.Y.shape[1]
                HtH = H_rows.T @ H_rows + self.alpha * torch.eye(
                    H_rows.shape[1], device=H_rows.device
                )
                for t in range(m):
                    b = H_rows.T @ g[:, t]
                    dY_t = -torch.linalg.solve(HtH, b)
                    layer.Y[:, t] += self.step * dY_t

        return loss.item(), logits


def train_solver(model, train_loader, test_loader, device,
                 epochs=8, alpha=1e-2, step=1.0):
    solver = GeneralizedPolySolver(model, alpha=alpha, step=step)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss, logits = solver.step_update(x, y, loss_fn)
            total_loss += loss * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = solver.forward_cache(x)
                test_correct += (logits.argmax(1) == y).sum().item()
                test_total += x.size(0)

        print(f"epoch {epoch}: train loss={total_loss/total:.4f}, "
              f"train acc={100*correct/total:.2f}%, "
              f"test acc={100*test_correct/test_total:.2f}%")

    return 100 * test_correct / test_total


def main():
    parser = argparse.ArgumentParser(description="MNIST polyharmonic convolutional classifier with linear solver")
    parser.add_argument("--layers", type=int, default=4,
                        help="number of convolutional polyharmonic layers")
    parser.add_argument("--channels", type=str, default=None,
                        help="comma-separated channel widths, e.g. 16,24,32,32")
    parser.add_argument("--patch-size", type=int, default=3,
                        help="patch / kernel size")
    parser.add_argument("--stride", type=int, default=1,
                        help="default patch stride; used if --strides is not supplied")
    parser.add_argument("--strides", type=str, default=None,
                        help="comma-separated per-layer strides, e.g. 2,2,1")
    parser.add_argument("--padding", type=str, default="same",
                        help="0 for valid, or 'same' for patch_size//2 padding")
    parser.add_argument("--head-dims", type=str, default="32,10",
                        help="comma-separated fully connected polyharmonic output dims")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda",
                        help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = build_model(args).to(device)
    print(model)
    print("trainable parameters:", sum(p.numel() for p in model.parameters()))

    final_acc = train_solver(
        model, train_loader, test_loader, device,
        epochs=args.epochs, alpha=args.alpha, step=args.step,
    )
    print("final test accuracy:", final_acc)


if __name__ == "__main__":
    main()
