import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from polyharmonic_conv import (
    PolyConvClassifier,
    default_channels,
    parse_padding,
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


def main():
    parser = argparse.ArgumentParser(description="MNIST polyharmonic convolutional classifier")
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
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

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                test_correct += (logits.argmax(dim=1) == y).sum().item()
                test_total += x.size(0)

        print(f"epoch {epoch}: train loss={total_loss/total:.4f}, "
              f"train acc={100*correct/total:.2f}%, "
              f"test acc={100*test_correct/test_total:.2f}%")

    print("final test accuracy:", 100 * test_correct / test_total)


if __name__ == "__main__":
    main()
