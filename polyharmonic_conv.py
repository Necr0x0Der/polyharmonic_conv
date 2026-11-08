import torch
import torch.nn as nn
import torch.nn.functional as F


def constellation(n: int, dtype=torch.float32, device="cpu"):
    """Hyperoctahedron + center: [0; -I; I], 2n+1 points."""
    eye = torch.eye(n, dtype=dtype, device=device)
    return torch.cat(
        [torch.zeros(1, n, dtype=dtype, device=device), -eye, eye],
        dim=0,
    )


def kernel_from_sqdist(d2: torch.Tensor, b: float = 5.0, c: float = 400.0,
                       eps: float = 1e-10):
    """k(tau) = 0.5 * ||tau||^2 * (ln ||tau||^2 - 2b) + c.
    At zero, the limiting value is c.
    """
    d2 = torch.clamp(d2, min=eps)
    return 0.5 * d2 * (torch.log(d2) - 2.0 * b) + c


def kernel_matrix(C: torch.Tensor, b: float, c: float, eps: float = 1e-10):
    d2 = torch.cdist(C, C, p=2) ** 2
    return kernel_from_sqdist(d2, b, c, eps)


def kernel_from_constellation(x: torch.Tensor, b: float = 5.0,
                              c: float = 400.0, eps: float = 1e-10):
    """Kernel between vectors x [N, n] and the fixed constellation [0; -I; I].

    For C = [0; -I; I], squared distances are:
      to 0:        ||x||^2
      to -e_i:     ||x||^2 + 1 + 2 x_i
      to +e_i:     ||x||^2 + 1 - 2 x_i

    This avoids an O(N * k * n) cdist and reduces the cost to O(N * k).
    """
    n = x.shape[1]
    q = (x * x).sum(dim=1, keepdim=True)       # [N, 1]
    d2 = torch.empty(x.shape[0], 2 * n + 1,
                     dtype=x.dtype, device=x.device)
    d2[:, 0] = q[:, 0]
    d2[:, 1:n + 1] = q + 1.0 + 2.0 * x
    d2[:, n + 1:2 * n + 1] = q + 1.0 - 2.0 * x
    return kernel_from_sqdist(d2, b, c, eps)


def default_channels(n_layers: int):
    """Default channel widths for a given number of convolutional layers."""
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    base = [16, 24, 32]
    if n_layers <= len(base):
        return base[:n_layers]
    return base + [32] * (n_layers - len(base))


def parse_padding(padding, patch_size: int) -> int:
    """padding=0 means valid convolution; padding='same' means patch_size // 2."""
    if isinstance(padding, str):
        if padding == "same":
            return patch_size // 2
        return int(padding)
    return int(padding)


class PolyConvLayer(nn.Module):
    """Local polyharmonic package with shared Y across all patch positions.

    Input:  x [B, C_in, H, W]
    Output: y [B, C_out, H_out, W_out]

    For every patch x_s in R^{p^2*C_in}:
        y_s = K(x_s, C) @ U @ Y
    C is fixed; Y is shared over all positions => translation-equivariant.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 patch_size: int = 3, stride: int = 1, padding="same",
                 b: float = 5.0, c: float = 400.0,
                 sigma2: float = 1e-2, use_tanh: bool = True):
        super().__init__()
        self.p = patch_size
        self.s = stride
        self.pad = parse_padding(padding, patch_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.b = b
        self.c = c
        self.use_tanh = use_tanh

        n = in_channels * patch_size * patch_size
        self.n = n
        k = 2 * n + 1

        C = constellation(n, dtype=torch.float64)
        Kc = kernel_matrix(C, b, c)
        # Keep every constant on Kc's device. Using torch.eye(...) without an
        # explicit device can fail when torch's default device is CUDA.
        I = torch.eye(k, dtype=torch.float64, device=Kc.device)
        U = torch.linalg.inv(Kc + sigma2 * I)
        self.register_buffer("C", C.float())
        self.register_buffer("U", U.float())

        # Shared function values at constellation points.
        self.Y = nn.Parameter(
            torch.randn(k, out_channels, device=U.device) * 0.1
        )

    def output_shape(self, H: int, W: int):
        Hp = H + 2 * self.pad
        Wp = W + 2 * self.pad
        Hout = (Hp - self.p) // self.s + 1
        Wout = (Wp - self.p) // self.s + 1
        return Hout, Wout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape

        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad))

        Hout, Wout = self.output_shape(H, W)
        L = Hout * Wout

        patches = F.unfold(x, kernel_size=self.p, stride=self.s)  # [B, n, L]
        patches = patches.transpose(1, 2).reshape(B * L, self.n)  # [B*L, n]

        K = kernel_from_constellation(patches, self.b, self.c)  # [B*L, k]

        # Compute U @ Y once (k x m), then K @ (U @ Y): O(N*k*m),
        # not O(N*k*k), which would dominate for large constellations.
        W = self.U @ self.Y                          # [k, m]
        y = K @ W                                    # [B*L, m]
        if self.use_tanh:
            y = torch.tanh(y)

        y = y.reshape(B, Hout, Wout, self.out_channels)
        y = y.permute(0, 3, 1, 2)                    # [B, m, Hout, Wout]
        return y


class PolyCascadeLayer(nn.Module):
    """Fully connected polyharmonic package: x in R^n -> y in R^m."""
    def __init__(self, in_dim: int, out_dim: int,
                 b: float = 5.0, c: float = 400.0,
                 sigma2: float = 1e-2, use_tanh: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.b = b
        self.c = c
        self.use_tanh = use_tanh

        n = in_dim
        k = 2 * n + 1
        C = constellation(n, dtype=torch.float64)
        Kc = kernel_matrix(C, b, c)
        I = torch.eye(k, dtype=torch.float64, device=Kc.device)
        U = torch.linalg.inv(Kc + sigma2 * I)
        self.register_buffer("C", C.float())
        self.register_buffer("U", U.float())
        self.Y = nn.Parameter(torch.randn(k, out_dim, device=U.device) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = kernel_from_constellation(x, self.b, self.c)
        W = self.U @ self.Y
        y = K @ W
        if self.use_tanh:
            y = torch.tanh(y)
        return y


class PolyConvClassifier(nn.Module):
    """
    A deep classifier made of convolutional polyharmonic layers plus
    fully connected polyharmonic layers.

    Parameters:
      conv_channels: channel widths for each convolutional layer.
      patch_size: patch / kernel size.
      stride: default patch stride.
      strides: optional per-layer strides.
      padding: 0 (valid) or "same" (patch_size//2 padding).
      head_dims: output dims of the fully connected polyharmonic layers.

    Defaults (4 layers, stride 1, "same" padding) preserve the spatial
    feature-map size across all convolutional layers.
    """
    def __init__(self, conv_channels=(16, 24, 32, 32),
                 patch_size: int = 3, stride: int = 1, strides=None,
                 padding="same", head_dims=(32, 10),
                 b: float = 5.0, c: float = 400.0, sigma2: float = 1e-2):
        super().__init__()
        conv_channels = list(conv_channels)
        head_dims = list(head_dims)

        if strides is None:
            strides = [stride] * len(conv_channels)
        elif isinstance(strides, int):
            strides = [strides] * len(conv_channels)
        else:
            strides = list(strides)

        if len(strides) != len(conv_channels):
            raise ValueError("strides must match the number of conv_channels")

        self.conv_layers = nn.ModuleList()
        in_ch = 1
        for out_ch, s in zip(conv_channels, strides):
            self.conv_layers.append(
                PolyConvLayer(in_ch, out_ch, patch_size, s, padding,
                              b, c, sigma2, use_tanh=True)
            )
            in_ch = out_ch

        self.head_layers = nn.ModuleList()
        in_dim = conv_channels[-1]
        for i, out_dim in enumerate(head_dims):
            self.head_layers.append(
                PolyCascadeLayer(in_dim, out_dim, b, c, sigma2,
                                use_tanh=(i < len(head_dims) - 1))
            )
            in_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.conv_layers:
            x = layer(x)
        x = x.mean(dim=(2, 3))          # global average pooling
        for layer in self.head_layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    model = PolyConvClassifier()
    print(model)
    print("params:", sum(p.numel() for p in model.parameters()))
    device = next(model.parameters()).device
    x = torch.randn(4, 1, 28, 28, device=device)
    y = model(x)
    print("output shape:", y.shape)
