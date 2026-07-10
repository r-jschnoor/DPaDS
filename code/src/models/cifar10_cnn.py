import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Residual block for ResNet20 (He et al.), BatchNorm replaced with GroupNorm.

    conv3x3(stride) -> GN -> ReLU -> conv3x3(1) -> GN, added to a shortcut path,
    then ReLU. The shortcut is the identity when the shape matches; otherwise a
    1x1-conv(stride)+GroupNorm projection handles the
    channel/spatial-size change between stages.
    """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(min(32, out_channels), out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(min(32, out_channels), out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(32, out_channels), out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out = out + identity
        out = self.relu2(out)
        return out


class Cifar10ResNet20(nn.Module):
    """
    ResNet20 for CIFAR-10 (32x32 RGB images, 10 classes).

    Matches the FLTrust paper's CIFAR-10 architecture.
    The standard ResNet20 (6n+2 = 20 layers, n=3) with BatchNorm
    replaced by GroupNorm throughout, since BatchNorm is incompatible with
    Opacus/DP-SGD (no valid per-sample gradients). Mirrors MnistCNN's role for
    MNIST: the model each simulated client trains locally.
    """

    def __init__(self):
        super().__init__()
        # ----- Stem -----
        self.stem_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_gn   = nn.GroupNorm(16, 16)
        self.stem_relu = nn.ReLU()

        # ----- Stage 1: 16 -> 16 channels, 32x32 -----
        self.stage1 = nn.Sequential(
            BasicBlock(16, 16, stride=1),
            BasicBlock(16, 16, stride=1),
            BasicBlock(16, 16, stride=1),
        )

        # ----- Stage 2: 16 -> 32 channels, 32x32 -> 16x16 -----
        self.stage2 = nn.Sequential(
            BasicBlock(16, 32, stride=2),
            BasicBlock(32, 32, stride=1),
            BasicBlock(32, 32, stride=1),
        )

        # ----- Stage 3: 32 -> 64 channels, 16x16 -> 8x8 -----
        self.stage3 = nn.Sequential(
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1),
        )

        # ----- Head -----
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(64, 10)

    def forward(self, x):
        x = self.stem_conv(x)   # (4, 3,  32, 32) -> (4, 16, 32, 32)
        x = self.stem_gn(x)
        x = self.stem_relu(x)

        x = self.stage1(x)      # (4, 16, 32, 32) -> (4, 16, 32, 32)
        x = self.stage2(x)      # (4, 16, 32, 32) -> (4, 32, 16, 16)
        x = self.stage3(x)      # (4, 32, 16, 16) -> (4, 64, 8,  8)

        x = self.pool(x)        # (4, 64, 8, 8) -> (4, 64, 1, 1)
        x = self.flatten(x)     # -> (4, 64)
        x = self.fc(x)          # -> (4, 10)
        return x


if __name__ == "__main__":
    model = Cifar10ResNet20()
    print(f"{model}\n")

    dummy = torch.zeros(4, 3, 32, 32)   # 4 Dummy CIFAR-10 images
    out = model(dummy)                  # Call the model with the dummy images
    print(f"Input shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")
