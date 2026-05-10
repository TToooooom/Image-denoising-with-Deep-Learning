import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution block:
        Conv2d -> BatchNorm2d -> ReLU
        Conv2d -> BatchNorm2d -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    """
    Downsampling block:
        MaxPool2d -> DoubleConv
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """
    Upsampling block:
        ConvTranspose2d -> concatenate skip feature -> DoubleConv
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Make sure spatial sizes match before concatenation.
        # This improves robustness when H/W are not perfectly divisible.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)

        return x


class UNetDenoiser(nn.Module):
    """
    U-Net for image denoising.

    For 128x128 input:
        Encoder:
            128 -> 64 -> 32 -> 16 -> 8

        Decoder:
            8 -> 16 -> 32 -> 64 -> 128

    Input:
        noisy image: [B, 3, H, W]

    Output:
        restored image: [B, 3, H, W]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        self.inc = DoubleConv(in_channels, base_channels)

        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        self.down4 = DownBlock(base_channels * 8, base_channels * 16)

        self.up4 = UpBlock(
            in_channels=base_channels * 16,
            skip_channels=base_channels * 8,
            out_channels=base_channels * 8,
        )

        self.up3 = UpBlock(
            in_channels=base_channels * 8,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4,
        )

        self.up2 = UpBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
        )

        self.up1 = UpBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
        )

        self.outc = nn.Sequential(
            nn.Conv2d(
                base_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)       # [B, C, H, W]
        x2 = self.down1(x1)    # [B, 2C, H/2, W/2]
        x3 = self.down2(x2)    # [B, 4C, H/4, W/4]
        x4 = self.down3(x3)    # [B, 8C, H/8, W/8]
        x5 = self.down4(x4)    # [B, 16C, H/16, W/16]

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        x = self.outc(x)

        return x


if __name__ == "__main__":
    model = UNetDenoiser()
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)