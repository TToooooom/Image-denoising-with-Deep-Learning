import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Basic convolution block:
        Conv2d -> BatchNorm2d -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AutoEncoderDenoiser(nn.Module):
    """
    Convolutional AutoEncoder for image denoising.

    For 128x128 input:
        Encoder:
            128 -> 64 -> 32 -> 16

        Decoder:
            16 -> 32 -> 64 -> 128

    Input:
        [B, 3, H, W]

    Output:
        [B, 3, H, W]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2),
            nn.Conv2d(
                base_channels * 2,
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.enc3 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4),
            nn.Conv2d(
                base_channels * 4,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8),
            ConvBlock(base_channels * 8, base_channels * 8),
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 8,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            ConvBlock(base_channels * 4, base_channels * 4),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 4,
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            ConvBlock(base_channels * 2, base_channels * 2),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 2,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            ConvBlock(base_channels, base_channels),
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(
                base_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.bottleneck(x)

        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = self.output_layer(x)

        return x


if __name__ == "__main__":
    model = AutoEncoderDenoiser()
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)