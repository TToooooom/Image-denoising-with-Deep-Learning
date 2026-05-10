import torch
import torch.nn as nn


class CNNDenoiser(nn.Module):
    """
    Simple CNN denoising baseline.

    Input:
        noisy image tensor: [B, 3, H, W]

    Output:
        restored image tensor: [B, 3, H, W]

    This model only uses local convolutional operations.
    It is used as the simplest baseline.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: int = 64,
        depth: int = 5,
    ) -> None:
        super().__init__()

        layers = []

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        layers.append(nn.ReLU(inplace=True))

        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(
                in_channels=features,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        # Keep output in [0, 1]
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    model = CNNDenoiser()
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)