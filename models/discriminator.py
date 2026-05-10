import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    Conditional PatchGAN discriminator.

    It receives:
        noisy image  : [B, 3, H, W]
        target image : [B, 3, H, W]

    Concatenated input:
        [noisy, target] -> [B, 6, H, W]

    Output:
        patch logits, not probabilities.
        Shape roughly [B, 1, H/16, W/16]
    """

    def __init__(
        self,
        in_channels: int = 6,
        base_channels: int = 64,
    ) -> None:
        super().__init__()

        def block(
            in_c: int,
            out_c: int,
            stride: int = 2,
            use_bn: bool = True,
        ):
            layers = [
                nn.Conv2d(
                    in_c,
                    out_c,
                    kernel_size=4,
                    stride=stride,
                    padding=1,
                    bias=not use_bn,
                )
            ]

            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            # [B, 6, H, W] -> [B, 64, H/2, W/2]
            block(in_channels, base_channels, stride=2, use_bn=False),

            # -> [B, 128, H/4, W/4]
            block(base_channels, base_channels * 2, stride=2, use_bn=True),

            # -> [B, 256, H/8, W/8]
            block(base_channels * 2, base_channels * 4, stride=2, use_bn=True),

            # -> [B, 512, H/16, W/16]
            block(base_channels * 4, base_channels * 8, stride=2, use_bn=True),

            # Patch logits
            nn.Conv2d(
                base_channels * 8,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
        )

    def forward(
        self,
        noisy: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([noisy, target], dim=1)
        logits = self.net(x)
        return logits


if __name__ == "__main__":
    model = PatchDiscriminator()
    noisy = torch.randn(2, 3, 128, 128)
    clean = torch.randn(2, 3, 128, 128)
    out = model(noisy, clean)

    print("Noisy shape:", noisy.shape)
    print("Clean shape:", clean.shape)
    print("Output logits shape:", out.shape)