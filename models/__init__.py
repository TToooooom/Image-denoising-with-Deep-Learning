from .cnn_denoiser import CNNDenoiser
from .autoencoder import AutoEncoderDenoiser
from .unet import UNetDenoiser
from .unet_residual import UNetResidualDenoiser


def build_model(
    model_name: str,
    in_channels: int = 3,
    out_channels: int = 3,
):
    """
    Build denoising model by name.

    Supported model_name:
        - cnn
        - autoencoder
        - unet
        - unet_residual
    """
    model_name = model_name.lower()

    if model_name == "cnn":
        return CNNDenoiser(
            in_channels=in_channels,
            out_channels=out_channels,
            features=64,
            depth=5,
        )

    if model_name == "autoencoder":
        return AutoEncoderDenoiser(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=32,
        )

    if model_name == "unet":
        return UNetDenoiser(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=32,
        )

    if model_name == "unet_residual":
        return UNetResidualDenoiser(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=32,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")