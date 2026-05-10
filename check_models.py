import torch

from models import build_model


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_single_model(model_name: str, device: torch.device) -> None:
    print("=" * 60)
    print(f"Checking model: {model_name}")

    model = build_model(
        model_name=model_name,
        in_channels=3,
        out_channels=3,
    ).to(device)

    model.eval()

    x = torch.randn(2, 3, 128, 128).to(device)

    with torch.no_grad():
        y = model(x)

    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Output range: min={y.min().item():.4f}, max={y.max().item():.4f}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    assert y.shape == x.shape, (
        f"Shape mismatch for {model_name}: "
        f"input={tuple(x.shape)}, output={tuple(y.shape)}"
    )

    print(f"{model_name} passed.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for model_name in ["cnn", "autoencoder", "unet", "unet_residual"]:
        check_single_model(model_name, device)

    print("=" * 60)
    print("All model checks passed.")


if __name__ == "__main__":
    main()