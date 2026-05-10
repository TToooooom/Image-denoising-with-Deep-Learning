import os

import matplotlib.pyplot as plt
import torch

from dataset import build_dataloaders


def show_batch_samples(
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    save_path: str,
    num_images: int = 6,
) -> None:
    """
    Save a comparison figure:
        clean image | noisy image
    """
    num_images = min(num_images, noisy_images.size(0))

    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_images,
        figsize=(num_images * 2.5, 5.0),
    )

    for i in range(num_images):
        clean_img = clean_images[i].detach().cpu().permute(1, 2, 0)
        noisy_img = noisy_images[i].detach().cpu().permute(1, 2, 0)

        axes[0, i].imshow(clean_img, vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[0, i].set_title("Clean", fontsize=10)

        axes[1, i].imshow(noisy_img, vmin=0, vmax=1)
        axes[1, i].axis("off")
        axes[1, i].set_title("Noisy", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    os.makedirs("./outputs/data_samples", exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        image_dir="./data/voc_images",
        image_size=128,
        resize_size=160,
        noise_type="gaussian",
        noise_level=0.1,
        batch_size=8,
        num_workers=0,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
    )

    noisy_images, clean_images = next(iter(train_loader))

    print("Local VOC image dataset check passed.")
    print(f"Noisy batch shape: {noisy_images.shape}")
    print(f"Clean batch shape: {clean_images.shape}")
    print(f"Noisy min/max: {noisy_images.min().item():.4f}, {noisy_images.max().item():.4f}")
    print(f"Clean min/max: {clean_images.min().item():.4f}, {clean_images.max().item():.4f}")

    save_path = "./outputs/data_samples/local_voc_gaussian_noise_0.1.png"

    show_batch_samples(
        noisy_images=noisy_images,
        clean_images=clean_images,
        save_path=save_path,
        num_images=6,
    )

    print(f"Saved sample visualization to: {save_path}")


if __name__ == "__main__":
    main()