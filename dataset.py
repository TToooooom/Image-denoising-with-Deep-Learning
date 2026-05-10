import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from noise import add_noise


IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_images(root_dir: str) -> List[str]:
    """
    Recursively find image files in root_dir.
    """
    image_paths = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in IMG_EXTENSIONS:
                image_paths.append(os.path.join(dirpath, filename))

    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in: {root_dir}")

    return image_paths


def split_image_paths(
    image_paths: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split image paths into train, val, and test subsets.
    """
    rng = random.Random(seed)
    paths = image_paths.copy()
    rng.shuffle(paths)

    n_total = len(paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_paths = paths[:n_train]
    val_paths = paths[n_train:n_train + n_val]
    test_paths = paths[n_train + n_val:]

    return train_paths, val_paths, test_paths


class LocalImageDenoisingDataset(Dataset):
    """
    Local image dataset for image denoising.

    Each sample:
        noisy_image: [3, H, W], values in [0, 1]
        clean_image: [3, H, W], values in [0, 1]

    No labels are needed.
    """

    def __init__(
        self,
        image_paths: List[str],
        mode: str = "train",
        image_size: int = 128,
        resize_size: int = 160,
        noise_type: str = "gaussian",
        noise_level: float = 0.1,
    ) -> None:
        self.image_paths = image_paths
        self.mode = mode
        self.image_size = image_size
        self.resize_size = resize_size
        self.noise_type = noise_type
        self.noise_level = noise_level

        if mode == "train":
            self.transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")
        clean_image = self.transform(image)

        noisy_image = add_noise(
            clean_image,
            noise_type=self.noise_type,
            noise_level=self.noise_level,
        )

        return noisy_image, clean_image


def build_dataloaders(
    image_dir: str = "./data/voc_images",
    image_size: int = 128,
    resize_size: int = 160,
    noise_type: str = "gaussian",
    noise_level: float = 0.1,
    batch_size: int = 16,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Build train, validation, and test dataloaders from a local image folder.

    Each batch:
        noisy_images, clean_images
    """
    image_paths = find_images(image_dir)

    train_paths, val_paths, test_paths = split_image_paths(
        image_paths=image_paths,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_dataset = LocalImageDenoisingDataset(
        image_paths=train_paths,
        mode="train",
        image_size=image_size,
        resize_size=resize_size,
        noise_type=noise_type,
        noise_level=noise_level,
    )

    val_dataset = LocalImageDenoisingDataset(
        image_paths=val_paths,
        mode="val",
        image_size=image_size,
        resize_size=resize_size,
        noise_type=noise_type,
        noise_level=noise_level,
    )

    test_dataset = LocalImageDenoisingDataset(
        image_paths=test_paths,
        mode="test",
        image_size=image_size,
        resize_size=resize_size,
        noise_type=noise_type,
        noise_level=noise_level,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Total images: {len(image_paths)}")
    print(f"Train images: {len(train_paths)}")
    print(f"Val images: {len(val_paths)}")
    print(f"Test images: {len(test_paths)}")

    return train_loader, val_loader, test_loader