import argparse
import csv
import os
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import build_dataloaders
from losses import compute_restoration_loss
from models import build_model


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mse_to_psnr(mse: float, max_pixel_value: float = 1.0) -> float:
    """
    Convert MSE to PSNR.
    """
    if mse <= 0:
        return 100.0

    return 10.0 * np.log10((max_pixel_value ** 2) / mse)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_type: str,
    l1_weight: float,
    mse_weight: float,
    charbonnier_weight: float,
) -> Dict[str, float]:
    """
    Train model for one epoch.
    """
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Train Epoch {epoch}",
        leave=False,
    )

    for noisy_images, clean_images in progress_bar:
        noisy_images = noisy_images.to(device, non_blocking=True)
        clean_images = clean_images.to(device, non_blocking=True)

        outputs = model(noisy_images)

        loss = compute_restoration_loss(
            pred=outputs,
            target=clean_images,
            loss_type=loss_type,
            charbonnier_weight=charbonnier_weight,
            mse_weight=mse_weight,
            l1_weight=l1_weight,
        )

        outputs_for_metric = outputs.clamp(0.0, 1.0)
        mse = F.mse_loss(outputs_for_metric, clean_images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = noisy_images.size(0)
        total_loss += loss.item() * batch_size
        total_mse += mse.item() * batch_size
        total_samples += batch_size

        avg_loss = total_loss / total_samples
        progress_bar.set_postfix({"loss": f"{avg_loss:.6f}"})

    epoch_loss = total_loss / total_samples
    epoch_mse = total_mse / total_samples
    epoch_psnr = mse_to_psnr(epoch_mse)

    return {
        "loss": epoch_loss,
        "mse": epoch_mse,
        "psnr": epoch_psnr,
    }


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    loss_type: str,
    l1_weight: float,
    mse_weight: float,
    charbonnier_weight: float,
) -> Dict[str, float]:
    """
    Validate model for one epoch.
    """
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Val Epoch {epoch}",
        leave=False,
    )

    for noisy_images, clean_images in progress_bar:
        noisy_images = noisy_images.to(device, non_blocking=True)
        clean_images = clean_images.to(device, non_blocking=True)

        outputs = model(noisy_images)

        loss = compute_restoration_loss(
            pred=outputs,
            target=clean_images,
            loss_type=loss_type,
            charbonnier_weight=charbonnier_weight,
            mse_weight=mse_weight,
            l1_weight=l1_weight,
        )

        outputs_for_metric = outputs.clamp(0.0, 1.0)
        mse = F.mse_loss(outputs_for_metric, clean_images)

        batch_size = noisy_images.size(0)
        total_loss += loss.item() * batch_size
        total_mse += mse.item() * batch_size
        total_samples += batch_size

        avg_loss = total_loss / total_samples
        progress_bar.set_postfix({"loss": f"{avg_loss:.6f}"})

    epoch_loss = total_loss / total_samples
    epoch_mse = total_mse / total_samples
    epoch_psnr = mse_to_psnr(epoch_mse)

    return {
        "loss": epoch_loss,
        "mse": epoch_mse,
        "psnr": epoch_psnr,
    }


def save_checkpoint(
    save_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    """
    Save checkpoint.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "model_name": args.model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "image_size": args.image_size,
        "resize_size": args.resize_size,
        "noise_type": args.noise_type,
        "noise_level": args.noise_level,
        "loss_type": args.loss_type,
        "l1_weight": args.l1_weight,
        "mse_weight": args.mse_weight,
        "charbonnier_weight": args.charbonnier_weight,
        "args": vars(args),
    }

    torch.save(checkpoint, save_path)


def save_history_csv(
    history: List[Dict[str, float]],
    save_path: str,
) -> None:
    """
    Save training history to CSV.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_mse",
        "val_mse",
        "train_psnr",
        "val_psnr",
        "lr",
    ]

    with open(save_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in history:
            writer.writerow(row)


def plot_history(
    history: List[Dict[str, float]],
    save_path: str,
    title: str,
) -> None:
    """
    Save training curves.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_psnr = [row["train_psnr"] for row in history]
    val_psnr = [row["val_psnr"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_loss.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_psnr, label="Train PSNR")
    plt.plot(epochs, val_psnr, label="Val PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title(f"{title} - PSNR")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_psnr.png"), dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train denoising models on local images."
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default="./data/voc_images",
        help="Path to local image folder.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Training crop size.",
    )
    parser.add_argument(
        "--resize_size",
        type=int,
        default=160,
        help="Resize shorter side before crop.",
    )

    parser.add_argument(
        "--noise_type",
        type=str,
        default="gaussian",
        choices=["gaussian", "salt_pepper"],
        help="Noise type.",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.2,
        help="Noise level. Gaussian: sigma; salt_pepper: corruption probability.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="unet",
        choices=["cnn", "autoencoder", "unet", "unet_residual"],
        help="Model name.",
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="charbonnier_mse",
        choices=["mse", "l1", "l1_mse", "charbonnier", "charbonnier_mse"],
        help="Loss function.",
    )
    parser.add_argument("--l1_weight", type=float, default=0.8)
    parser.add_argument("--mse_weight", type=float, default=0.2)
    parser.add_argument("--charbonnier_weight", type=float, default=0.8)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )

    parser.add_argument(
        "--val_every",
        type=int,
        default=5,
        help="Run validation every N epochs. The last epoch is always validated.",
    )

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--output_dir", type=str, default="./outputs")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("Training configuration")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Image directory: {args.image_dir}")
    print(f"Model: {args.model_name}")
    print(f"Image size: {args.image_size}")
    print(f"Resize size: {args.resize_size}")
    print(f"Noise type: {args.noise_type}")
    print(f"Noise level: {args.noise_level}")
    print(f"Loss type: {args.loss_type}")
    print(f"L1 weight: {args.l1_weight}")
    print(f"MSE weight: {args.mse_weight}")
    print(f"Charbonnier weight: {args.charbonnier_weight}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Validation every: {args.val_every} epoch(s)")
    print("=" * 80)

    train_loader, val_loader, _ = build_dataloaders(
        image_dir=args.image_dir,
        image_size=args.image_size,
        resize_size=args.resize_size,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=args.seed,
    )

    model = build_model(
        model_name=args.model_name,
        in_channels=3,
        out_channels=3,
    ).to(device)

    print(f"Trainable parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    run_name = (
        f"{args.model_name}"
        f"_voc"
        f"_size{args.image_size}"
        f"_{args.noise_type}"
        f"{args.noise_level}"
        f"_{args.loss_type}"
    )

    best_checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"{run_name}_best.pth",
    )

    latest_checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"{run_name}_latest.pth",
    )

    history_csv_path = os.path.join(
        args.output_dir,
        "metrics",
        f"{run_name}_history.csv",
    )

    curve_save_path = os.path.join(
        args.output_dir,
        "train_curves",
        f"{run_name}.png",
    )

    best_val_loss = float("inf")

    last_val_metrics = {
        "loss": float("nan"),
        "mse": float("nan"),
        "psnr": float("nan"),
    }

    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_type=args.loss_type,
            l1_weight=args.l1_weight,
            mse_weight=args.mse_weight,
            charbonnier_weight=args.charbonnier_weight,
        )

        should_validate = (
            epoch % args.val_every == 0
            or epoch == args.epochs
        )

        if should_validate:
            val_metrics = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                epoch=epoch,
                loss_type=args.loss_type,
                l1_weight=args.l1_weight,
                mse_weight=args.mse_weight,
                charbonnier_weight=args.charbonnier_weight,
            )

            last_val_metrics = val_metrics
            scheduler.step(val_metrics["loss"])
        else:
            val_metrics = last_val_metrics

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "val_mse": val_metrics["mse"],
            "train_psnr": train_metrics["psnr"],
            "val_psnr": val_metrics["psnr"],
            "lr": current_lr,
        }

        history.append(row)

        print(
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Train PSNR: {train_metrics['psnr']:.2f} | "
            f"Val PSNR: {val_metrics['psnr']:.2f} | "
            f"LR: {current_lr:.6e}"
        )

        save_checkpoint(
            save_path=latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            args=args,
        )

        if should_validate and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            save_checkpoint(
                save_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                args=args,
            )

            print(f"Saved best checkpoint to: {best_checkpoint_path}")

        save_history_csv(
            history=history,
            save_path=history_csv_path,
        )

        plot_history(
            history=history,
            save_path=curve_save_path,
            title=run_name,
        )

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"History CSV: {history_csv_path}")
    print(f"Training curves saved to: {os.path.dirname(curve_save_path)}")


if __name__ == "__main__":
    main()