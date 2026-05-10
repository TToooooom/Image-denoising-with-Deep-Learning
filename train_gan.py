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
from models.discriminator import PatchDiscriminator


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_requires_grad(model: torch.nn.Module, requires_grad: bool) -> None:
    """
    Enable or disable gradients for all parameters in a model.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad


def mse_to_psnr(mse: float, max_pixel_value: float = 1.0) -> float:
    """
    Convert MSE to PSNR.
    """
    if mse <= 0:
        return 100.0

    return 10.0 * np.log10((max_pixel_value ** 2) / mse)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Simple gradient / edge loss.

    It encourages the restored image to preserve local edge structures.
    """
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_x = F.l1_loss(pred_dx, target_dx)
    loss_y = F.l1_loss(pred_dy, target_dy)

    return loss_x + loss_y


def load_generator_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load a trained ordinary U-Net as GAN generator.

    The generator architecture is still ordinary U-Net, not U-Net residual.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Pretrained generator checkpoint not found: {checkpoint_path}"
        )

    generator = build_model(
        model_name="unet",
        in_channels=3,
        out_channels=3,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    generator.load_state_dict(state_dict, strict=True)

    return generator


def save_checkpoint(
    save_path: str,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int,
    best_val_mse: float,
    args: argparse.Namespace,
) -> None:
    """
    Save GAN fine-tuned checkpoint.

    Important:
        "model_state_dict" stores generator weights.
        Therefore evaluate.py can load this checkpoint using:
            --model_name unet
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "model_name": "unet",
        "training_type": "unet_gan_finetune",
        "model_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "epoch": epoch,
        "best_val_mse": best_val_mse,
        "image_size": args.image_size,
        "resize_size": args.resize_size,
        "noise_type": args.noise_type,
        "noise_level": args.noise_level,
        "pretrained_generator": args.pretrained_generator,
        "lambda_rec": args.lambda_rec,
        "lambda_adv": args.lambda_adv,
        "lambda_grad": args.lambda_grad,
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
        "train_g_loss",
        "train_d_loss",
        "train_rec_loss",
        "train_adv_loss",
        "train_grad_loss",
        "train_mse",
        "train_psnr",
        "val_mse",
        "val_psnr",
        "lr_g",
        "lr_d",
    ]

    with open(save_path, "w", newline="", encoding="utf-8") as f:
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
    Save GAN loss curve and PSNR curve.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = [row["epoch"] for row in history]

    train_g_loss = [row["train_g_loss"] for row in history]
    train_d_loss = [row["train_d_loss"] for row in history]

    train_psnr = [row["train_psnr"] for row in history]
    val_psnr = [row["val_psnr"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_g_loss, label="Generator Loss")
    plt.plot(epochs, train_d_loss, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - GAN Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_gan_loss.png"), dpi=300)
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


def train_one_epoch_gan(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """
    Train generator and discriminator for one epoch.
    """
    generator.train()
    discriminator.train()

    bce_loss = torch.nn.BCEWithLogitsLoss()

    total_g_loss = 0.0
    total_d_loss = 0.0
    total_rec_loss = 0.0
    total_adv_loss = 0.0
    total_grad_loss = 0.0
    total_mse = 0.0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"GAN Train Epoch {epoch}",
        leave=False,
    )

    for noisy_images, clean_images in progress_bar:
        noisy_images = noisy_images.to(device, non_blocking=True)
        clean_images = clean_images.to(device, non_blocking=True)

        batch_size = noisy_images.size(0)

        # ==================================================
        # 1. Train Discriminator
        # ==================================================
        set_requires_grad(discriminator, True)

        with torch.no_grad():
            fake_images = generator(noisy_images)
            fake_images = fake_images.clamp(0.0, 1.0)

        real_logits = discriminator(noisy_images, clean_images)
        fake_logits = discriminator(noisy_images, fake_images.detach())

        real_targets = torch.ones_like(real_logits) * args.real_label
        fake_targets = torch.zeros_like(fake_logits)

        d_real_loss = bce_loss(real_logits, real_targets)
        d_fake_loss = bce_loss(fake_logits, fake_targets)
        d_loss = 0.5 * (d_real_loss + d_fake_loss)

        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # ==================================================
        # 2. Train Generator
        # ==================================================
        set_requires_grad(discriminator, False)

        restored_images = generator(noisy_images)
        restored_for_d = restored_images.clamp(0.0, 1.0)

        fake_logits_for_g = discriminator(noisy_images, restored_for_d)
        adv_targets = torch.ones_like(fake_logits_for_g)

        adv_loss = bce_loss(fake_logits_for_g, adv_targets)

        rec_loss = compute_restoration_loss(
            pred=restored_images,
            target=clean_images,
            loss_type=args.rec_loss_type,
            charbonnier_weight=args.charbonnier_weight,
            mse_weight=args.mse_weight,
            l1_weight=args.l1_weight,
        )

        grad_loss = gradient_loss(
            pred=restored_for_d,
            target=clean_images,
        )

        g_loss = (
            args.lambda_rec * rec_loss
            + args.lambda_adv * adv_loss
            + args.lambda_grad * grad_loss
        )

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

        restored_for_metric = restored_images.clamp(0.0, 1.0)
        mse = F.mse_loss(restored_for_metric, clean_images)

        total_g_loss += g_loss.item() * batch_size
        total_d_loss += d_loss.item() * batch_size
        total_rec_loss += rec_loss.item() * batch_size
        total_adv_loss += adv_loss.item() * batch_size
        total_grad_loss += grad_loss.item() * batch_size
        total_mse += mse.item() * batch_size
        total_samples += batch_size

        avg_g = total_g_loss / total_samples
        avg_d = total_d_loss / total_samples
        avg_mse = total_mse / total_samples
        avg_psnr = mse_to_psnr(avg_mse)

        progress_bar.set_postfix(
            {
                "G": f"{avg_g:.4f}",
                "D": f"{avg_d:.4f}",
                "PSNR": f"{avg_psnr:.2f}",
            }
        )

    epoch_mse = total_mse / total_samples

    return {
        "g_loss": total_g_loss / total_samples,
        "d_loss": total_d_loss / total_samples,
        "rec_loss": total_rec_loss / total_samples,
        "adv_loss": total_adv_loss / total_samples,
        "grad_loss": total_grad_loss / total_samples,
        "mse": epoch_mse,
        "psnr": mse_to_psnr(epoch_mse),
    }


@torch.no_grad()
def validate_generator(
    generator: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Validate only the generator.
    """
    generator.eval()

    total_mse = 0.0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"GAN Val Epoch {epoch}",
        leave=False,
    )

    for noisy_images, clean_images in progress_bar:
        noisy_images = noisy_images.to(device, non_blocking=True)
        clean_images = clean_images.to(device, non_blocking=True)

        restored_images = generator(noisy_images)
        restored_images = restored_images.clamp(0.0, 1.0)

        mse = F.mse_loss(restored_images, clean_images)

        batch_size = noisy_images.size(0)
        total_mse += mse.item() * batch_size
        total_samples += batch_size

        avg_mse = total_mse / total_samples
        progress_bar.set_postfix({"PSNR": f"{mse_to_psnr(avg_mse):.2f}"})

    val_mse = total_mse / total_samples

    return {
        "mse": val_mse,
        "psnr": mse_to_psnr(val_mse),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GAN fine-tuning for U-Net image denoising."
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default="./data/voc_images",
    )

    parser.add_argument(
        "--pretrained_generator",
        type=str,
        required=True,
        help="Path to trained ordinary U-Net checkpoint.",
    )

    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--resize_size", type=int, default=160)

    parser.add_argument(
        "--noise_type",
        type=str,
        default="gaussian",
        choices=["gaussian", "salt_pepper"],
    )
    parser.add_argument("--noise_level", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--lr_g", type=float, default=1e-5)
    parser.add_argument("--lr_d", type=float, default=1e-4)

    parser.add_argument(
        "--rec_loss_type",
        type=str,
        default="charbonnier_mse",
        choices=["mse", "l1", "l1_mse", "charbonnier", "charbonnier_mse"],
    )

    parser.add_argument("--l1_weight", type=float, default=0.8)
    parser.add_argument("--mse_weight", type=float, default=0.2)
    parser.add_argument("--charbonnier_weight", type=float, default=0.8)

    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=0.001)
    parser.add_argument("--lambda_grad", type=float, default=0.05)

    parser.add_argument(
        "--real_label",
        type=float,
        default=0.9,
        help="One-sided label smoothing for real samples.",
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )

    parser.add_argument("--val_every", type=int, default=1)

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
    print("U-Net GAN Fine-tuning Configuration")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Image dir: {args.image_dir}")
    print(f"Pretrained generator: {args.pretrained_generator}")
    print(f"Image size: {args.image_size}")
    print(f"Resize size: {args.resize_size}")
    print(f"Noise: {args.noise_type}, level={args.noise_level}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"lr_g: {args.lr_g}")
    print(f"lr_d: {args.lr_d}")
    print(f"rec_loss_type: {args.rec_loss_type}")
    print(f"lambda_rec: {args.lambda_rec}")
    print(f"lambda_adv: {args.lambda_adv}")
    print(f"lambda_grad: {args.lambda_grad}")
    print(f"real_label: {args.real_label}")
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

    generator = load_generator_from_checkpoint(
        checkpoint_path=args.pretrained_generator,
        device=device,
    )

    discriminator = PatchDiscriminator(
        in_channels=6,
        base_channels=64,
    ).to(device)

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr_g,
        betas=(0.5, 0.999),
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr_d,
        betas=(0.5, 0.999),
    )

    run_name = (
        f"unet_gan"
        f"_voc"
        f"_size{args.image_size}"
        f"_{args.noise_type}"
        f"{args.noise_level}"
        f"_adv{args.lambda_adv}"
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

    best_val_mse = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nGAN Fine-tune Epoch [{epoch}/{args.epochs}]")

        train_metrics = train_one_epoch_gan(
            generator=generator,
            discriminator=discriminator,
            dataloader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=device,
            epoch=epoch,
            args=args,
        )

        should_validate = (
            epoch % args.val_every == 0
            or epoch == args.epochs
        )

        if should_validate:
            val_metrics = validate_generator(
                generator=generator,
                dataloader=val_loader,
                device=device,
                epoch=epoch,
            )
        else:
            val_metrics = {
                "mse": float("nan"),
                "psnr": float("nan"),
            }

        row = {
            "epoch": epoch,
            "train_g_loss": train_metrics["g_loss"],
            "train_d_loss": train_metrics["d_loss"],
            "train_rec_loss": train_metrics["rec_loss"],
            "train_adv_loss": train_metrics["adv_loss"],
            "train_grad_loss": train_metrics["grad_loss"],
            "train_mse": train_metrics["mse"],
            "train_psnr": train_metrics["psnr"],
            "val_mse": val_metrics["mse"],
            "val_psnr": val_metrics["psnr"],
            "lr_g": optimizer_g.param_groups[0]["lr"],
            "lr_d": optimizer_d.param_groups[0]["lr"],
        }

        history.append(row)

        print(
            f"G Loss: {train_metrics['g_loss']:.6f} | "
            f"D Loss: {train_metrics['d_loss']:.6f} | "
            f"Rec: {train_metrics['rec_loss']:.6f} | "
            f"Adv: {train_metrics['adv_loss']:.6f} | "
            f"Grad: {train_metrics['grad_loss']:.6f} | "
            f"Train PSNR: {train_metrics['psnr']:.2f} | "
            f"Val PSNR: {val_metrics['psnr']:.2f}"
        )

        save_checkpoint(
            save_path=latest_checkpoint_path,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            epoch=epoch,
            best_val_mse=best_val_mse,
            args=args,
        )

        if should_validate and val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]

            save_checkpoint(
                save_path=best_checkpoint_path,
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                epoch=epoch,
                best_val_mse=best_val_mse,
                args=args,
            )

            print(f"Saved best GAN fine-tuned generator to: {best_checkpoint_path}")

        save_history_csv(
            history=history,
            save_path=history_csv_path,
        )

        plot_history(
            history=history,
            save_path=curve_save_path,
            title=run_name,
        )

    print("\nGAN fine-tuning finished.")
    print(f"Best val MSE: {best_val_mse:.6f}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Latest checkpoint: {latest_checkpoint_path}")
    print(f"History CSV: {history_csv_path}")


if __name__ == "__main__":
    main()