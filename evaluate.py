import argparse
import csv
import os
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import build_dataloaders
from models import build_model


try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducible test noise.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_numpy_image(x: torch.Tensor) -> np.ndarray:
    """
    Convert tensor [3, H, W] in [0, 1] to numpy image [H, W, 3].
    """
    x = x.detach().cpu().clamp(0.0, 1.0)
    x = x.permute(1, 2, 0).numpy()
    return x


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute MSE for one image tensor.
    """
    return torch.mean((pred - target) ** 2).item()


def mse_to_psnr(mse: float, max_pixel_value: float = 1.0) -> float:
    """
    Convert MSE to PSNR.
    """
    if mse <= 0:
        return 100.0
    return 10.0 * np.log10((max_pixel_value ** 2) / mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute SSIM for one RGB image.
    """
    if not HAS_SKIMAGE:
        return float("nan")

    pred_np = tensor_to_numpy_image(pred)
    target_np = tensor_to_numpy_image(target)

    return float(
        ssim(
            target_np,
            pred_np,
            channel_axis=2,
            data_range=1.0,
        )
    )


def compute_image_metrics(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    restored: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute metrics for one sample:
        noisy vs clean
        restored vs clean
    """
    noisy_mse = compute_mse(noisy, clean)
    restored_mse = compute_mse(restored, clean)

    metrics = {
        "noisy_mse": noisy_mse,
        "restored_mse": restored_mse,
        "noisy_psnr": mse_to_psnr(noisy_mse),
        "restored_psnr": mse_to_psnr(restored_mse),
        "noisy_ssim": compute_ssim(noisy, clean),
        "restored_ssim": compute_ssim(restored, clean),
    }

    return metrics


def save_result_figure(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    restored: torch.Tensor,
    metrics: Dict[str, float],
    save_path: str,
    title: str,
) -> None:
    """
    Save a comparison figure:
        clean | noisy | restored
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    clean_np = tensor_to_numpy_image(clean)
    noisy_np = tensor_to_numpy_image(noisy)
    restored_np = tensor_to_numpy_image(restored)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(clean_np)
    axes[0].axis("off")
    axes[0].set_title("Clean image")

    axes[1].imshow(noisy_np)
    axes[1].axis("off")
    axes[1].set_title(
        f"Noisy input\n"
        f"PSNR={metrics['noisy_psnr']:.2f}, "
        f"SSIM={metrics['noisy_ssim']:.3f}"
    )

    axes[2].imshow(restored_np)
    axes[2].axis("off")
    axes[2].set_title(
        f"Denoised output\n"
        f"PSNR={metrics['restored_psnr']:.2f}, "
        f"SSIM={metrics['restored_ssim']:.3f}"
    )

    fig.suptitle(
        f"{title} | "
        f"MSE: noisy={metrics['noisy_mse']:.5f}, "
        f"restored={metrics['restored_mse']:.5f}",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_summary_csv(
    summary: Dict[str, float],
    save_path: str,
) -> None:
    """
    Save overall test summary.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


def save_per_image_csv(
    rows: List[Dict[str, float]],
    save_path: str,
) -> None:
    """
    Save per-image metrics.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def load_checkpoint(
    checkpoint_path: str,
    model_name: str,
    device: torch.device,
):
    """
    Build model and load checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(
        model_name=model_name,
        in_channels=3,
        out_channels=3,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, checkpoint


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    eval_name: str,
    output_dir: str,
    num_save_images: int = 10,
):
    """
    Evaluate model on test set and save sample visualizations.
    """
    criterion = nn.MSELoss(reduction="mean")

    all_rows = []
    saved_count = 0

    total_restored_mse = 0.0
    total_noisy_mse = 0.0
    total_restored_psnr = 0.0
    total_noisy_psnr = 0.0
    total_restored_ssim = 0.0
    total_noisy_ssim = 0.0
    total_samples = 0

    result_image_dir = os.path.join(
        output_dir,
        "eval_results",
        eval_name,
        "images",
    )

    progress_bar = tqdm(test_loader, desc=f"Evaluating {eval_name}")

    for batch_idx, (noisy_images, clean_images) in enumerate(progress_bar):
        noisy_images = noisy_images.to(device, non_blocking=True)
        clean_images = clean_images.to(device, non_blocking=True)

        restored_images = model(noisy_images)
        restored_images = restored_images.clamp(0.0, 1.0)

        batch_loss = criterion(restored_images, clean_images).item()
        batch_size = noisy_images.size(0)

        for i in range(batch_size):
            clean = clean_images[i]
            noisy = noisy_images[i]
            restored = restored_images[i]

            metrics = compute_image_metrics(
                clean=clean,
                noisy=noisy,
                restored=restored,
            )

            sample_index = total_samples

            row = {
                "sample_index": sample_index,
                "batch_index": batch_idx,
                "noisy_mse": metrics["noisy_mse"],
                "restored_mse": metrics["restored_mse"],
                "noisy_psnr": metrics["noisy_psnr"],
                "restored_psnr": metrics["restored_psnr"],
                "noisy_ssim": metrics["noisy_ssim"],
                "restored_ssim": metrics["restored_ssim"],
            }

            all_rows.append(row)

            total_noisy_mse += metrics["noisy_mse"]
            total_restored_mse += metrics["restored_mse"]
            total_noisy_psnr += metrics["noisy_psnr"]
            total_restored_psnr += metrics["restored_psnr"]

            if not np.isnan(metrics["noisy_ssim"]):
                total_noisy_ssim += metrics["noisy_ssim"]
                total_restored_ssim += metrics["restored_ssim"]

            if saved_count < num_save_images:
                save_path = os.path.join(
                    result_image_dir,
                    f"sample_{saved_count:03d}.png",
                )

                save_result_figure(
                    clean=clean,
                    noisy=noisy,
                    restored=restored,
                    metrics=metrics,
                    save_path=save_path,
                    title=f"{eval_name} sample {saved_count}",
                )

                saved_count += 1

            total_samples += 1

        avg_restored_mse = total_restored_mse / max(total_samples, 1)
        progress_bar.set_postfix(
            {
                "batch_loss": f"{batch_loss:.5f}",
                "avg_mse": f"{avg_restored_mse:.5f}",
            }
        )

    summary = {
        "num_test_samples": total_samples,
        "avg_noisy_mse": total_noisy_mse / total_samples,
        "avg_restored_mse": total_restored_mse / total_samples,
        "avg_noisy_psnr": total_noisy_psnr / total_samples,
        "avg_restored_psnr": total_restored_psnr / total_samples,
        "avg_noisy_ssim": total_noisy_ssim / total_samples if HAS_SKIMAGE else float("nan"),
        "avg_restored_ssim": total_restored_ssim / total_samples if HAS_SKIMAGE else float("nan"),
        "saved_result_images": saved_count,
    }

    metrics_dir = os.path.join(output_dir, "eval_results", eval_name)
    summary_csv_path = os.path.join(metrics_dir, "summary.csv")
    per_image_csv_path = os.path.join(metrics_dir, "per_image_metrics.csv")

    save_summary_csv(summary, summary_csv_path)
    save_per_image_csv(all_rows, per_image_csv_path)

    print("=" * 80)
    print(f"Evaluation finished: {eval_name}")
    print("=" * 80)

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print(f"Summary saved to: {summary_csv_path}")
    print(f"Per-image metrics saved to: {per_image_csv_path}")
    print(f"Result images saved to: {result_image_dir}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate denoising model on test set."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["cnn", "autoencoder", "unet", "unet_residual"],
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default="./data/voc_images",
    )

    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--resize_size", type=int, default=160)

    parser.add_argument(
        "--noise_type",
        type=str,
        default="gaussian",
        choices=["gaussian", "salt_pepper"],
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.2,
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_save_images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name used for saving evaluation results.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.run_name is None:
        eval_name = (
            f"{args.model_name}"
            f"_voc"
            f"_size{args.image_size}"
            f"_{args.noise_type}"
            f"{args.noise_level}"
        )
    else:
        eval_name = args.run_name

    print("=" * 80)
    print("Evaluation configuration")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Image dir: {args.image_dir}")
    print(f"Noise: {args.noise_type}, level={args.noise_level}")
    print(f"Image size: {args.image_size}")
    print(f"Resize size: {args.resize_size}")
    print(f"Eval name: {eval_name}")
    print("=" * 80)

    model, checkpoint = load_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        device=device,
    )

    set_seed(args.seed)

    _, _, test_loader = build_dataloaders(
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

    set_seed(args.seed)

    evaluate(
        model=model,
        test_loader=test_loader,
        device=device,
        eval_name=eval_name,
        output_dir=args.output_dir,
        num_save_images=args.num_save_images,
    )


if __name__ == "__main__":
    main()