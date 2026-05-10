import argparse
import os
import subprocess
import sys
from datetime import datetime


def run_command(command, log_path: str) -> None:
    """
    Run command and save stdout/stderr to log file.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print("=" * 100)
    print("Running command:")
    print(" ".join(command))
    print(f"Log file: {log_path}")
    print("=" * 100)

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"Command failed with return code {return_code}")


def build_run_name(
    model_name: str,
    args: argparse.Namespace,
) -> str:
    return (
        f"{model_name}"
        f"_voc"
        f"_size{args.image_size}"
        f"_{args.noise_type}"
        f"{args.noise_level}"
        f"_{args.loss_type}"
    )


def build_train_command(
    args: argparse.Namespace,
    model_name: str,
):
    command = [
        sys.executable,
        "train.py",
        "--model_name", model_name,
        "--image_dir", args.image_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--image_size", str(args.image_size),
        "--resize_size", str(args.resize_size),
        "--noise_type", args.noise_type,
        "--noise_level", str(args.noise_level),
        "--loss_type", args.loss_type,
        "--l1_weight", str(args.l1_weight),
        "--mse_weight", str(args.mse_weight),
        "--charbonnier_weight", str(args.charbonnier_weight),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--device", args.device,
        "--val_every", str(args.val_every),
        "--checkpoint_dir", args.checkpoint_dir,
        "--output_dir", args.output_dir,
    ]

    return command


def build_eval_command(
    args: argparse.Namespace,
    model_name: str,
):
    run_name = build_run_name(model_name, args)

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"{run_name}_best.pth",
    )

    command = [
        sys.executable,
        "evaluate.py",
        "--model_name", model_name,
        "--checkpoint_path", checkpoint_path,
        "--image_dir", args.image_dir,
        "--image_size", str(args.image_size),
        "--resize_size", str(args.resize_size),
        "--noise_type", args.noise_type,
        "--noise_level", str(args.noise_level),
        "--batch_size", str(args.eval_batch_size),
        "--num_workers", str(args.num_workers),
        "--num_save_images", str(args.num_save_images),
        "--seed", str(args.seed),
        "--device", args.device,
        "--output_dir", args.output_dir,
        "--run_name", run_name,
    ]

    return command


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CNN, AutoEncoder, U-Net, and U-Net Residual training sequentially."
    )

    parser.add_argument("--image_dir", type=str, default="./data/voc_images")

    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--resize_size", type=int, default=160)

    parser.add_argument(
        "--noise_type",
        type=str,
        default="gaussian",
        choices=["gaussian", "salt_pepper"],
    )
    parser.add_argument("--noise_level", type=float, default=0.2)

    parser.add_argument(
        "--loss_type",
        type=str,
        default="charbonnier_mse",
        choices=["mse", "l1", "l1_mse", "charbonnier", "charbonnier_mse"],
    )
    parser.add_argument("--l1_weight", type=float, default=0.8)
    parser.add_argument("--mse_weight", type=float, default=0.2)
    parser.add_argument("--charbonnier_weight", type=float, default=0.8)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)

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
        help="Run validation every N epochs.",
    )

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--output_dir", type=str, default="./outputs")

    parser.add_argument(
        "--run_eval",
        action="store_true",
        help="Run evaluate.py after each model is trained.",
    )

    parser.add_argument(
        "--num_save_images",
        type=int,
        default=10,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    experiment_plan = [
        "cnn",
        "autoencoder",
        "unet",
        "unet_residual",
    ]

    print("=" * 100)
    print("Sequential experiment plan")
    print("=" * 100)
    print(f"Image dir: {args.image_dir}")
    print(f"Noise: {args.noise_type}, level={args.noise_level}")
    print(f"Loss type: {args.loss_type}")
    print(f"Image size: {args.image_size}")
    print(f"Resize size: {args.resize_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs per model: {args.epochs}")
    print(f"Validation every: {args.val_every} epoch(s)")
    print(f"Run evaluation: {args.run_eval}")
    print("Models:")

    for model_name in experiment_plan:
        print(f"  - {model_name}")

    print("=" * 100)

    for model_name in experiment_plan:
        train_command = build_train_command(
            args=args,
            model_name=model_name,
        )

        train_log_path = os.path.join(
            log_dir,
            f"{model_name}_train.log",
        )

        run_command(train_command, train_log_path)

        if args.run_eval:
            eval_command = build_eval_command(
                args=args,
                model_name=model_name,
            )

            eval_log_path = os.path.join(
                log_dir,
                f"{model_name}_eval.log",
            )

            run_command(eval_command, eval_log_path)

    print("=" * 100)
    print("All experiments finished.")
    print(f"Logs saved to: {log_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()