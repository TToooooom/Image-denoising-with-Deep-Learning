import glob
import math
import os
from functools import lru_cache
from typing import List, Tuple

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torchvision.transforms.functional import to_pil_image, to_tensor

from models import build_model
from noise import add_noise


try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


MODEL_NAMES = ["cnn", "autoencoder", "unet", "unet_residual"]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_checkpoints() -> List[str]:
    paths = sorted(glob.glob("./checkpoints/*.pth"))

    def score(path: str) -> int:
        name = os.path.basename(path).lower()

        if "unet_voc_size256_gaussian0.1" in name:
            return 0
        if "unet_residual_voc_size256_gaussian0.1" in name:
            return 1
        if "unet_gan" in name:
            return 2
        if "unet_voc_size128_gaussian0.1" in name:
            return 3
        if "unet_residual" in name:
            return 4
        if "unet" in name:
            return 5
        if "autoencoder" in name:
            return 6
        if "cnn" in name:
            return 7
        return 99

    return sorted(paths, key=score)


def infer_model_name_from_checkpoint(checkpoint: dict, checkpoint_path: str) -> str:
    """
    Infer model architecture from checkpoint metadata or filename.

    GAN fine-tuned checkpoint still uses ordinary U-Net as generator.
    Therefore its model_name should be "unet".
    """
    if isinstance(checkpoint, dict):
        model_name = checkpoint.get("model_name", None)
        if model_name in MODEL_NAMES:
            return model_name

    filename = os.path.basename(checkpoint_path).lower()

    if "unet_gan" in filename:
        return "unet"
    if "unet_residual" in filename:
        return "unet_residual"
    if "autoencoder" in filename:
        return "autoencoder"
    if "cnn" in filename:
        return "cnn"
    if "unet" in filename:
        return "unet"

    raise ValueError(f"Cannot infer model name from checkpoint: {checkpoint_path}")


@lru_cache(maxsize=8)
def load_model_cached(checkpoint_path: str, device_name: str):
    device = torch.device(device_name)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = infer_model_name_from_checkpoint(checkpoint, checkpoint_path)

    model = build_model(
        model_name=model_name,
        in_channels=3,
        out_channels=3,
    ).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, model_name


def pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor, dict]:
    """
    Pad tensor [B, C, H, W] so that H and W are divisible by multiple.
    """
    if x.dim() != 4:
        raise ValueError(f"Expected [B, C, H, W], got {x.shape}")

    _, _, h, w = x.shape

    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    padded = F.pad(
        x,
        pad=(0, pad_w, 0, pad_h),
        mode="replicate",
    )

    info = {
        "original_h": h,
        "original_w": w,
    }

    return padded, info


def crop_to_original(x: torch.Tensor, info: dict) -> torch.Tensor:
    h = info["original_h"]
    w = info["original_w"]
    return x[:, :, :h, :w]


def maybe_resize_image(image: Image.Image, max_side: int) -> Image.Image:
    """
    If max_side <= 0, keep original size.
    Otherwise resize image while keeping aspect ratio.
    """
    image = image.convert("RGB")

    if max_side <= 0:
        return image

    w, h = image.size
    current_max = max(w, h)

    if current_max <= max_side:
        return image

    scale = max_side / current_max
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    return image.resize((new_w, new_h), Image.BICUBIC)


def tensor_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def tensor_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    if not HAS_SKIMAGE:
        return float("nan")

    pred_np = pred.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    target_np = target.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

    return float(
        ssim(
            target_np,
            pred_np,
            channel_axis=2,
            data_range=1.0,
        )
    )


def apply_sharpen(image: Image.Image, strength: float) -> Image.Image:
    """
    Optional visual post-processing.
    strength=0 means disabled.
    """
    if strength <= 0:
        return image

    percent = int(80 + strength * 120)
    return image.filter(
        ImageFilter.UnsharpMask(
            radius=1.2,
            percent=percent,
            threshold=2,
        )
    )


def apply_gaussian_filter(image: Image.Image, radius: float) -> Image.Image:
    """
    Traditional Gaussian filtering baseline.

    radius controls smoothing strength:
        0.3-0.6: weak smoothing
        0.8-1.2: moderate smoothing
        1.5-2.5: strong smoothing
    """
    image = image.convert("RGB")

    if radius <= 0:
        return image

    return image.filter(ImageFilter.GaussianBlur(radius=float(radius)))


def make_comparison(images: List[Image.Image], labels: List[str]) -> Image.Image:
    """
    Concatenate images horizontally with labels.
    """
    images = [img.convert("RGB") for img in images]
    widths = [img.size[0] for img in images]
    heights = [img.size[1] for img in images]

    label_h = 34
    out_w = sum(widths)
    out_h = max(heights) + label_h

    canvas = Image.new("RGB", (out_w, out_h), "white")
    draw = ImageDraw.Draw(canvas)

    x_offset = 0
    for img, label in zip(images, labels):
        canvas.paste(img, (x_offset, label_h))
        draw.text((x_offset + 8, 8), label, fill=(0, 0, 0))
        x_offset += img.size[0]

    return canvas


def prepare_input_image(
    input_image: Image.Image,
    mode: str,
    noise_type: str,
    noise_level: float,
    seed: int,
    max_side: int,
):
    """
    Prepare image tensors and PIL images for both deep learning and Gaussian filter methods.
    """
    image = maybe_resize_image(input_image, max_side=max_side)
    clean_tensor = to_tensor(image).float().clamp(0.0, 1.0)

    if mode == "对干净图像加噪后去噪":
        torch.manual_seed(int(seed))
        noisy_tensor = add_noise(
            clean_tensor,
            noise_type=noise_type,
            noise_level=float(noise_level),
        ).clamp(0.0, 1.0)
        has_clean_target = True
    else:
        noisy_tensor = clean_tensor.clone()
        has_clean_target = False

    noisy_pil = to_pil_image(noisy_tensor)

    return image, clean_tensor, noisy_tensor, noisy_pil, has_clean_target


def run_deep_model(
    noisy_tensor: torch.Tensor,
    checkpoint_path: str,
):
    """
    Run trained deep denoising model.
    """
    if checkpoint_path is None or checkpoint_path == "":
        raise gr.Error("没有找到 checkpoint。使用深度学习模型时，请先训练模型并选择 checkpoint。")

    device = get_device()
    device_name = str(device)

    model, model_name = load_model_cached(checkpoint_path, device_name)

    x = noisy_tensor.unsqueeze(0).to(device)
    x_pad, pad_info = pad_to_multiple(x, multiple=16)

    with torch.no_grad():
        restored = model(x_pad)
        restored = crop_to_original(restored, pad_info)
        restored = restored.clamp(0.0, 1.0)

    restored_tensor = restored.squeeze(0).detach().cpu()
    restored_pil = to_pil_image(restored_tensor)

    return restored_tensor, restored_pil, model_name


def run_gaussian_filter(
    noisy_pil: Image.Image,
    filter_radius: float,
):
    """
    Run traditional Gaussian filter.
    """
    filtered_pil = apply_gaussian_filter(noisy_pil, radius=float(filter_radius))
    filtered_tensor = to_tensor(filtered_pil).float().clamp(0.0, 1.0)

    return filtered_tensor, filtered_pil


def build_metrics_text(
    method: str,
    mode: str,
    image: Image.Image,
    noise_type: str,
    noise_level: float,
    clean_tensor: torch.Tensor,
    noisy_tensor: torch.Tensor,
    restored_tensor: torch.Tensor,
    has_clean_target: bool,
    checkpoint_path: str = "",
    model_name: str = "",
    filter_radius: float = 0.0,
    sharpen_strength: float = 0.0,
):
    """
    Build metric and method information text.
    """
    if method == "深度学习模型去噪":
        checkpoint_name = os.path.basename(checkpoint_path) if checkpoint_path else "None"

        metrics_text = (
            f"Method: 深度学习模型去噪\n"
            f"Model architecture: {model_name}\n"
            f"Checkpoint: {checkpoint_name}\n"
            f"Mode: {mode}\n"
            f"Image size used: {image.size[0]} x {image.size[1]}\n"
            f"Noise: {noise_type}, level={noise_level}\n"
        )
    else:
        metrics_text = (
            f"Method: 高斯滤波传统去噪\n"
            f"Mode: {mode}\n"
            f"Image size used: {image.size[0]} x {image.size[1]}\n"
            f"Noise: {noise_type}, level={noise_level}\n"
            f"Gaussian filter radius: {filter_radius}\n"
        )

    if has_clean_target:
        noisy_psnr = tensor_psnr(noisy_tensor, clean_tensor)
        restored_psnr = tensor_psnr(restored_tensor, clean_tensor)

        noisy_ssim = tensor_ssim(noisy_tensor, clean_tensor)
        restored_ssim = tensor_ssim(restored_tensor, clean_tensor)

        metrics_text += (
            f"\n"
            f"Noisy PSNR: {noisy_psnr:.2f}\n"
            f"Restored PSNR: {restored_psnr:.2f}\n"
        )

        if HAS_SKIMAGE:
            metrics_text += (
                f"Noisy SSIM: {noisy_ssim:.4f}\n"
                f"Restored SSIM: {restored_ssim:.4f}\n"
            )
        else:
            metrics_text += "SSIM unavailable: scikit-image not installed.\n"
    else:
        metrics_text += "\nNo clean target is available, so PSNR/SSIM are not computed.\n"

    if sharpen_strength > 0 and method == "深度学习模型去噪":
        metrics_text += (
            "\nNote: 输出图像启用了轻微锐化后处理；"
            "指标基于模型原始输出计算，不基于锐化结果。\n"
        )

    return metrics_text


def denoise_demo(
    input_image: Image.Image,
    method: str,
    checkpoint_path: str,
    mode: str,
    noise_type: str,
    noise_level: float,
    seed: int,
    max_side: int,
    sharpen_strength: float,
    gaussian_radius: float,
):
    """
    Unified denoising demo:
        - Deep learning model
        - Traditional Gaussian filter
    """
    if input_image is None:
        raise gr.Error("请先上传图片。")

    image, clean_tensor, noisy_tensor, noisy_pil, has_clean_target = prepare_input_image(
        input_image=input_image,
        mode=mode,
        noise_type=noise_type,
        noise_level=noise_level,
        seed=seed,
        max_side=max_side,
    )

    if method == "深度学习模型去噪":
        restored_tensor, restored_pil_raw, model_name = run_deep_model(
            noisy_tensor=noisy_tensor,
            checkpoint_path=checkpoint_path,
        )

        restored_pil = apply_sharpen(restored_pil_raw, sharpen_strength)

        metrics_text = build_metrics_text(
            method=method,
            mode=mode,
            image=image,
            noise_type=noise_type,
            noise_level=noise_level,
            clean_tensor=clean_tensor,
            noisy_tensor=noisy_tensor,
            restored_tensor=restored_tensor,
            has_clean_target=has_clean_target,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            filter_radius=gaussian_radius,
            sharpen_strength=sharpen_strength,
        )

        output_label = "Denoised output"

    else:
        restored_tensor, restored_pil = run_gaussian_filter(
            noisy_pil=noisy_pil,
            filter_radius=gaussian_radius,
        )

        metrics_text = build_metrics_text(
            method=method,
            mode=mode,
            image=image,
            noise_type=noise_type,
            noise_level=noise_level,
            clean_tensor=clean_tensor,
            noisy_tensor=noisy_tensor,
            restored_tensor=restored_tensor,
            has_clean_target=has_clean_target,
            checkpoint_path="",
            model_name="Gaussian Filter",
            filter_radius=gaussian_radius,
            sharpen_strength=0.0,
        )

        output_label = "Gaussian filtered"

    if has_clean_target:
        comparison = make_comparison(
            images=[image, noisy_pil, restored_pil],
            labels=["Clean", "Noisy input", output_label],
        )
    else:
        comparison = make_comparison(
            images=[image, restored_pil],
            labels=["Input image", output_label],
        )

    return noisy_pil, restored_pil, comparison, metrics_text


def build_app():
    checkpoints = find_checkpoints()

    with gr.Blocks(title="Noisy2Clean 图像去噪系统") as demo:
        gr.Markdown(
            """
            # Noisy2Clean 图像去噪与恢复系统

            本页面支持两类方法：

            1. **深度学习模型去噪**：CNN / AutoEncoder / U-Net / U-Net Residual / U-Net-GAN checkpoint  
            2. **高斯滤波传统去噪**：无需训练，通过高斯核进行局部平滑

            高斯滤波能降低随机噪声，但通常会模糊边缘和纹理。
            """
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="上传图像",
                    type="pil",
                )

                method = gr.Radio(
                    choices=[
                        "深度学习模型去噪",
                        "高斯滤波传统去噪",
                    ],
                    value="深度学习模型去噪",
                    label="方法选择",
                )

                checkpoint_dropdown = gr.Dropdown(
                    choices=checkpoints,
                    value=checkpoints[0] if len(checkpoints) > 0 else None,
                    label="选择模型 checkpoint（高斯滤波不需要）",
                )

                mode = gr.Radio(
                    choices=[
                        "对干净图像加噪后去噪",
                        "直接去噪上传图像",
                    ],
                    value="对干净图像加噪后去噪",
                    label="运行模式",
                )

                noise_type = gr.Dropdown(
                    choices=["gaussian", "salt_pepper"],
                    value="gaussian",
                    label="噪声类型",
                )

                noise_level = gr.Slider(
                    minimum=0.0,
                    maximum=0.3,
                    value=0.1,
                    step=0.01,
                    label="噪声强度：Gaussian 下等于 sigma，椒盐噪声下等于污染比例",
                )

                seed = gr.Number(
                    value=42,
                    precision=0,
                    label="随机种子",
                )

                max_side = gr.Slider(
                    minimum=0,
                    maximum=1024,
                    value=0,
                    step=64,
                    label="最大边限制：0 表示保持原图尺寸",
                )

                sharpen_strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.1,
                    label="深度学习输出的可选锐化后处理：0 表示关闭",
                )

                gaussian_radius = gr.Slider(
                    minimum=0.0,
                    maximum=3.0,
                    value=0.8,
                    step=0.1,
                    label="高斯滤波半径 radius（仅高斯滤波方法使用）",
                )

                run_button = gr.Button("运行去噪", variant="primary")

            with gr.Column():
                noisy_output = gr.Image(
                    label="Noisy / Input",
                    type="pil",
                )

                restored_output = gr.Image(
                    label="Denoised / Filtered Output",
                    type="pil",
                )

                comparison_output = gr.Image(
                    label="Comparison",
                    type="pil",
                )

                metrics_output = gr.Textbox(
                    label="指标与方法信息",
                    lines=14,
                )

        run_button.click(
            fn=denoise_demo,
            inputs=[
                input_image,
                method,
                checkpoint_dropdown,
                mode,
                noise_type,
                noise_level,
                seed,
                max_side,
                sharpen_strength,
                gaussian_radius,
            ],
            outputs=[
                noisy_output,
                restored_output,
                comparison_output,
                metrics_output,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860)