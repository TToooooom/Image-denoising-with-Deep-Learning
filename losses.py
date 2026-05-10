import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss.

    L = mean(sqrt((pred - target)^2 + eps^2))

    It can be regarded as a smooth L1 loss. It is commonly used in image
    restoration tasks because it is less over-smoothing than MSE and smoother
    than pure L1.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


def compute_restoration_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "charbonnier_mse",
    charbonnier_weight: float = 0.8,
    mse_weight: float = 0.2,
    l1_weight: float = 0.8,
) -> torch.Tensor:
    """
    Unified loss function for image restoration.

    Supported loss_type:
        - mse
        - l1
        - l1_mse
        - charbonnier
        - charbonnier_mse
    """
    if loss_type == "mse":
        return F.mse_loss(pred, target)

    if loss_type == "l1":
        return F.l1_loss(pred, target)

    if loss_type == "l1_mse":
        l1 = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)
        return l1_weight * l1 + mse_weight * mse

    if loss_type == "charbonnier":
        charbonnier = CharbonnierLoss()
        return charbonnier(pred, target)

    if loss_type == "charbonnier_mse":
        charbonnier = CharbonnierLoss()
        charbonnier_loss = charbonnier(pred, target)
        mse = F.mse_loss(pred, target)
        return charbonnier_weight * charbonnier_loss + mse_weight * mse

    raise ValueError(f"Unsupported loss_type: {loss_type}")