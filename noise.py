import torch


def add_gaussian_noise(
    image: torch.Tensor,
    sigma: float = 0.2,
    clamp: bool = True,
) -> torch.Tensor:
    """
    Add Gaussian noise to an image tensor.

    Args:
        image: Tensor with shape [C, H, W], values in [0, 1].
        sigma: Standard deviation of Gaussian noise.
        clamp: Whether to clamp noisy image to [0, 1].

    Returns:
        Noisy image tensor with the same shape as input.
    """
    noise = torch.randn_like(image) * sigma
    noisy_image = image + noise

    if clamp:
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)

    return noisy_image


def add_salt_pepper_noise(
    image: torch.Tensor,
    amount: float = 0.1,
) -> torch.Tensor:
    """
    Add salt-and-pepper noise to an image tensor.

    Args:
        image: Tensor with shape [C, H, W], values in [0, 1].
        amount: Probability of pixels being corrupted.

    Returns:
        Noisy image tensor with the same shape as input.
    """
    noisy_image = image.clone()

    # random mask with same shape as image
    random_values = torch.rand_like(image)

    salt_mask = random_values < (amount / 2)
    pepper_mask = (random_values >= (amount / 2)) & (random_values < amount)

    noisy_image[salt_mask] = 1.0
    noisy_image[pepper_mask] = 0.0

    return noisy_image


def add_noise(
    image: torch.Tensor,
    noise_type: str = "gaussian",
    noise_level: float = 0.2,
) -> torch.Tensor:
    """
    Unified noise interface.

    Args:
        image: Tensor with shape [C, H, W], values in [0, 1].
        noise_type: "gaussian" or "salt_pepper".
        noise_level:
            - for Gaussian noise: sigma
            - for salt-and-pepper noise: corruption probability

    Returns:
        Noisy image tensor.
    """
    if noise_type == "gaussian":
        return add_gaussian_noise(image, sigma=noise_level)

    if noise_type == "salt_pepper":
        return add_salt_pepper_noise(image, amount=noise_level)

    raise ValueError(f"Unsupported noise_type: {noise_type}")