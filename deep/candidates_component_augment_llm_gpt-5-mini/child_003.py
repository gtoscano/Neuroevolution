import torch
from torch import nn
import math


def augment(x: torch.Tensor) -> torch.Tensor:
    """
    Batch augmentation for FashionMNIST-like images.

    Operations (applied in this order):
    - random horizontal flip (per image, p=0.5)
    - per-image contrast (0.8-1.2) and brightness (-0.15..0.15) jitter
    - additive Gaussian noise (per-image std 0..0.12)
    - random square cutout (per-image p=0.5, size 4..8 px)

    Input: x tensor shaped (N,1,28,28), already normalized (mean 0, std 1).
    Returns tensor of same shape and dtype.
    Uses torch random functions so behavior is reproducible with torch.manual_seed.
    This is a lightweight, vectorized-first implementation; cutout uses a small Python loop
    only for the selected examples (fast enough for batch size 64 on CPU).
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.dim() != 4:
        raise ValueError("x must have shape (N, C, H, W)")

    # Work on a copy to avoid in-place surprises
    x_out = x.clone()
    N, C, H, W = x_out.shape
    device = x_out.device
    dtype = x_out.dtype

    # 1) Random horizontal flip
    if N > 0:
        flip_mask = torch.rand(N, device=device) < 0.5
        if flip_mask.any():
            # flip last dimension (width)
            x_out[flip_mask] = x_out[flip_mask].flip(-1)

    # 2) Per-image contrast and brightness
    # contrast in [0.8, 1.2], brightness shift in [-0.15, 0.15]
    contrast = torch.empty(N, device=device).uniform_(0.8, 1.2).to(dtype)
    brightness = torch.empty(N, device=device).uniform_(-0.15, 0.15).to(dtype)
    contrast = contrast.view(N, 1, 1, 1)
    brightness = brightness.view(N, 1, 1, 1)
    x_out = x_out * contrast + brightness

    # 3) Per-image Gaussian noise
    noise_std = torch.empty(N, device=device).uniform_(0.0, 0.12).to(dtype)
    noise = torch.randn_like(x_out) * noise_std.view(N, 1, 1, 1)
    x_out = x_out + noise

    # 4) Random square cutout (set to zero) per image with p=0.5
    cutout_mask = torch.rand(N, device=device) < 0.5
    if cutout_mask.any():
        # We'll loop only over selected indices; batch size small so this is fine.
        selected = torch.nonzero(cutout_mask, as_tuple=False).view(-1).tolist()
        for i in selected:
            # size between 4 and 8 (inclusive)
            s = int(torch.randint(4, 9, (1,), device=device).item())
            # choose center uniformly
            cy = int(torch.randint(0, H, (1,), device=device).item())
            cx = int(torch.randint(0, W, (1,), device=device).item())
            y1 = max(0, cy - s // 2)
            x1 = max(0, cx - s // 2)
            y2 = min(H, y1 + s)
            x2 = min(W, x1 + s)
            x_out[i, :, y1:y2, x1:x2] = 0.0

    return x_out
