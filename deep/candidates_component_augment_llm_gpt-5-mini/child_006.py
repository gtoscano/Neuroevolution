import torch
from torch import nn
import math


def augment(x: torch.Tensor) -> torch.Tensor:
    """
    Per-image random integer translation (pixel shifts) applied to every training batch.

    Behavior (one substantive change vs parent):
    - Instead of performing a single square cutout, this augmentation randomly shifts each image by
      an integer offset in x and y independently, sampled uniformly from [-max_shift, max_shift].
      The empty regions introduced by the shift are filled with 0.0 (the normalized image mean ~ 0).

    Args:
        x: tensor of shape (N, 1, 28, 28), normalized (mean 0, std 1).
    Returns:
        tensor of same shape with each image randomly translated by up to ±max_shift pixels.

    Notes:
    - Uses only torch RNG so results are seeded by the training pipeline.
    - Implemented to avoid wrap-around (no torched roll with circular wrap).
    """
    if x.ndim != 4:
        raise ValueError(f"expected input of shape (N,1,28,28), got {x.shape}")

    N, C, H, W = x.shape
    # maximum integer pixel shift in each direction (inclusive)
    max_shift = 3  # shifts in [-3, 3]

    # Prepare output (defensive copy)
    out = torch.empty_like(x)

    # Sample integer shifts for each image using torch RNG
    # torch.randint requires non-negative low, so sample in [0, 2*max_shift+1) and subtract max_shift
    if max_shift <= 0:
        return x.clone()

    span = 2 * max_shift + 1
    shifts = torch.randint(0, span, (N, 2), device=x.device) - max_shift
    # shifts[:, 0] -> dy (vertical), shifts[:, 1] -> dx (horizontal)

    # For each image, copy a cropped window from the source into a zeroed target at the shifted location
    for i in range(N):
        dy = int(shifts[i, 0].item())
        dx = int(shifts[i, 1].item())

        src = x[i]  # shape (C, H, W)
        dst = torch.zeros_like(src)

        # Compute source and destination slice indices for height
        if dy >= 0:
            src_y0 = 0
            dst_y0 = dy
            h_src = H - dy
        else:
            src_y0 = -dy
            dst_y0 = 0
            h_src = H + dy  # because dy is negative

        # Compute source and destination slice indices for width
        if dx >= 0:
            src_x0 = 0
            dst_x0 = dx
            w_src = W - dx
        else:
            src_x0 = -dx
            dst_x0 = 0
            w_src = W + dx

        # If by any chance the computed source size is non-positive (shouldn't happen), skip copy
        if h_src > 0 and w_src > 0:
            dst[:, dst_y0:dst_y0 + h_src, dst_x0:dst_x0 + w_src] = \
                src[:, src_y0:src_y0 + h_src, src_x0:src_x0 + w_src]

        out[i] = dst

    return out
