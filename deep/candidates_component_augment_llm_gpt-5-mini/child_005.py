import torch
from torch import nn
import math


def augment(x: torch.Tensor) -> torch.Tensor:
    """
    Per-image random erasing (single square cutout) applied to every training batch.

    Args:
        x: tensor of shape (N, 1, 28, 28), normalized (mean 0, std 1).
    Returns:
        tensor of same shape with random square regions zeroed out in some images.

    Behavior (one substantive change vs parent):
    - For each image independently, with probability p=0.5, choose a square size s ~ Uniform(4,10)
      and a random location; set that square region to 0 (the normalized image mean is ~0).

    This uses only torch RNG so results are seeded by the training pipeline.
    """
    # Defensive copy so we don't modify caller's tensor in-place
    out = x.clone()
    if out.ndim != 4:
        raise ValueError(f"expected input of shape (N,1,28,28), got {out.shape}")

    N, C, H, W = out.shape
    # Parameters
    p = 0.5  # probability to apply cutout per image
    min_size = 4
    max_size = 10  # inclusive upper bound

    # Decide which images get cutout
    probs = torch.rand(N, device=out.device)
    apply = probs < p
    if not apply.any():
        return out

    # For each selected image, sample a size and top-left corner, then zero the patch
    ids = torch.nonzero(apply, as_tuple=False).squeeze(1)
    for idx in ids.tolist():
        # sample size s in [min_size, max_size]
        s = int(torch.randint(min_size, max_size + 1, (1,), device=out.device).item())
        # choose top-left so the square fits entirely
        if W - s <= 0:
            x0 = 0
        else:
            x0 = int(torch.randint(0, W - s + 1, (1,), device=out.device).item())
        if H - s <= 0:
            y0 = 0
        else:
            y0 = int(torch.randint(0, H - s + 1, (1,), device=out.device).item())
        out[idx, :, y0 : y0 + s, x0 : x0 + s] = 0.0

    return out
