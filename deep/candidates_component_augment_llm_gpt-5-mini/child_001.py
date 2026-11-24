import torch
from torch import nn
import math

# Augment function applied to every training batch x (N,1,28,28), normalized (mean 0, std 1).
# Uses only torch and torch.nn.functional (via nn.functional). Fast, vectorized where possible.

def augment(x: torch.Tensor) -> torch.Tensor:
    """
    Apply small random affine transforms, random horizontal flip, random cutout, and small Gaussian noise.
    All randomness uses torch.* so it's controlled by the global seed.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")

    device = x.device
    dtype = x.dtype
    N, C, H, W = x.shape

    # Parameters (kept modest to avoid making task harder)
    max_deg = 15.0            # rotation range in degrees
    max_trans_px = 3.0        # translation in pixels
    min_scale = 0.95
    max_scale = 1.05
    flip_prob = 0.5
    cutout_prob = 0.5
    cutout_min = 4            # pixels
    cutout_max = 10           # pixels (exclusive upper for randint)
    noise_std = 0.02

    # --- Affine transform (vectorized over batch) ---
    # Sample parameters per-image
    angles = (torch.rand(N, device=device, dtype=dtype) * (2.0 * max_deg) - max_deg)  # degrees
    angles_rad = angles * (math.pi / 180.0)
    scales = torch.rand(N, device=device, dtype=dtype) * (max_scale - min_scale) + min_scale

    tx_pixels = (torch.rand(N, device=device, dtype=dtype) * (2.0 * max_trans_px) - max_trans_px)
    ty_pixels = (torch.rand(N, device=device, dtype=dtype) * (2.0 * max_trans_px) - max_trans_px)

    # Convert pixel translations to normalized grid coordinates.
    # For affine_grid with align_corners=False, one pixel corresponds to 2/W in normalized coords approximately.
    tx = 2.0 * tx_pixels / float(W)
    ty = 2.0 * ty_pixels / float(H)

    # Build affine matrices: [ [s*cos, -s*sin, tx], [s*sin, s*cos, ty] ]
    cos = torch.cos(angles_rad) * scales
    sin = torch.sin(angles_rad) * scales

    theta = torch.zeros((N, 2, 3), device=device, dtype=dtype)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty

    # Create grid and sample
    # Use nn.functional via imported nn; keep align_corners=False for stability on small images
    grid = nn.functional.affine_grid(theta, x.size(), align_corners=False)
    x = nn.functional.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

    # --- Random horizontal flip per image ---
    if flip_prob > 0.0:
        flips = torch.rand(N, device=device) < float(flip_prob)
        if flips.any():
            # flip along width axis (last dim)
            x[flips] = torch.flip(x[flips], dims=[-1])

    # --- Random cutout (small square erase) ---
    if cutout_prob > 0.0:
        do_cutout = torch.rand(N, device=device) < float(cutout_prob)
        for i in range(N):
            if not do_cutout[i]:
                continue
            size = int(torch.randint(cutout_min, cutout_max, (1,), device=device).item())
            # center coordinates in pixel indices
            cy = int(torch.randint(0, H, (1,), device=device).item())
            cx = int(torch.randint(0, W, (1,), device=device).item())
            y0 = max(0, cy - size // 2)
            y1 = min(H, y0 + size)
            x0 = max(0, cx - size // 2)
            x1 = min(W, x0 + size)
            # set to zero (normalized mean is 0, so zeroing is a neutral occlusion)
            x[i, :, y0:y1, x0:x1] = 0.0

    # --- Small additive Gaussian noise ---
    if noise_std > 0.0:
        x = x + torch.randn_like(x) * float(noise_std)

    return x
