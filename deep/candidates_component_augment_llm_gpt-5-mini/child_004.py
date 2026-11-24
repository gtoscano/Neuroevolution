import torch
from torch import nn
import math

# Augmentation applied to every training batch tensor x of shape (N, 1, 28, 28).
# Uses only torch RNGs (seeded by outer experiment). Fast, CPU-friendly.

def augment(x: torch.Tensor) -> torch.Tensor:
    """Apply a small set of lightweight, label-preserving augmentations:
    - per-image intensity jitter (multiply + add)
    - with probability p_affine: small random affine (rotation ±10°, scale 0.95–1.05, translation ±2 px)
    - with probability p_erase: Random Erasing (area fraction 0.02–0.20, random noise fill)

    Input x is assumed normalized (mean 0, std 1). Returns a new tensor same shape as x.
    """
    # quick aliases
    F = torch.nn.functional

    if not isinstance(x, torch.Tensor):
        raise TypeError("augment expects a torch.Tensor")

    # Work on a clone to avoid unwanted in-place changes
    x = x.clone()
    N, C, H, W = x.shape
    device = x.device

    # 1) Intensity jitter (always): small multiplicative + additive change
    # Multiplicative scale in [0.90, 1.10], additive shift in [-0.08, 0.08]
    scales = torch.empty((N, 1, 1, 1), device=device).uniform_(0.90, 1.10)
    shifts = torch.empty((N, 1, 1, 1), device=device).uniform_(-0.08, 0.08)
    x = x * scales + shifts

    # 2) Small random affine (batched via affine_grid + grid_sample)
    p_affine = 0.5
    do_aff = torch.rand(N, device=device) < p_affine

    if do_aff.any():
        # Prepare theta for all images; identity for those without affine
        theta = torch.zeros((N, 2, 3), dtype=torch.float32, device=device)
        # default identity
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = 0.0
        theta[:, 1, 2] = 0.0

        # parameters for affected images
        idx = torch.nonzero(do_aff, as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            # angles in radians between -10 and 10 deg
            angles = torch.empty(idx.numel(), device=device).uniform_(-10.0, 10.0) * (math.pi / 180.0)
            scales_aff = torch.empty(idx.numel(), device=device).uniform_(0.95, 1.05)
            # translations in pixels between -2 and 2
            tx_px = torch.empty(idx.numel(), device=device).uniform_(-2.0, 2.0)
            ty_px = torch.empty(idx.numel(), device=device).uniform_(-2.0, 2.0)

            cos = torch.cos(angles) * scales_aff
            sin = torch.sin(angles) * scales_aff

            # normalized translation for affine_grid: tx_norm = 2 * tx_px / (W-1)
            tx = 2.0 * tx_px / float(max(W - 1, 1))
            ty = 2.0 * ty_px / float(max(H - 1, 1))

            # Fill theta for these indices
            theta[idx, 0, 0] = cos
            theta[idx, 0, 1] = -sin
            theta[idx, 1, 0] = sin
            theta[idx, 1, 1] = cos
            theta[idx, 0, 2] = tx
            theta[idx, 1, 2] = ty

        # Create grid and apply sampling
        # Note: align_corners=False is a safe default for small images
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # 3) Random Erasing (per-image): preserve labels, fill with small gaussian noise
    p_erase = 0.5
    do_erase = torch.rand(N, device=device) < p_erase

    if do_erase.any():
        area = float(H * W)
        for i in torch.nonzero(do_erase, as_tuple=False).squeeze(1).tolist():
            # Try a few times to find a valid rectangle
            for _ in range(10):
                target_area = area * float(torch.empty(1, device=device).uniform_(0.02, 0.20).item())
                aspect = float(torch.empty(1, device=device).uniform_(0.3, 3.3).item())
                h = int(round(math.sqrt(target_area * aspect)))
                w = int(round(math.sqrt(target_area / aspect)))
                if 0 < h < H and 0 < w < W:
                    top = int(torch.empty(1, device=device).uniform_(0, H - h + 1).item())
                    left = int(torch.empty(1, device=device).uniform_(0, W - w + 1).item())
                    # Fill with small gaussian noise (std 0.1) so we don't produce extreme outliers
                    noise = torch.randn((C, h, w), device=device) * 0.10
                    x[i, :, top:top + h, left:left + w] = noise
                    break
            # if no valid rectangle found, skip erasing for this image

    return x
