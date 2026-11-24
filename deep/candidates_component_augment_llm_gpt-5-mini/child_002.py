import torch
from torch import nn
import math


def augment(x: torch.Tensor) -> torch.Tensor:
    """
    Fast, batch-friendly augmentation for (N,1,28,28) normalized tensors.

    Pipeline (per-image, independent):
    - 50% horizontal flip
    - random integer translation in x,y in [-3,3] via torch.roll
    - 30% mild brightness scale in [0.9, 1.1]
    - 50% random cutout (erase) of a square of size uniformly in [6,10]

    All operations are implemented with PyTorch ops; loops are only over the batch
    dimension for a few cheap in-place rolls/erases (batch size is small: 64).
    Returns a tensor of the same shape and dtype as the input.
    """
    # Expect x shape (N, 1, 28, 28)
    if not isinstance(x, torch.Tensor):
        raise TypeError("augment: expected a torch.Tensor")
    if x.dim() != 4:
        raise ValueError("augment: expected input of shape (N,1,28,28)")

    N, C, H, W = x.shape
    device = x.device

    # Work on a copy to avoid modifying caller's tensor in-place
    out = x.clone()

    # 1) Horizontal flip with p=0.5
    flip_mask = torch.rand(N, device=device) < 0.5
    if flip_mask.any().item():
        out[flip_mask] = out[flip_mask].flip(-1)

    # 2) Small integer translations via torch.roll (per-image)
    max_shift = 3
    # sample in { -max_shift, ..., +max_shift }
    shifts = torch.randint(0, 2 * max_shift + 1, (N, 2), device=device) - max_shift
    # shifts[:,0] -> dx (width), shifts[:,1] -> dy (height)
    for i in range(N):
        dx = int(shifts[i, 0].item())
        dy = int(shifts[i, 1].item())
        if dx == 0 and dy == 0:
            continue
        # roll expects shifts in order of dims (-2, -1) == (height, width)
        out[i] = torch.roll(out[i], shifts=(dy, dx), dims=(-2, -1))

    # 3) Mild brightness jitter with p=0.3
    bright_mask = torch.rand(N, device=device) < 0.3
    if bright_mask.any().item():
        factors = torch.empty(N, device=device).uniform_(0.9, 1.1)
        out[bright_mask] = out[bright_mask] * factors.view(-1, 1, 1, 1)

    # 4) Random cutout / erasing with p=0.5; square size in [6,10]
    erase_mask = torch.rand(N, device=device) < 0.5
    idx = torch.nonzero(erase_mask, as_tuple=False).squeeze(1)
    if idx.numel() != 0:
        # iterate only over images to erase (small loop)
        for i in idx.tolist():
            s = int(torch.randint(6, 11, (1,), device=device).item())  # size 6..10
            if s >= H or s >= W:
                # fallback to full image if weird, though H==W==28 so this won't trigger
                out[i] = 0.0
                continue
            top = int(torch.randint(0, H - s + 1, (1,), device=device).item())
            left = int(torch.randint(0, W - s + 1, (1,), device=device).item())
            out[i, :, top:top + s, left:left + s] = 0.0

    return out
