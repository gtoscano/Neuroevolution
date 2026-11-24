import torch
from torch import nn
import math

# Learning rate schedule for this search budget.
# One substantive change vs parent:
# - A single, short mid-training rewarm: a triangular bump that raises the LR
#   (up to 1.5x peak) for a narrow window centered around ~60% of training.
# This is a variant of warm-restart / rewarm techniques, applied as one short
# transient boost rather than multiple restarts.

def lr(step: int, total: int) -> float:
    """
    Return learning rate for optimizer step in [0, total-1].
    Must return a float between 0 and 0.05.

    Schedule summary:
    - Small linear warmup (≈8% of total) from base_lr to peak_lr
    - Cosine decay from peak_lr down to min_lr over the rest of training
    - Single triangular rewarm bump centered at ~60% of (post-warmup) progress
      that raises LR up to rewarm_peak_factor * peak_lr for a brief window.
    """
    # Safety and bounds
    if total <= 0:
        return 1e-3

    # Hyperparameters (conservative for Adam + small dataset)
    base_lr = 1e-6       # lr at step 0
    peak_lr = 4.0e-3     # modestly higher than parent
    min_lr = 5e-6        # final floor for stability

    # Warmup: ~8% of training, at least 3 steps
    warmup_steps = max(3, int(math.ceil(total * 0.08)))

    # Rewarm (single transient bump) settings:
    # center is offset from warmup by a fraction of the remaining steps
    rewarm_center_frac = 0.60   # center at 60% of the post-warmup span
    rewarm_width_frac = 0.12    # width ~12% of post-warmup span
    rewarm_peak_factor = 1.5    # bump raises LR up to 1.5 * peak_lr at center

    # Clamp step into [0, total-1]
    s = max(0, min(int(step), max(0, total - 1)))

    # Warmup phase
    if s < warmup_steps and warmup_steps > 0:
        t = s / float(warmup_steps)
        lr_val = base_lr + (peak_lr - base_lr) * t
    else:
        # Cosine decay from peak_lr down to min_lr over remaining steps
        remaining = total - warmup_steps
        if remaining <= 0:
            lr_val = peak_lr
        else:
            t = (s - warmup_steps) / float(max(1, remaining))
            # cosine annealing (t in [0,1])
            cos_decay = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))
            lr_val = min_lr + (peak_lr - min_lr) * cos_decay

        # Apply one short triangular rewarm bump
        # Compute center and width (in absolute steps)
        center = warmup_steps + int(round(remaining * rewarm_center_frac))
        width = max(3, int(round(remaining * rewarm_width_frac)))
        half = width / 2.0
        dist = abs(s - center)
        if dist <= half:
            # triangular shape in [0,1], 1 at center, 0 at edges
            tri = 1.0 - (dist / half) if half > 0 else 1.0
            bump_factor = 1.0 + (rewarm_peak_factor - 1.0) * tri
            lr_val = lr_val * bump_factor

    # Ensure bounds [0, 0.05]
    if lr_val < 0.0:
        lr_val = 0.0
    elif lr_val > 0.05:
        lr_val = 0.05

    return float(lr_val)
