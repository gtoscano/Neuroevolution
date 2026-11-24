import torch
from torch import nn
import math

# Learning rate schedule variant: warmup -> hold (flat top) -> cosine decay
# One substantive change vs parent (which does warmup then immediate cosine decay):
# - After the linear warmup, hold the LR at the peak for a short plateau (~30% of total steps)
#   before applying cosine decay to the final floor LR. The small sinusoidal micro-oscillation
#   from the parent is retained (minor) so the only substantive change is the insertion of
#   the flat hold period. This is a variant of common warmup+cosine schedules (warmup+hold+decay).

def lr(step: int, total: int) -> float:
    """
    Return learning rate for optimizer step in [0, total-1].
    Must return a float between 0 and 0.05.
    """
    if total <= 0:
        return 1e-3

    # Base hyperparameters (conservative for Adam + small dataset)
    peak_lr = 3.5e-3    # modestly higher than the textbook default
    min_lr = 5e-6       # final floor for stability
    base_lr = 1e-6      # starting lr at step 0

    # Keep parent's warmup fraction (roughly 10% of training)
    warmup_steps = max(3, int(math.ceil(total * 0.10)))

    # NEW (substantive): hold the peak LR for a fraction of training after warmup
    # Use a flat-top for about 30% of the total steps (can be zero if tiny total)
    hold_steps = int(math.floor(total * 0.30))

    # Clamp step into [0, total-1]
    s = max(0, min(step, total - 1))

    if s < warmup_steps and warmup_steps > 0:
        # Linear warmup from base_lr to peak_lr
        t = s / float(warmup_steps)
        lr_val = base_lr + (peak_lr - base_lr) * t
    elif s < warmup_steps + hold_steps:
        # HOLD: keep LR at the peak for a short plateau
        lr_val = peak_lr
    else:
        # Cosine decay from peak_lr down to min_lr over remaining steps
        remaining = total - (warmup_steps + hold_steps)
        if remaining <= 0:
            # If no remaining steps, keep peak (edge case when total is small)
            lr_val = peak_lr
        else:
            t = (s - (warmup_steps + hold_steps)) / float(remaining)
            # cosine annealing (t in [0,1])
            cos_decay = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))
            lr_val = min_lr + (peak_lr - min_lr) * cos_decay

    # Keep the parent's small sinusoidal micro-oscillation (~2%) to help exploration
    osc = 1.0 + 0.02 * math.sin(2.0 * math.pi * (s / 10.0))
    lr_val *= osc

    # Ensure bounds [0, 0.05]
    if lr_val < 0.0:
        lr_val = 0.0
    elif lr_val > 0.05:
        lr_val = 0.05

    return float(lr_val)
