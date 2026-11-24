import torch
from torch import nn
import math

# Learning rate schedule for this search budget.
# One substantive change vs parent (constant 1e-3):
# - Linear warmup for the first ~10% of steps to a modestly higher peak LR,
#   then cosine decay to a small floor LR for stability. A tiny sinusoidal
#   micro-oscillation (~2%) is layered on top to help exploration.
# This is a variant of common warmup + cosine decay schedules.

def lr(step: int, total: int) -> float:
    """
    Return learning rate for optimizer step in [0, total-1].
    Must return a float between 0 and 0.05.
    """
    # Safety and bounds
    if total <= 0:
        return 1e-3

    # Hyperparameters (kept modest to be robust on Adam + small data)
    peak_lr = 3.5e-3    # modestly higher than parent 1e-3
    min_lr = 5e-6       # small floor for final stability
    base_lr = 1e-6      # starting lr at step 0

    # Warmup for roughly 10% of training, at least 3 steps
    warmup_steps = max(3, int(math.ceil(total * 0.10)))

    # Clamp step into [0, total-1]
    s = max(0, min(step, total - 1))

    if s < warmup_steps and warmup_steps > 0:
        # Linear warmup from base_lr to peak_lr
        t = s / float(warmup_steps)
        lr_val = base_lr + (peak_lr - base_lr) * t
    else:
        # Cosine decay from peak_lr down to min_lr over remaining steps
        remaining = total - warmup_steps
        if remaining <= 0:
            lr_val = peak_lr
        else:
            t = (s - warmup_steps) / float(remaining)
            # cosine annealing (t in [0,1])
            cos_decay = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))
            lr_val = min_lr + (peak_lr - min_lr) * cos_decay

    # Small sinusoidal micro-oscillation to help briefly jump out of sharp minima
    # Frequency chosen so there are several oscillations across training; amplitude small (~2%)
    osc = 1.0 + 0.02 * math.sin(2.0 * math.pi * (s / 10.0))
    lr_val *= osc

    # Ensure bounds [0, 0.05]
    if lr_val < 0.0:
        lr_val = 0.0
    elif lr_val > 0.05:
        lr_val = 0.05

    return float(lr_val)
