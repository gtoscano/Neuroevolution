import torch
from torch import nn
import math

# Learning rate schedule for this search budget.
# Single substantive change vs parent (which used a small continuous sinusoidal micro-oscillation):
# - Replace the continuous micro-oscillation with a single, modest Gaussian-shaped "re-warm" bump
#   centered near 80% of training (amplitude ~+20%, narrow width). The main warmup + cosine
#   decay structure is unchanged. This is a small variant of warm-restart/re-warm ideas.

def lr(step: int, total: int) -> float:
    """
    Return learning rate for optimizer step in [0, total-1].
    Must return a float between 0 and 0.05.
    """
    # Safety and bounds
    if total <= 0:
        return 1e-3

    # Hyperparameters (kept modest to be robust on Adam + small data)
    peak_lr = 3.5e-3    # modestly higher than textbook 1e-3
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

    # ONE substantive change vs parent:
    # Replace ongoing micro-oscillation with a single late-stage Gaussian "re-warm" bump
    # centered at frac_center of the schedule, amplitude ~+20%, and moderate width.
    # This is intended to give a short boost late in training to help escape shallow minima.
    frac_center = 0.80
    amp = 0.20  # 20% multiplicative increase at the center
    # Width: a few percent of total steps; ensure at least 1 step of width
    sigma_steps = max(1.0, total * 0.06)
    center_step = frac_center * float(max(0, total - 1))

    # Gaussian bump multiplier
    rel = (s - center_step) / float(sigma_steps)
    bump = 1.0 + amp * math.exp(-0.5 * (rel * rel))

    lr_val *= bump

    # Ensure bounds [0, 0.05]
    if lr_val < 0.0:
        lr_val = 0.0
    elif lr_val > 0.05:
        lr_val = 0.05

    return float(lr_val)
