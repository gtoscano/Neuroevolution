import torch
from torch import nn
import math

# Learning rate schedule for this search budget.
# One substantive change vs parent (which used linear warmup + a single cosine decay
# with a tiny sinusoidal micro-oscillation):
# - After the warmup, use cosine annealing with a small number of warm restarts
#   (SGDR-style). Each restart raises the LR back toward the peak for a short
#   period, then decays again; this periodically injects higher LR to help
#   escape shallow/early minima in the tiny-budget regime.
# This is a variant of a known method (cosine annealing with restarts).


def lr(step: int, total: int) -> float:
    """
    Return learning rate for optimizer step in [0, total-1].
    Must return a float between 0 and 0.05.
    """
    # Safety and bounds
    if total <= 0:
        return 1e-3

    # Hyperparameters (kept modest to be robust on Adam + small data)
    peak_lr = 3.5e-3    # same modest peak as parent
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
        # Cosine annealing with warm restarts across the remaining steps
        remaining = total - warmup_steps
        if remaining <= 0:
            lr_val = peak_lr
        else:
            progress = s - warmup_steps  # 0 .. remaining-1

            # Number of restarts (choose a small integer to create several short cycles)
            cycles = 3
            # Don't create more cycles than remaining steps
            cycles = min(cycles, remaining)

            base_len = remaining // cycles
            extra = remaining % cycles

            # Find which cycle we're in and the local position inside it
            cum = 0
            cycle_index = 0
            cycle_len = 0
            cycle_start = 0
            for i in range(cycles):
                l = base_len + (1 if i < extra else 0)
                if progress < cum + l:
                    cycle_index = i
                    cycle_len = l
                    cycle_start = cum
                    break
                cum += l
            else:
                # Fallback to last cycle
                cycle_index = cycles - 1
                cycle_len = base_len + (1 if (cycles - 1) < extra else 0)
                cycle_start = remaining - cycle_len

            # local progress in this cycle [0,1]
            local_t = (progress - cycle_start) / float(max(1, cycle_len))
            # cosine anneal within the cycle from peak -> min
            cos_decay = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, local_t))))

            # Peak is reset at each restart (no extra decay across restarts here)
            peak_for_cycle = peak_lr
            lr_val = min_lr + (peak_for_cycle - min_lr) * cos_decay

    # Keep the parent's small sinusoidal micro-oscillation (~2%) to preserve a
    # tiny extra per-step jitter that can help brief escapes.
    osc = 1.0 + 0.02 * math.sin(2.0 * math.pi * (s / 10.0))
    lr_val *= osc

    # Ensure bounds [0, 0.05]
    if lr_val < 0.0:
        lr_val = 0.0
    elif lr_val > 0.05:
        lr_val = 0.05

    return float(lr_val)
