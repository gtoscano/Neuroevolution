import math

# One substantive change vs parent: a OneCycle-style schedule with linear warmup
# to a higher peak lr, then cosine annealing down to a small final lr.
# This is a variant of the well-known OneCycle / warmup+cosine schedule.

def lr(step: int, total: int) -> float:
    """Return learning rate for optimizer step in [0..total-1].

    Schedule parameters chosen for a very short training run (188 steps):
    - small linear warmup from base_lr to max_lr over pct_start of total steps
    - cosine anneal from max_lr down to final_lr for the remaining steps

    All returned values are guaranteed in [0.0, 0.05].
    """
    # Safety for degenerate totals
    if total <= 0:
        return 1e-3

    # Hyperparameters (single substantive change from constant lr):
    base_lr = 1e-4      # start of warmup
    max_lr = 2.0e-2     # peak lr reached at end of warmup
    final_lr = 1e-5     # final lr at end of schedule
    pct_start = 0.30    # fraction of total steps used for linear warmup

    # clamp step
    s = max(0, min(step, total - 1))

    warmup_steps = max(1, int(total * pct_start))

    if s < warmup_steps:
        # linear warmup from base_lr -> max_lr (inclusive at end of warmup)
        t = s / float(warmup_steps)
        lr_val = base_lr + t * (max_lr - base_lr)
    else:
        # cosine anneal from max_lr -> final_lr over remaining steps
        decay_steps = max(1, total - warmup_steps)
        t = (s - warmup_steps) / float(decay_steps)
        # cosine interpolation (t=0 -> cos=1 -> lr=max_lr; t=1 -> cos=0 -> lr=final_lr)
        cos_term = 0.5 * (1.0 + math.cos(math.pi * t))
        lr_val = final_lr + (max_lr - final_lr) * cos_term

    # enforce allowed bounds
    if lr_val < 0.0:
        lr_val = 0.0
    if lr_val > 0.05:
        lr_val = 0.05

    return float(lr_val)
