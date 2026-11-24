import torch
from torch import nn
import math

# Adaptive label smoothing loss.
# Variant of known method: label smoothing whose strength is adjusted per-example
# according to the model's current confidence (max predicted probability).
# More confident predictions receive stronger smoothing to discourage overconfidence.
# Cheap: one log_softmax per batch and a small amount of elementwise tensor work.


def loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Adaptive label-smoothed cross-entropy.

    logits: (N, C)
    y: (N,) integer class labels

    Returns a differentiable scalar loss (mean over the batch).
    """
    # compute log-probabilities once for numerical stability
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)

    N, C = logits.shape

    # Compute per-example confidence (max predicted probability) without creating
    # a gradient path through the target distribution.
    with torch.no_grad():
        probs = torch.exp(log_probs)  # numerically stable softmax
        p_max = probs.max(dim=1).values  # shape (N,)

        # Adaptive smoothing parameters (tunable). We map p_max in [1/C, 1]
        # to smoothing in [min_s, max_s]. When the model is near-uniform (p_max~=1/C)
        # use minimal smoothing; when it's confident (p_max->1) use maximal smoothing.
        min_s = 0.05
        max_s = 0.20

        # avoid division by zero for degenerate C
        if C <= 1:
            smoothing = torch.full((N,), min_s, device=logits.device, dtype=logits.dtype)
        else:
            base = 1.0 / float(C)
            denom = 1.0 - base
            # scale factor in [0,1]
            factor = (p_max - base) / denom
            factor = factor.clamp(0.0, 1.0)
            smoothing = (min_s + (max_s - min_s) * factor).clamp(min_s, max_s)

        # build smoothed target distributions per example
        # each non-target entry gets smoothing/(C-1), target gets 1 - smoothing
        off_val = (smoothing / float(C - 1)).unsqueeze(1)  # (N,1)
        on_val = (1.0 - smoothing).unsqueeze(1)  # (N,1)

        # create smoothed target matrix
        smoothed = off_val.expand(-1, C).clone()
        smoothed.scatter_(1, y.unsqueeze(1), on_val)

    # negative log-likelihood of smoothed targets
    loss_per_example = -(smoothed * log_probs).sum(dim=1)
    return loss_per_example.mean()
