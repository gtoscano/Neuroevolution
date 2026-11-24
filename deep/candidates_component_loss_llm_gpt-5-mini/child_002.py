import torch
from torch import nn
import math

# Label-smoothed cross-entropy loss.
# Known method: replace hard one-hot targets with a smoothed distribution and compute
# the cross-entropy against the model log-probabilities. Cheap (one log_softmax)
# and differentiable.

def loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute label-smoothed cross-entropy.

    logits: (N, C)
    y: (N,) integer class labels

    Returns a differentiable scalar loss (mean over the batch).
    """
    # smoothing strength: fraction of probability mass moved off the true label
    smoothing = 0.1

    # compute log-probabilities once for numerical stability
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)

    # number of classes
    C = logits.size(1)

    # create the smoothed target distribution in a numerically-safe way
    with torch.no_grad():
        # fill with smoothing mass distributed uniformly to non-target classes
        smoothed = torch.full_like(log_probs, smoothing / (C - 1))
        smoothed.scatter_(1, y.unsqueeze(1), 1.0 - smoothing)

    # negative log-likelihood of smoothed targets
    loss_per_example = -(smoothed * log_probs).sum(dim=1)
    return loss_per_example.mean()
