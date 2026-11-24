import torch
from torch import nn
import math

# Focal loss for multi-class classification (known method).
# Focal loss down-weights well-classified examples by (1 - p_t)^gamma, where p_t
# is the model probability for the true class. This encourages the optimizer to
# focus on harder examples and can improve generalization in limited-data regimes.
# Implemented in a numerically-stable way using log_softmax.


def loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute multi-class focal loss.

    logits: (N, C)
    y: (N,) integer class labels

    Returns a differentiable scalar (mean over the batch).
    """
    # focal loss hyperparameters
    gamma = 2.0   # focusing parameter; gamma=0 recovers standard CE
    alpha = 1.0   # optional scalar class weight (set to 1.0 means no class reweighting)

    # compute log-probabilities once for numerical stability
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)  # (N, C)
    probs = log_probs.exp()  # (N, C)

    # gather probabilities/log-probs of the true class
    idx = y.view(-1, 1)
    p_t = probs.gather(1, idx).squeeze(1).clamp(min=1e-12)        # (N,)
    log_p_t = log_probs.gather(1, idx).squeeze(1)                 # (N,)

    # focal loss per example: -(1 - p_t)^gamma * log(p_t)
    modulating = (1.0 - p_t) ** gamma
    loss_per_example = -alpha * modulating * log_p_t

    return loss_per_example.mean()
