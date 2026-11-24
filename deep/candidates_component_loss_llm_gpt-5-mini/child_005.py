import torch
from torch import nn
import math

# Self-distillation style label smoothing (variant of label smoothing / knowledge distillation).
# For each example we form a soft target that is a convex combination of the hard one-hot
# label and the model's own softened prediction (teacher) computed at a temperature T.
# The teacher probabilities are detached so gradients flow only through the usual model
# log-probabilities. This can be seen as adaptive smoothing: when the model is uncertain
# its softened teacher is diffuse and the target is closer to one-hot; when confident the
# teacher provides informative soft mass across classes.
#
# Rationale for this budget: with only 2 epochs and small data, using the model's own
# soft predictions as the smoothing prior gives more informative regularization than
# uniform smoothing while remaining cheap (one softmax and one log_softmax per batch).

def loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute self-distillation label-smoothed cross-entropy.

    logits: (N, C)
    y: (N,) integer class labels

    Returns a differentiable scalar loss (mean over the batch).
    """
    # mixing coefficient: fraction of target mass coming from the softened model predictions
    smoothing = 0.10  # like conventional label smoothing strength
    # temperature to soften the model's own predictions (teacher); >1 -> softer
    temperature = 2.0

    # log-probabilities for the student (used by loss) - stable single op
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)

    # build hard one-hot targets
    with torch.no_grad():
        C = logits.size(1)
        # one-hot target
        one_hot = torch.zeros_like(logits).scatter_(1, y.unsqueeze(1), 1.0)

        # teacher probabilities: softened model predictions; detach to avoid gradients through targets
        teacher_probs = torch.nn.functional.softmax(logits / float(temperature), dim=1)
        teacher_probs = teacher_probs.detach()

        # convex combination: (1 - smoothing) * one_hot + smoothing * teacher_probs
        # This preserves a valid probability distribution per-example.
        target = (1.0 - smoothing) * one_hot + smoothing * teacher_probs

    # cross-entropy against the mixed (soft) target
    loss_per_example = -(target * log_probs).sum(dim=1)
    return loss_per_example.mean()
