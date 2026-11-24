import torch
from torch import nn
import math

# Cross-entropy with a small confidence penalty (aka entropy regularization).
# Known method: add lambda * sum p * log p to penalize overconfident predictions.
# This implementation is cheap (one softmax/log-softmax) and returns a scalar.

def loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy plus a confidence penalty (negative entropy term).

    logits: (N, C)
    y: (N,) integer class labels

    Returns a differentiable scalar loss.
    """
    # strength of the confidence penalty; small value to avoid overpowering CE
    beta = 0.1

    # compute log-probabilities once for numerical stability
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)

    # standard cross-entropy (mean over batch)
    ce = torch.nn.functional.nll_loss(log_probs, y, reduction='mean')

    # predictive distribution
    probs = torch.exp(log_probs)

    # sum p * log p per example (this is the negative entropy); mean over batch
    neg_entropy = (probs * log_probs).sum(dim=1).mean()

    # final loss: CE + beta * (sum p log p)
    return ce + beta * neg_entropy
