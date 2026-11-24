import torch
from torch import nn
import math

# Top-k label smoothing.
# Variant: instead of distributing the smoothing mass uniformly across all non-target classes,
# distribute it uniformly among the model's top-k predicted non-target classes for each example.
# Rationale: the smoothed mass is concentrated on the most-plausible confusions, providing
# a more informative soft target while still regularizing overconfidence.


def loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute top-k label-smoothed cross-entropy.

    logits: (N, C)
    y: (N,) integer class labels

    Returns a differentiable scalar loss (mean over the batch).
    """
    # base smoothing strength
    smoothing = 0.10
    # how many non-target classes receive the smoothing mass per example
    top_k = 3

    # compute log-probabilities once for numerical stability
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    C = logits.size(1)
    N = logits.size(0)

    # prepare smoothed target distributions (constructed without gradient)
    with torch.no_grad():
        # initialize all zeros (we'll set mass only on true label and on selected non-targets)
        smoothed = torch.zeros_like(log_probs)

        # ensure k is not larger than available non-target classes
        k = min(top_k, max(1, C - 1))

        # find top-(k+1) indices by logit value so we can skip the true label if present
        # (k+1 to allow for the case where the true label is among top-k predictions)
        Kplus = min(C, k + 1)
        top_indices = torch.topk(logits, k=Kplus, dim=1).indices  # (N, Kplus)

        # For each example, choose up to k non-target top indices. If there are fewer than k
        # candidates in the top-Kplus (rare when C is small), fall back to distributing the
        # smoothing uniformly across all non-target classes.
        y_unsq = y.unsqueeze(1)

        for i in range(N):
            yi = int(y[i].item())
            # set mass on the true label
            smoothed[i, yi] = 1.0 - smoothing

            # collect up to k non-target indices from the top candidates
            chosen = []
            for idx in top_indices[i].tolist():
                if idx != yi:
                    chosen.append(idx)
                    if len(chosen) >= k:
                        break

            if len(chosen) > 0:
                per = smoothing / float(len(chosen))
                smoothed[i, chosen] = per
            else:
                # fallback: distribute smoothing uniformly to all non-target classes
                if C - 1 > 0:
                    per = smoothing / float(C - 1)
                    # create a mask of non-target indices
                    mask = torch.ones(C, device=logits.device, dtype=logits.dtype)
                    mask[yi] = 0.0
                    smoothed[i] = smoothed[i] + mask * per

    # negative log-likelihood of smoothed targets
    loss_per_example = -(smoothed * log_probs).sum(dim=1)
    return loss_per_example.mean()
