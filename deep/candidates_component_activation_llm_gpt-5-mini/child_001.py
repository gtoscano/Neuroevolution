import torch
from torch import nn
import math

# A lightweight, slightly scaled Swish (SiLU) variant:
# act(x) = scale * x * sigmoid(beta * x)
# - Swish/SiLU is a smooth, non-monotonic activation that often trains faster than ReLU.
# - We use a mild scaling and beta<1 to make gradients well-conditioned for this small network
#   and the short (2-epoch) training budget.
# This function is stateless and fast (only a sigmoid and elementwise ops).

_SCALE = 1.15
_BETA = 0.9


def act(x: torch.Tensor) -> torch.Tensor:
    """Activation applied after every conv and hidden linear layer.

    Returns tensor of the same shape as input.

    Implementation: scaled Swish (x * sigmoid(beta * x)) to give a smooth, nonzero-centered
    activation with gentle gradient flow for negative inputs.
    """
    # Use torch.sigmoid which is implemented efficiently; keep operations vectorized.
    return _SCALE * x * torch.sigmoid(_BETA * x)
