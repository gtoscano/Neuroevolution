import torch
from torch import nn
import math

# SmallMish: a lightweight, tempered variant of Mish
# act(x) = scale * x * tanh( softplus(beta * x) )
# - Mish = x * tanh(softplus(x)) is a smooth, non-monotonic activation that often
#   improves optimization vs. ReLU/SiLU. Here we mildly temper the input to softplus
#   (beta < 1) and apply a modest scale to keep activation magnitudes well-conditioned
#   for this small network and its short, CPU-bound training budget.
# - Stateless, fully elementwise, uses torch.nn.functional.softplus and torch.tanh.

_SCALE = 0.95
_BETA = 0.8


def act(x: torch.Tensor) -> torch.Tensor:
    """Activation applied after every conv and hidden linear layer.

    Returns a tensor with the same shape as the input.

    Implementation: tempered+scaled Mish: scale * x * tanh( softplus(beta * x) ).
    """
    # Use the efficient softplus from torch.nn.functional and vectorized ops.
    return _SCALE * x * torch.tanh(torch.nn.functional.softplus(_BETA * x))
