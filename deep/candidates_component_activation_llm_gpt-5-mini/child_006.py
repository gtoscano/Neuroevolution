import torch
from torch import nn
import math

# RMS-damped tempered Mish (variant)
# act(x) = SCALE * x * tanh( softplus(BETA * x) ) / sqrt(1 + GAMMA * x^2)
# - Keeps the smooth, non-monotonic shape of Mish while mildly damping large-magnitude
#   activations via a cheap per-element RMS-style denominator. This aims to reduce
#   activation explosion/instability on this small AlexNet and short CPU training budget
#   without adding state or expensive ops.
# - Stateless, elementwise, uses torch.nn.functional.softplus and torch.tanh; all ops are
#   vectorized and cheap.

_SCALE = 0.98
_BETA = 0.9
_GAMMA = 0.035


def act(x: torch.Tensor) -> torch.Tensor:
    """RMS-damped tempered Mish activation.

    Returns a tensor with the same shape as input `x`.

    Implementation: SCALE * x * tanh( softplus(BETA * x) ) / sqrt(1 + GAMMA * x^2)
    """
    # softplus(beta * x)
    sp = torch.nn.functional.softplus(_BETA * x)
    # classic Mish core
    mish_core = x * torch.tanh(sp)
    # RMS-style damping denominator (>= 1), keeps outputs bounded for large |x|
    denom = torch.sqrt(1.0 + _GAMMA * (x * x))
    return _SCALE * mish_core / denom
