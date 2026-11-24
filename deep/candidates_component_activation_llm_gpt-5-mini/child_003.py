import torch
from torch import nn
import math

# LeakySmallMish: tempered Mish with a small linear negative tail
# act(x) = scale * ( x * tanh( softplus(beta * x) ) + leak * clamp(x, max=0) )
# - Keeps the smooth, non-monotonic behaviour of Mish for most inputs but adds a
#   small negative-side linear slope (leak) to improve gradient flow for negative
#   activations (similar in spirit to LeakyReLU) without making the function
#   piecewise-discontinuous.
# - Stateless, fully elementwise, uses torch.nn.functional.softplus and torch.tanh.

_SCALE = 0.96   # modest overall scale to keep magnitudes conditioned for this small net
_BETA = 0.85    # temper softplus input slightly (like SmallMish)
_LEAK = 0.03    # small negative-side linear tail (only affects x < 0)


def act(x: torch.Tensor) -> torch.Tensor:
    """Activation applied after every conv and hidden linear layer.

    Returns a tensor with the same shape as the input.

    Implementation: tempered Mish with a small negative linear tail
    (LeakySmallMish):
        out = scale * ( x * tanh( softplus(beta * x) ) + leak * clamp(x, max=0) )

    The extra clamp term is cheap, elementwise, and preserves shape and dtype.
    """
    sp = torch.nn.functional.softplus(_BETA * x)
    mish_core = x * torch.tanh(sp)
    negative_tail = torch.clamp(x, max=0.0) * _LEAK
    return _SCALE * (mish_core + negative_tail)
