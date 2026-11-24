import torch
from torch import nn
import math

# Leaky-GELU (lightweight variant):
# act(x) = scale * (GELU(x) + leak * x * I[x < 0])
# - GELU (approximate tanh form) gives smooth, non-monotonic responses that often
#   help optimization vs. ReLU/SiLU in small networks.
# - We add a small linear negative tail (leak) to ensure steady gradient flow for
#   strongly negative pre-activations (cheap, elementwise) and apply a mild scale
#   to keep activation magnitudes well-conditioned for this small CPU-bound run.
# - Stateless, fully elementwise, uses only torch.ops and math, so it's fast per-batch.

_SCALE = 0.97
_LEAK = 0.06


def _gelu_approx(x: torch.Tensor) -> torch.Tensor:
    # Approximate GELU: 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def act(x: torch.Tensor) -> torch.Tensor:
    """Activation applied after every conv and hidden linear layer.

    Returns a tensor with the same shape as the input.

    Implementation: scaled GELU with a small linear negative tail added for x<0.
    """
    g = _gelu_approx(x)
    # Add a tiny linear leak only on negative inputs to keep gradients flowing.
    neg_tail = torch.where(x < 0, _LEAK * x, torch.zeros_like(x))
    return _SCALE * (g + neg_tail)
