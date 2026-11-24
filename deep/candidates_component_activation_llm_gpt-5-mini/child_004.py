import torch
from torch import nn
import math

# SmoothCompressedHardSwish: a variant of Hard-Swish with mild positive compression
# and a small negative linear tail.
# - Core: hardswish(x) = x * clamp((x + 3) / 6, 0, 1) (cheap, piecewise linear)
# - Mild positive compression: multiply core by (1 - c * sigmoid(k*x)) to slightly
#   reduce very large positive activations (stabilizes magnitudes in small nets).
# - Small negative linear tail (leak) for x < 0 to improve gradient flow on negative side.
# - A modest overall scale keeps magnitudes conditioned for this model size.

_SCALE = 0.98   # slight overall scaling
_LEAK = 0.05    # small negative slope for x < 0
_COMP_C = 0.09  # compression strength for large positives
_COMP_K = 0.6   # compression sharpness


def act(x: torch.Tensor) -> torch.Tensor:
    """Activation applied after every conv and hidden linear layer.

    Returns a tensor with the same shape and dtype as the input.

    Implementation details (SmoothCompressedHardSwish):
      t = clamp((x + 3) / 6, 0, 1)
      hswish = x * t
      comp = 1 - COMP_C * sigmoid(COMP_K * x)
      out_core = hswish * comp
      negative_tail = LEAK * clamp(x, max=0)
      out = SCALE * (out_core + negative_tail)

    This is a stateless, elementwise function that is cheap to compute on CPU.
    """
    # Hard-swish core (cheap piecewise linear gating)
    t = torch.clamp((x + 3.0) / 6.0, min=0.0, max=1.0)
    hswish = x * t

    # Mild smooth compression on the positive side to limit extreme positive magnitudes
    comp = 1.0 - _COMP_C * torch.sigmoid(_COMP_K * x)
    out_core = hswish * comp

    # Small negative linear tail for x < 0
    negative_tail = torch.clamp(x, max=0.0) * _LEAK

    return _SCALE * (out_core + negative_tail)