"""The four component targets for llm_evolve_component.py: the network stays fixed, one function evolves.

Each target is ONE function with a fixed interface, plugged into the same mini-AlexNet and the same
2-epoch proxy as llm_evolve_cnn.py:

    activation   act(x) -> tensor of the same shape        replaces every ReLU in the network
    loss         loss(logits, y) -> scalar tensor          replaces cross-entropy during training
    schedule     lr(step, total) -> float                  Adam's learning rate at each step
    augment      augment(x) -> tensor of the same shape    applied to each training batch (never to validation)

The baseline (seed) of every target is the textbook default: ReLU, cross-entropy, a constant 1e-3, no augmentation.
The offline operator samples from a small hand-written family per target (a structured genome of KNOWN options);
the LLM operator may write anything that fits the interface.
"""
import math
import random

import torch
from torch import nn

from evolve_cnn import MINI_ALEXNET

# ---------------------------------------------------------------- the fixed network
class Act(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


def fixed_net(act=None):
    """mini-AlexNet, exactly as rendered by llm_evolve_cnn.genome_to_code, with a pluggable activation."""
    A = (lambda: Act(act)) if act else nn.ReLU
    g, layers, ch, size = MINI_ALEXNET, [], 1, 28
    for b in g["conv"]:
        layers += [nn.Conv2d(ch, b["f"], kernel_size=b["k"], padding=b["k"] // 2), A()]
        if b["pool"]:
            layers.append(nn.MaxPool2d(2))
            size //= 2
        ch = b["f"]
    head, width = [nn.Flatten()], ch * size * size
    for h in g["fc"]:
        head += [nn.Dropout(g["dropout"]), nn.Linear(width, h), A()]
        width = h
    head.append(nn.Linear(width, 10))
    return nn.Sequential(*layers, *head)


# ---------------------------------------------------------------- the targets
BATCH, EPOCHS, N_TRAIN = 64, 2, 6000
TOTAL_STEPS = EPOCHS * math.ceil(N_TRAIN / BATCH)              # 188 optimizer steps in the proxy

TARGETS = {
    "activation": {
        "func": "act", "signature": "def act(x: torch.Tensor) -> torch.Tensor",
        "role": "the activation function used after every convolution and hidden linear layer (replaces ReLU); "
                "x has any shape and the output must have the same shape",
        "seed": "def act(x):\n    return torch.relu(x)\n"},
    "loss": {
        "func": "loss", "signature": "def loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor",
        "role": "the training loss (replaces cross-entropy); logits is (N, 10), y holds N integer class labels; "
                "return a differentiable scalar. Validation still measures plain accuracy",
        "seed": "def loss(logits, y):\n    return torch.nn.functional.cross_entropy(logits, y)\n"},
    "schedule": {
        "func": "lr", "signature": "def lr(step: int, total: int) -> float",
        "role": f"Adam's learning rate at optimizer step `step` (0 .. total-1, total = {TOTAL_STEPS}); "
                "return a float between 0 and 0.05",
        "seed": "def lr(step, total):\n    return 1e-3\n"},
    "augment": {
        "func": "augment", "signature": "def augment(x: torch.Tensor) -> torch.Tensor",
        "role": "data augmentation applied to every TRAINING batch x of shape (N, 1, 28, 28), normalized pixels "
                "(mean 0, std 1); return a tensor of the same shape. Use torch's random functions (they are seeded)",
        "seed": "def augment(x):\n    return x\n"},
}


def check(target, f):
    """Interface checks run before any training: wrong shapes, NaNs or out-of-range values fail fast."""
    if target == "activation":
        x = torch.randn(4, 32, 7, 7)
        out = f(x)
        if out.shape != x.shape or not torch.isfinite(out).all():
            raise ValueError("act(x) must return a finite tensor of the same shape")
    elif target == "loss":
        logits, y = torch.randn(8, 10, requires_grad=True), torch.randint(0, 10, (8,))
        out = f(logits, y)
        if not isinstance(out, torch.Tensor) or out.dim() != 0 or not torch.isfinite(out):
            raise ValueError("loss(logits, y) must return a finite scalar tensor")
        out.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise ValueError("loss must be differentiable with finite gradients")
    elif target == "schedule":
        vals = [float(f(s, TOTAL_STEPS)) for s in range(TOTAL_STEPS)]
        if not all(math.isfinite(v) and 0 <= v <= 0.05 for v in vals):
            raise ValueError("lr(step, total) must stay between 0 and 0.05")
    elif target == "augment":
        x = torch.randn(64, 1, 28, 28)
        out = f(x)
        if out.shape != x.shape or not torch.isfinite(out).all():
            raise ValueError("augment(x) must return a finite tensor of the same shape")


def hooks(target, f):
    """Turn the candidate function into (model, keyword arguments for cnn_common.train)."""
    if target == "activation":
        return fixed_net(act=f), {}
    if target == "loss":
        return fixed_net(), {"loss_fn": f}
    if target == "schedule":
        return fixed_net(), {"lr_fn": f}
    return fixed_net(), {"augment": f}


# ---------------------------------------------------------------- offline operator: a hand-written family
def offline_sample(target, rng: random.Random):
    """Random member of a small family of KNOWN designs, with random constants. Returns (code, description)."""
    u = rng.uniform
    if target == "activation":
        a, b, c = u(0.01, 0.3), u(0.5, 3.0), u(1.0, 6.0)
        forms = [(f"torch.where(x > 0, x, {a:.2f} * x)", f"leaky ReLU, slope {a:.2f}"),
                 (f"x * torch.sigmoid({b:.2f} * x)", f"swish, beta {b:.2f}"),
                 (f"torch.clamp(x, min=0.0, max={c:.1f})", f"clipped ReLU at {c:.1f}"),
                 (f"torch.nn.functional.softplus(x, beta={b:.1f})", f"softplus, beta {b:.1f}"),
                 ("torch.nn.functional.elu(x)", "ELU"),
                 ("torch.tanh(x)", "tanh")]
        body, desc = rng.choice(forms)
        return f"def act(x):\n    return {body}\n", desc
    if target == "loss":
        s, gm, w = u(0.02, 0.2), u(0.5, 3.0), u(0.02, 0.3)
        forms = [(f"    return torch.nn.functional.cross_entropy(logits, y, label_smoothing={s:.2f})\n",
                  f"label smoothing {s:.2f}"),
                 (f"    logp = torch.log_softmax(logits, 1).gather(1, y[:, None]).squeeze(1)\n"
                  f"    return (-(1 - logp.exp()) ** {gm:.2f} * logp).mean()\n", f"focal loss, gamma {gm:.2f}"),
                 (f"    p = torch.log_softmax(logits, 1)\n"
                  f"    entropy = -(p.exp() * p).sum(1).mean()\n"
                  f"    return torch.nn.functional.cross_entropy(logits, y) - {w:.2f} * entropy\n",
                  f"confidence penalty {w:.2f}")]
        body, desc = rng.choice(forms)
        return f"def loss(logits, y):\n{body}", desc
    if target == "schedule":
        peak, warm = 10 ** u(-3.5, -2.3), rng.randint(5, 60)
        forms = [(f"    return {peak:.2e}\n", f"constant {peak:.1e}"),
                 (f"    return {peak:.2e} * 0.5 * (1 + math.cos(math.pi * step / total))\n", f"cosine from {peak:.1e}"),
                 (f"    if step < {warm}:\n        return {peak:.2e} * (step + 1) / {warm}\n"
                  f"    return {peak:.2e} * (total - step) / (total - {warm})\n",
                  f"warm-up {warm} steps, linear decay, peak {peak:.1e}"),
                 (f"    return {peak:.2e} * (0.1 if step > 0.75 * total else 1.0)\n", f"step decay, {peak:.1e}")]
        body, desc = rng.choice(forms)
        return f"import math\n\n\ndef lr(step, total):\n{body}", desc
    p, s, k = u(0.05, 0.4), rng.randint(1, 3), rng.randint(4, 12)
    forms = [(f"    dx, dy = torch.randint(-{s}, {s + 1}, (2,)).tolist()\n"
              f"    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))\n", f"random shift up to {s} px"),
             ("    flip = torch.rand(x.shape[0], 1, 1, 1) < 0.5\n"
              "    return torch.where(flip, x.flip(3), x)\n", "horizontal flip, p 0.5"),
             (f"    return x + {p:.2f} * torch.randn_like(x)\n", f"Gaussian noise, sigma {p:.2f}"),
             (f"    i, j = torch.randint(0, {28 - k}, (2,)).tolist()\n"
              f"    x = x.clone()\n    x[:, :, i:i + {k}, j:j + {k}] = 0\n    return x\n", f"erase a {k}x{k} patch")]
    body, desc = rng.choice(forms)
    return f"def augment(x):\n{body}", desc
