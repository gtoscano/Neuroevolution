"""Shared pieces for the deep neuroevolution examples: data, the AlexNet-style template, training, scoring.

Data: FashionMNIST (28x28 gray images, 10 classes), downloaded once by torchvision into ./data.
To keep every candidate cheap on a laptop CPU, the search uses a low-fidelity proxy:
6,000 training images, 2,000 validation images, two epochs of Adam (batch 64). The final comparison
retrains the winners for longer and reports accuracy on 10,000 test images that the search never saw.
"""
import os
import time

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))


def load_data(n_train=6000, n_val=2000, seed=0):
    from torchvision import datasets
    tr = datasets.FashionMNIST(DATA, train=True, download=True)
    te = datasets.FashionMNIST(DATA, train=False, download=True)
    X = tr.data.float().div(255).unsqueeze(1)
    X = (X - 0.286) / 0.353                                   # FashionMNIST mean / std
    Y = tr.targets
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    itr, iva, ifull = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
    Xt = (te.data.float().div(255).unsqueeze(1) - 0.286) / 0.353
    return {"train": (X[itr], Y[itr]), "val": (X[iva], Y[iva]),
            "train_full": (torch.cat([X[itr], X[ifull]]), torch.cat([Y[itr], Y[ifull]])),
            "test": (Xt, te.targets)}


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train(model, data, epochs=1, lr=1e-3, batch=128, seed=0, loss_fn=None, lr_fn=None, augment=None):
    """Adam training. Optional hooks (used by llm_evolve_component.py; the defaults give the plain loop):
    loss_fn(logits, y) -> scalar, lr_fn(step, total_steps) -> learning rate, augment(x) -> x (training batches only)."""
    torch.manual_seed(seed)
    X, Y = data
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    total, step = epochs * ((len(X) + batch - 1) // batch), 0
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            xb = augment(X[idx]) if augment else X[idx]
            if lr_fn:
                for group in opt.param_groups:
                    group["lr"] = float(lr_fn(step, total))
            opt.zero_grad()
            loss_fn(model(xb), Y[idx]).backward()
            opt.step()
            step += 1
    return model


@torch.no_grad()
def accuracy(model, data, batch=1000):
    model.eval()
    X, Y = data
    ok = sum((model(X[i:i + batch]).argmax(1) == Y[i:i + batch]).sum().item() for i in range(0, len(X), batch))
    return ok / len(X)


def proxy_fitness(model, D, seed=0):
    """Low-fidelity score used during the search: 2 epochs on 6,000 images, accuracy on 2,000 validation images."""
    t0 = time.time()
    train(model, D["train"], epochs=2, batch=64, seed=seed)
    return accuracy(model, D["val"]), time.time() - t0
