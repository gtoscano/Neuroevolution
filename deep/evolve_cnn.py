#!/usr/bin/env python3
"""Neuroevolution of an AlexNet-style CNN on FashionMNIST with regularized ("aging") evolution.

    uv run --with torch --with torchvision python evolve_cnn.py              # about 30 minutes on a 14-core CPU
    uv run --with torch --with torchvision python evolve_cnn.py --cycles 4 --final-epochs 1   # a quick demo

The genome keeps AlexNet's overall shape -- a stack of convolution blocks (conv, ReLU, optional
max-pooling), then fully connected layers with dropout -- but evolves how many blocks and layers,
how many filters, which kernel sizes, where to pool, how wide, and how much dropout.
Weights are NOT evolved: every candidate is trained with Adam (session 5's gradient descent).
Search: regularized evolution (Real et al., 2019): keep a population of P networks; repeatedly
sample S of them, mutate the best, train the child, add it, and remove the OLDEST network.
"""
import argparse
import copy
import json
import os
import random

from torch import nn

from cnn_common import HERE, accuracy, load_data, n_params, proxy_fitness, train

FILTERS, KERNELS, WIDTHS, DROPOUTS = [8, 16, 32, 64], [3, 5], [64, 128, 256], [0.0, 0.25, 0.5]

# AlexNet (2012): 5 conv layers (96, 256, 384, 384, 256 filters; kernels 11, 5, 3, 3, 3; max-pool after 1, 2, 5),
# then 3 fully connected layers (4096, 4096, 1000) with dropout 0.5. Scaled down for 28x28 gray images:
MINI_ALEXNET = {"conv": [{"f": 32, "k": 5, "pool": True}, {"f": 64, "k": 3, "pool": True},
                         {"f": 64, "k": 3, "pool": False}, {"f": 64, "k": 3, "pool": False},
                         {"f": 32, "k": 3, "pool": True}],
                "fc": [128, 128], "dropout": 0.5}


def repair(g):
    """Keep every genome buildable: 1-5 conv blocks, at most 3 poolings (28 -> 14 -> 7 -> 3), 0-2 FC layers."""
    g["conv"] = g["conv"][:5] or [{"f": 16, "k": 3, "pool": True}]
    pools = 0
    for b in g["conv"]:
        if b["pool"]:
            pools += 1
            b["pool"] = pools <= 3
    g["fc"] = g["fc"][:2]
    return g


def build(g):
    """Genome -> PyTorch model. Padding k//2 keeps the size; only max-pooling halves it."""
    layers, ch, size = [], 1, 28
    for b in g["conv"]:
        layers += [nn.Conv2d(ch, b["f"], b["k"], padding=b["k"] // 2), nn.ReLU()]
        if b["pool"]:
            layers.append(nn.MaxPool2d(2))
            size //= 2
        ch = b["f"]
    layers.append(nn.Flatten())
    width = ch * size * size
    for h in g["fc"]:
        layers += [nn.Dropout(g["dropout"]), nn.Linear(width, h), nn.ReLU()]
        width = h
    layers.append(nn.Linear(width, 10))
    return nn.Sequential(*layers)


def describe(g):
    conv = " ".join(f"{b['f']}k{b['k']}{'P' if b['pool'] else ''}" for b in g["conv"])
    return f"conv[{conv}] fc{g['fc']} drop{g['dropout']}"


def mutate(g, rng):
    """One random structural or parameter change -- the evolutionary 'variation' step.
    Only operators that apply are drawn, and the child always differs from its parent."""
    while True:
        c = copy.deepcopy(g)
        ops = ["filters", "kernel", "pool", "dropout"]
        ops += ["add_conv"] if len(c["conv"]) < 5 else []
        ops += ["remove_conv"] if len(c["conv"]) > 1 else []
        ops += ["add_fc"] if len(c["fc"]) < 2 else []
        ops += ["remove_fc", "width"] if c["fc"] else []
        op = rng.choice(ops)
        i = rng.randrange(len(c["conv"]))
        if op == "add_conv":
            c["conv"].insert(i + 1, {"f": rng.choice(FILTERS), "k": rng.choice(KERNELS), "pool": False})
        elif op == "remove_conv":
            c["conv"].pop(i)
        elif op == "filters":
            c["conv"][i]["f"] = rng.choice(FILTERS)
        elif op == "kernel":
            c["conv"][i]["k"] = rng.choice(KERNELS)
        elif op == "pool":
            c["conv"][i]["pool"] = not c["conv"][i]["pool"]
        elif op == "add_fc":
            c["fc"].append(rng.choice(WIDTHS))
        elif op == "remove_fc":
            c["fc"].pop()
        elif op == "width":
            c["fc"][rng.randrange(len(c["fc"]))] = rng.choice(WIDTHS)
        else:
            c["dropout"] = rng.choice(DROPOUTS)
        c = repair(c)
        if describe(c) != describe(g):
            return c, op


def regularized_evolution(D, P=8, S=3, cycles=24, seed=0):
    rng = random.Random(seed)
    cache, history = {}, []

    def evaluate(g, origin):
        key = describe(g)
        if key not in cache:                                   # deduplicate: never train the same genome twice
            m = build(g)
            acc, sec = proxy_fitness(m, D, seed=seed)
            cache[key] = {"val_acc": acc, "params": n_params(m), "seconds": sec}
        rec = {"genome": g, "arch": key, "origin": origin, **cache[key]}
        history.append(rec)
        print(f"{len(history):3d}  val {rec['val_acc']:.3f}  params {rec['params']:>9,}  {origin:12s} {key}")
        return rec

    population = [evaluate(repair(copy.deepcopy(MINI_ALEXNET)), "mini-AlexNet")]
    while len(population) < P:                                 # start: mutants of the mini-AlexNet
        g, op = mutate(MINI_ALEXNET, rng)
        population.append(evaluate(g, f"init:{op}"))
    for _ in range(cycles):
        sample = rng.sample(population, S)                     # tournament
        parent = max(sample, key=lambda r: r["val_acc"])
        child, op = mutate(parent["genome"], rng)
        population.append(evaluate(child, op))
        population.pop(0)                                      # remove the OLDEST, not the worst
    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=24)
    ap.add_argument("--final-epochs", type=int, default=5)
    args = ap.parse_args()
    D = load_data()
    print("proxy: 2 epochs on 6,000 images, accuracy on 2,000 validation images\n")
    hist = regularized_evolution(D, cycles=args.cycles)
    best = max(hist, key=lambda r: r["val_acc"])
    print(f"\nbest found: {best['arch']}  (proxy val {best['val_acc']:.3f}, {best['params']:,} parameters)")

    print(f"\n== Final check: train on 58,000 images for {args.final_epochs} epochs, test on 10,000 unseen images ==")
    final = {}
    for name, g in [("mini-AlexNet", repair(copy.deepcopy(MINI_ALEXNET))), ("evolved", best["genome"])]:
        m = train(build(g), D["train_full"], epochs=args.final_epochs)
        final[name] = {"arch": describe(g), "params": n_params(build(g)), "test_acc": accuracy(m, D["test"])}
        print(f"{name:13s} test {final[name]['test_acc']:.4f}   params {final[name]['params']:>9,}   {describe(g)}")
    with open(os.path.join(HERE, "results_cnn.json"), "w") as fh:
        json.dump({"history": hist, "final": final}, fh)
