#!/usr/bin/env python3
"""Evaluate ONE candidate component file in its own process (called by llm_evolve_component.py).

    python run_component.py activation candidates_component_activation_offline/child_003.py

The file defines one function (act, loss, lr or augment, see components.py). It is plugged into the fixed
mini-AlexNet and scored with the 2-epoch proxy, once per seed; the network's initial weights and the data
order depend only on the seed, so every candidate is compared under identical conditions.
Prints one JSON line: {"ok": true, "val_acc": mean, "val_accs": [...], "seconds": ...} or {"ok": false, "error": ...}.
"""
import json
import math
import sys
import time

SEEDS = (0, 1, 2)


def main(target, path):
    import torch
    from torch import nn
    from cnn_common import accuracy, load_data, train
    from components import BATCH, EPOCHS, TARGETS, check, hooks
    src = open(path).read()
    namespace = {"torch": torch, "nn": nn, "math": math}
    exec(compile(src, path, "exec"), namespace)              # source already passed the static check
    f = namespace[TARGETS[target]["func"]]
    torch.manual_seed(0)
    check(target, f)
    D, accs, t0 = load_data(), [], time.time()
    for seed in SEEDS:
        torch.manual_seed(seed)                              # same initial weights for every candidate
        model, kw = hooks(target, f)
        if tuple(model(torch.zeros(2, 1, 28, 28)).shape) != (2, 10):
            raise ValueError("network output is not (2, 10)")
        train(model, D["train"], epochs=EPOCHS, batch=BATCH, seed=seed, **kw)
        accs.append(accuracy(model, D["val"]))
    return {"ok": True, "val_acc": sum(accs) / len(accs), "val_accs": accs, "seconds": time.time() - t0}


if __name__ == "__main__":
    try:
        print(json.dumps(main(sys.argv[1], sys.argv[2])))
    except Exception as e:                                    # report, never crash the search
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}))
