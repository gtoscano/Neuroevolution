#!/usr/bin/env python3
"""Evaluate ONE candidate architecture file in its own process (called by llm_evolve_cnn.py).

    python run_candidate.py candidates/child_003.py

The file must define build_model() returning a torch.nn.Module that maps (N, 1, 28, 28) -> (N, 10).
Prints one JSON line: {"ok": true, "val_acc": ..., "params": ..., "seconds": ...} or {"ok": false, "error": ...}.
"""
import json
import sys

MAX_PARAMS = 1_000_000


def main(path):
    import torch
    from torch import nn
    from cnn_common import load_data, n_params, proxy_fitness
    src = open(path).read()
    namespace = {"torch": torch, "nn": nn}
    exec(compile(src, path, "exec"), namespace)              # source already passed the static check
    model = namespace["build_model"]()
    if not isinstance(model, nn.Module):
        raise TypeError("build_model() must return an nn.Module")
    out = model(torch.zeros(2, 1, 28, 28))
    if tuple(out.shape) != (2, 10):
        raise ValueError(f"output shape {tuple(out.shape)}, expected (2, 10)")
    p = n_params(model)
    if p > MAX_PARAMS:
        raise ValueError(f"{p:,} parameters, budget is {MAX_PARAMS:,}")
    acc, sec = proxy_fitness(model, load_data())
    return {"ok": True, "val_acc": acc, "params": p, "seconds": sec}


if __name__ == "__main__":
    try:
        print(json.dumps(main(sys.argv[1])))
    except Exception as e:                                    # report, never crash the search
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}))
