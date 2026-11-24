#!/usr/bin/env python3
"""Component-level neuroevolution: the network is FIXED (mini-AlexNet); the genome is ONE function.

    python llm_evolve_component.py --target activation --offline        # hand-written family of known designs
    python llm_evolve_component.py --target loss                         # LLM mutation (OPENAI_API_KEY from ../../.env)
    python llm_evolve_component.py --target schedule --model gpt-6-luna
    python llm_evolve_component.py --target augment

Targets (see components.py): activation act(x), loss loss(logits, y), schedule lr(step, total), augment augment(x).
The loop, the safety checks and the 2-epoch proxy are the ones of llm_evolve_cnn.py. Two differences:
  * the code genome is one short function with a fixed interface, not a whole network;
  * every candidate is scored with 3 seeds (identical initial weights across candidates), and the fitness is the
    mean, so a gain must beat the seed-to-seed spread to count.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import textwrap

from cnn_common import HERE
from components import BATCH, EPOCHS, N_TRAIN, TARGETS, TOTAL_STEPS, offline_sample
from llm_evolve_cnn import ALLOWED_IMPORTS, ENV_FILE, load_env, static_check

TIMEOUT = 300                                                # seconds per candidate (3 trainings)

SYSTEM = textwrap.dedent(f"""\
    You are a deep-learning researcher taking part in an evolutionary search over ONE component of a training
    pipeline. The network is FIXED: a small AlexNet-style CNN (5 conv layers, 3 max-pools, two hidden linear
    layers of 128 with dropout 0.5, about 166k parameters). Training is FIXED except for your component:
    Adam, batch {BATCH}, {EPOCHS} epochs on {N_TRAIN:,} FashionMNIST images ({TOTAL_STEPS} steps), on a CPU.
    Fitness: mean validation accuracy over 3 training seeds. Differences below about 0.01 are noise.

    You write: {{role}}.
    Interface: `{{signature}}`. Return a complete Python file defining that function (and helpers if needed).
    Only `import torch`, `from torch import nn` and `import math` are allowed; `torch.nn.functional` may be used
    as an attribute. No file, network or system access. Keep it fast: it runs on every batch.

    The textbook default is the starting point. Known alternatives are allowed, but the purpose of this search
    is to find something that fits THIS budget, so new or unusual designs are welcome. Make ONE substantive,
    testable change to the parent, say whether it is a known method, a variant of one, or new,
    and state a falsifiable hypothesis about mean validation accuracy.""")

PROMPT = textwrap.dedent("""\
    PARENT (mean validation accuracy {acc:.3f}, per seed {accs}):
    ```python
    {code}
    ```
    ARCHIVE SO FAR (best first; failures show what to avoid):
    {archive}
    Return the complete new file, your hypothesis and the novelty label.""")


def evaluate(src, name, target, cand):
    """Static check, then a separate process without API keys and with a time limit."""
    try:
        static_check(src, required=TARGETS[target]["func"], allowed=ALLOWED_IMPORTS)
    except Exception as e:
        return {"ok": False, "error": f"static check: {e}"}
    path = os.path.join(cand, name + ".py")
    with open(path, "w") as fh:
        fh.write(src)
    env = {k: v for k, v in os.environ.items() if "KEY" not in k.upper() and "TOKEN" not in k.upper()}
    try:
        out = subprocess.run([sys.executable, os.path.join(HERE, "run_component.py"), target, path], cwd=HERE,
                             env=env, capture_output=True, text=True, timeout=TIMEOUT)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {TIMEOUT} s"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}


def llm_mutation(parent, archive, client, model, target):
    from typing import Literal
    from pydantic import BaseModel, Field

    class Child(BaseModel):
        hypothesis: str = Field(description="falsifiable claim about mean validation accuracy")
        novelty: Literal["known", "variant", "new"] = Field(description="known method, variant of one, or new")
        code: str = Field(description="the complete new Python file defining the function")

    summary = "\n".join(f"- {r['name']}: " + (f"val {r['val_acc']:.3f} ({r['novelty']}) {r['hypothesis'][:90]}"
                                              if r["ok"] else f"FAILED ({r['error'][:90]})")
                        for r in sorted(archive, key=lambda r: -r.get("val_acc", -1))[:8])
    t = TARGETS[target]
    resp = client.responses.parse(
        model=model, instructions=SYSTEM.format(role=t["role"], signature=t["signature"]), text_format=Child,
        input=PROMPT.format(acc=parent["val_acc"], accs=", ".join(f"{a:.3f}" for a in parent["val_accs"]),
                            code=parent["code"], archive=summary))
    c = resp.output_parsed
    return c.code, c.hypothesis, c.novelty


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(TARGETS), required=True)
    ap.add_argument("--children", type=int, default=6)
    ap.add_argument("--offline", action="store_true", help="sample a hand-written family of known designs, no LLM")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tag = f"{args.target}_" + ("offline" if args.offline else f"llm_{args.model}")
    cand = os.path.join(HERE, f"candidates_component_{tag}")
    os.makedirs(cand, exist_ok=True)
    rng = random.Random(args.seed)
    client = None
    if not args.offline:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit(f"OPENAI_API_KEY not found in the environment or {ENV_FILE}; use --offline")
        from openai import OpenAI
        client = OpenAI()

    code = TARGETS[args.target]["seed"]
    archive = [{"name": "seed", "parent": None, "code": code, "hypothesis": "textbook default", "novelty": "known",
                **evaluate(code, "seed", args.target, cand)}]
    if not archive[0]["ok"]:
        raise SystemExit(f"the seed failed: {archive[0]['error']}")
    print(f"[{args.target}] seed: val {archive[0]['val_acc']:.3f}  per seed {archive[0]['val_accs']}")
    for i in range(1, args.children + 1):
        ok = [r for r in archive if r["ok"]]
        parent = max(rng.sample(ok, min(3, len(ok))), key=lambda r: r["val_acc"])
        if args.offline:
            code, desc = offline_sample(args.target, rng)
            hyp, novelty = f"known design: {desc}", "known"
        else:
            try:
                code, hyp, novelty = llm_mutation(parent, archive, client, args.model, args.target)
            except Exception as e:                      # never print the error text: it can contain the key
                print(f"child {i}: LLM request failed ({type(e).__name__}"
                      f"{', cause ' + type(e.__cause__).__name__ if e.__cause__ else ''}); skipped")
                continue
        rec = {"name": f"child_{i:03d}", "parent": parent["name"], "code": code, "hypothesis": hyp,
               "novelty": novelty, **evaluate(code, f"child_{i:03d}", args.target, cand)}
        archive.append(rec)
        status = (f"val {rec['val_acc']:.3f}  per seed {', '.join(f'{a:.3f}' for a in rec['val_accs'])}"
                  if rec["ok"] else f"FAILED: {rec['error']}")
        print(f"child {i} (from {parent['name']}, {novelty}): {status}\n    {hyp}")
    best = max((r for r in archive if r["ok"]), key=lambda r: r["val_acc"])
    print(f"\nbest: {best['name']}  val {best['val_acc']:.3f}  (seed {archive[0]['val_acc']:.3f})"
          f"  -> {os.path.basename(cand)}/{best['name']}.py")
    with open(os.path.join(HERE, f"results_component_{tag}.json"), "w") as fh:
        json.dump(archive, fh, indent=1)


if __name__ == "__main__":
    main()
