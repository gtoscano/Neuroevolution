#!/usr/bin/env python3
"""Code-level neuroevolution: the genome is PyTorch source code, and an LLM is the mutation operator.

    # offline: the same loop, with a random (non-LLM) code mutation -- runs without an API key
    uv run --with torch --with torchvision python llm_evolve_cnn.py --offline --children 6

    # with an LLM (OpenAI, as in the agentic labs): reads OPENAI_API_KEY (and optional OPENAI_MODEL,
    # default gpt-5-mini) from session-5-6-neuroevolution/.env
    uv run --with torch --with torchvision --with openai --with pydantic python llm_evolve_cnn.py --children 6

Loop (an archive-based evolution, as in the Agentic-DGM lab):
  1. pick a parent from the archive (tournament on validation accuracy)
  2. ask the mutation operator for a child: a NEW build_model() source file plus a hypothesis
  3. static check of the source (allowed imports and names only)
  4. run it in a separate process with a time limit: shape check, parameter budget, proxy training
  5. add the scored child to the archive (failed children are kept too, with their error)
"""
import argparse
import ast
import json
import os
import random
import subprocess
import sys
import textwrap

from cnn_common import HERE
from evolve_cnn import MINI_ALEXNET, describe, mutate, repair

CAND = os.path.join(HERE, "candidates")
ENV_FILE = os.path.join(HERE, "..", "..", ".env")           # session-5-6-neuroevolution/.env


def load_env(path=ENV_FILE):
    """Read KEY=value lines from the local .env file. The file WINS over the shell environment,
    so the key kept next to the course material is the one used."""
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'").strip()   # quotes, then spaces
ALLOWED_IMPORTS = {"torch", "torch.nn", "math"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals",
                   "getattr", "setattr", "os", "sys", "subprocess", "socket", "shutil", "pathlib"}
TIMEOUT = 300                                                # seconds per candidate


# ---------------------------------------------------------------- the genome is code
def genome_to_code(g, title="mini-AlexNet"):
    """Render an AlexNet-style genome as a readable PyTorch file (used for the seed and the offline mutation)."""
    conv, ch, size = [], 1, 28
    for b in g["conv"]:
        conv.append(f"nn.Conv2d({ch}, {b['f']}, kernel_size={b['k']}, padding={b['k'] // 2}), nn.ReLU(),")
        if b["pool"]:
            conv.append("nn.MaxPool2d(2),")
            size //= 2
        ch = b["f"]
    fc, width = [], ch * size * size
    for h in g["fc"]:
        fc.append(f"nn.Dropout({g['dropout']}), nn.Linear({width}, {h}), nn.ReLU(),")
        width = h
    fc.append(f"nn.Linear({width}, 10),")
    body = "\n".join("            " + line for line in conv)
    head = "\n".join("            " + line for line in fc)
    return f'''"""{title}"""
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
{body}
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
{head}
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model():
    return Net()
'''


def static_check(src, required="build_model", allowed=ALLOWED_IMPORTS):
    """Reject code that imports or calls anything outside a small whitelist, before it ever runs."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if a.name not in allowed]
        elif isinstance(node, ast.ImportFrom):
            bad = [] if node.module in allowed else [node.module]
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            bad = [node.id]
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr != "__init__":
            bad = [node.attr]
        else:
            continue
        if bad:
            raise ValueError(f"not allowed: {bad}")
    if not any(isinstance(n, ast.FunctionDef) and n.name == required for n in tree.body):
        raise ValueError(f"the file must define {required}()")


def evaluate(src, name):
    """Static check, then a separate process without API keys and with a time limit."""
    try:
        static_check(src)
    except Exception as e:
        return {"ok": False, "error": f"static check: {e}"}
    path = os.path.join(CAND, name + ".py")
    with open(path, "w") as fh:
        fh.write(src)
    env = {k: v for k, v in os.environ.items() if "KEY" not in k.upper() and "TOKEN" not in k.upper()}
    try:
        out = subprocess.run([sys.executable, os.path.join(HERE, "run_candidate.py"), path], cwd=HERE, env=env,
                             capture_output=True, text=True, timeout=TIMEOUT)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {TIMEOUT} s"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}


# ---------------------------------------------------------------- the mutation operators
SYSTEM = textwrap.dedent("""\
    You are a neural-architecture designer taking part in an evolutionary search.
    You receive a PyTorch file defining build_model() and its score, and return a modified file.
    Rules: input is a batch of 1x28x28 gray images (FashionMNIST), output must be 10 logits;
    at most 1,000,000 parameters; only `import torch` and `from torch import nn`;
    every candidate is trained for only 2 epochs on 6,000 images, so it must learn fast on a CPU.
    Make ONE substantive, testable change (for example: a residual connection, batch normalization,
    global average pooling instead of large fully connected layers, a different kernel layout),
    and state a falsifiable hypothesis about its effect on validation accuracy or size.""")

PROMPT = textwrap.dedent("""\
    PARENT (validation accuracy {acc:.3f}, {params:,} parameters):
    ```python
    {code}
    ```
    ARCHIVE SO FAR (best first; failures show what to avoid):
    {archive}
    Return the complete new file and your hypothesis.""")


def llm_mutation(parent, archive, client, model):
    from pydantic import BaseModel, Field

    class Child(BaseModel):
        hypothesis: str = Field(description="falsifiable claim about the change")
        code: str = Field(description="the complete new Python file defining build_model()")

    summary = "\n".join(f"- {r['name']}: " + (f"val {r['val_acc']:.3f}, {r['params']:,} params, {r['hypothesis'][:80]}"
                                              if r["ok"] else f"FAILED ({r['error'][:80]})")
                        for r in sorted(archive, key=lambda r: -r.get("val_acc", -1))[:8])
    resp = client.responses.parse(model=model, instructions=SYSTEM, text_format=Child,
                                  input=PROMPT.format(acc=parent["val_acc"], params=parent["params"],
                                                      code=parent["code"], archive=summary))
    child = resp.output_parsed
    return child.code, child.hypothesis


def offline_mutation(parent, rng):
    """No LLM: mutate the parent's genome and re-render it as code. Same loop, dumber operator."""
    g, op = mutate(parent["genome"], rng)
    return genome_to_code(g, f"offline mutation: {op}"), f"random mutation '{op}'", g


# ---------------------------------------------------------------- the search loop
def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--children", type=int, default=6)
    ap.add_argument("--offline", action="store_true", help="random code mutation instead of an LLM")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    global CAND
    tag = "offline" if args.offline else f"llm_{args.model}"     # one folder per model, so runs never overwrite
    CAND = os.path.join(HERE, f"candidates_{tag}")              # keep the two modes apart
    os.makedirs(CAND, exist_ok=True)
    rng = random.Random(args.seed)
    client = None
    if not args.offline:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit(f"OPENAI_API_KEY not found in the environment or {ENV_FILE}; use --offline to run without an LLM")
        from openai import OpenAI
        client = OpenAI()

    seed_genome = repair(json.loads(json.dumps(MINI_ALEXNET)))
    code = genome_to_code(seed_genome)
    archive = [{"name": "seed", "parent": None, "code": code, "genome": seed_genome,
                "hypothesis": "AlexNet-style starting point", **evaluate(code, "seed")}]
    print(f"seed: val {archive[0].get('val_acc', 0):.3f}  params {archive[0].get('params', 0):,}")
    for i in range(1, args.children + 1):
        ok = [r for r in archive if r["ok"]]
        parent = max(rng.sample(ok, min(3, len(ok))), key=lambda r: r["val_acc"])
        genome = None
        if args.offline:
            code, hyp, genome = offline_mutation(parent, rng)
        else:
            try:
                code, hyp = llm_mutation(parent, archive, client, args.model)
            except Exception as e:                      # never print the error text: it can contain the key
                print(f"child {i}: LLM request failed ({type(e).__name__}"
                      f"{', cause ' + type(e.__cause__).__name__ if e.__cause__ else ''}); skipped")
                continue
        rec = {"name": f"child_{i:03d}", "parent": parent["name"], "code": code, "genome": genome,
               "hypothesis": hyp, **evaluate(code, f"child_{i:03d}")}
        archive.append(rec)
        status = f"val {rec['val_acc']:.3f}  params {rec['params']:,}" if rec["ok"] else f"FAILED: {rec['error']}"
        print(f"child {i} (from {parent['name']}): {status}\n    hypothesis: {hyp}")
    best = max((r for r in archive if r["ok"]), key=lambda r: r["val_acc"])
    print(f"\nbest: {best['name']}  val {best['val_acc']:.3f}  params {best['params']:,}  -> {os.path.basename(CAND)}/{best['name']}.py")
    with open(os.path.join(HERE, f"results_{tag}.json"), "w") as fh:
        json.dump(archive, fh, indent=1)


if __name__ == "__main__":
    main()
