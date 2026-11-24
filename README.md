# Running the code for the neuroevolution decks

Run every command from this folder (`src/`) unless a step says `cd deep`.

| step | deck | script | time |
|---|---|---|---|
| 1 | `ci-5-6-neuroevolution.pdf` | `neuroevolve_rbf.py` | under 1 s |
| 2–4 | `ci-5-6-neuroevolution-deep.pdf` | `deep/evolve_cnn.py`, `deep/llm_evolve_cnn.py` | minutes to 30 min |

## Requirements

- Python 3 with NumPy (step 1).
- [`uv`](https://docs.astral.sh/uv/) (steps 2–4; it fetches PyTorch for you, nothing to install by hand).
- An OpenAI API key (step 4c only).

## 1. RBF network designed by PSO (first deck)

```bash
python3 neuroevolve_rbf.py              # one run, shows the swarm's progress
python3 neuroevolve_rbf.py --seeds 10   # ten runs + summary table -> results.json
```

Data: `data/moons.csv` (included). Both runs overwrite `results.json`; run `--seeds 10` before drawing figures.

## 2. Set up PyTorch (deep deck; once per terminal)

```bash
cd deep
torchrun_() { uv run --no-project --with torch --with torchvision --with openai --with pydantic \
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "$@"; }
```

The first run downloads PyTorch (CPU) and FashionMNIST (about 30 MB, into `deep/data/`).

## 3. Evolve an AlexNet-style CNN (Part 1)

```bash
torchrun_ python evolve_cnn.py --cycles 4 --final-epochs 1   # quick demo, a few minutes
torchrun_ python evolve_cnn.py                               # full run, about 30 min -> results_cnn.json
python3 draw_deep_figures.py                                 # redraw the deck's figures -> ../../images/
```

Both runs overwrite `results_cnn.json`; draw figures only after the full run.

## 4. Architectures written by an LLM (Part 3)

a. Without an LLM (random code edits; no key needed):

```bash
torchrun_ python llm_evolve_cnn.py --offline --children 6
```

b. Create `session-5-6-neuroevolution/.env` (two folders up; it is git-ignored, never share it):

```
OPENAI_API_KEY=sk-...
```

c. With an LLM:

```bash
torchrun_ python llm_evolve_cnn.py --children 6                       # gpt-5-mini (default)
torchrun_ python llm_evolve_cnn.py --children 6 --model gpt-6-luna    # any other model
```

Each run writes `results_<offline|llm_MODEL>.json` and a `candidates_<...>/` folder with every generated network.

## Expected results (as on the slides)

| step | result |
|---|---|
| 1 | PSO: 2–4 hidden units, test accuracy 0.945 |
| 3 | evolved CNN test 0.917 vs mini-AlexNet 0.893 |
| 4 | best proxy: offline 0.821, gpt-5-mini 0.831, gpt-6-luna 0.819 |

Your numbers can differ by about 0.02 in steps 3 and 4: training starts from random weights.
