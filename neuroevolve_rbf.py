#!/usr/bin/env python3
"""Neuroevolution of an RBF network with particle swarm optimization (PSO).

    python3 neuroevolve_rbf.py              # one run, seed 0, with progress
    python3 neuroevolve_rbf.py --seeds 10   # ten runs and a summary table

Only NumPy is needed.

The problem: classify the two-moons data set (2 inputs, 2 classes, 300 noisy points;
data/moons.csv, created by make_moons() below).
The network: an RBF network as in session 6, i.e. Gaussian units
    y_j = exp(-||x - mu_j||^2 / (2 sigma_j^2))
and a linear output layer whose weights are solved by least squares (no epochs).
What evolves: the hidden layer. Each of up to JMAX units has 4 genes:
    [on/off, mu_1, mu_2, log sigma]
so PSO searches the STRUCTURE (how many units are on) and the PARAMETERS (where the
centers are and how wide they are) at the same time.
Fitness (to maximize): -(validation error + LAMBDA * number of units).
The baseline: session 6's two-stage recipe, where k-means places the centers and the global rule sets
sigma = d_max / sqrt(J).
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JMAX = 8          # at most 8 hidden units
LAMBDA = 0.01     # fitness cost of one extra unit: prefers small networks
RIDGE = 1e-3      # tiny ridge term in the least-squares solve


# ---------------------------------------------------------------- data
def make_moons(n=150, noise=0.2, seed=42):
    """Two interleaving half circles, as in scikit-learn's make_moons."""
    r = np.random.default_rng(seed)
    t1, t2 = r.uniform(0, np.pi, n), r.uniform(0, np.pi, n)
    a = np.c_[np.cos(t1), np.sin(t1)]
    b = np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]
    X = np.vstack([a, b]) + r.normal(0, noise, (2 * n, 2))
    return X, np.r_[np.zeros(n, int), np.ones(n, int)]


def load_data():
    path = os.path.join(HERE, "data", "moons.csv")
    if not os.path.exists(path):
        X, y = make_moons()
        np.savetxt(path, np.c_[X, y], delimiter=",", header="x1,x2,class", comments="", fmt=["%.4f", "%.4f", "%d"])
    D = np.genfromtxt(path, delimiter=",", skip_header=1)
    X, y = D[:, :2], D[:, 2].astype(int)
    X = (X - X.mean(0)) / X.std(0)                       # standardize, as RBF distances require
    r = np.random.default_rng(0)                         # fixed split: 180 train / 60 validation / 60 test
    idx = [r.permutation(np.where(y == c)[0]) for c in (0, 1)]
    tr = np.concatenate([i[:90] for i in idx])
    va = np.concatenate([i[90:120] for i in idx])
    te = np.concatenate([i[120:] for i in idx])
    return X, y, tr, va, te


X, y, TR, VA, TE = load_data()
T = np.eye(2)[y]                                         # one-hot targets


# ---------------------------------------------------------------- the RBF network (session 6)
def hidden(Xs, mu, sig):
    """Responses of the Gaussian units, one column per unit, plus a bias column of 1s."""
    d2 = ((Xs[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
    return np.hstack([np.exp(-d2 / (2 * sig ** 2)), np.ones((len(Xs), 1))])


def fit_output(mu, sig):
    """Output weights by (ridge) least squares on the training set: one solve, no epochs."""
    P = hidden(X[TR], mu, sig)
    return np.linalg.solve(P.T @ P + RIDGE * np.eye(P.shape[1]), P.T @ T[TR])


def evaluate(mu, sig, ix):
    W = fit_output(mu, sig)
    o = hidden(X[ix], mu, sig) @ W
    return ((o - T[ix]) ** 2).mean(), (o.argmax(1) == y[ix]).mean()      # (mse, accuracy)


# ---------------------------------------------------------------- the encoding
GENES = 4                                                # on/off, mu_1, mu_2, log sigma
LB = np.tile([0.0, -2.5, -2.5, np.log(0.1)], JMAX)
UB = np.tile([1.0, 2.5, 2.5, np.log(3.0)], JMAX)


def decode(v):
    """Particle position -> (centers, widths). A unit is on when its first gene is > 0.5."""
    g = v.reshape(JMAX, GENES)
    on = g[:, 0] > 0.5
    if not on.any():
        on[np.argmax(g[:, 0])] = True                    # keep at least one unit
    return g[on, 1:3], np.exp(g[on, 3])


ARCHIVE = []                                             # every evaluation: (units, validation error)


def fitness(v):
    mu, sig = decode(v)
    mse, acc = evaluate(mu, sig, VA)
    ARCHIVE.append((len(mu), 1 - acc, mse))
    return -(mse + LAMBDA * len(mu))


def swarm_fitness(x):
    return np.array([fitness(row) for row in x])


# ---------------------------------------------------------------- PSO (as in labs/pybullet/arm/pso.py)
def pso(N, n, lb, ub, T, W, C1, C2, func, rng, verbose=False):
    x = lb + rng.random((N, n)) * (ub - lb)
    v = np.zeros((N, n))
    fx = func(x)
    p, fx_p = x.copy(), fx.copy()                        # personal bests
    g = p[np.argmax(fx_p)].copy()                        # global best
    history = [fx_p.max()]
    for t in range(T):
        v = W * v + C1 * rng.random((N, n)) * (p - x) + C2 * rng.random((N, n)) * (g - x)
        x = np.clip(x + v, lb, ub)
        fx = func(x)
        better = fx > fx_p
        p[better], fx_p[better] = x[better], fx[better]
        g = p[np.argmax(fx_p)].copy()
        history.append(fx_p.max())
        if verbose and (t % 10 == 9 or t == 0):
            mu, _ = decode(g)
            print(f"iteration {t + 1:3d}   best fitness {fx_p.max():+.4f}   units {len(mu)}")
    return g, fx_p.max(), history


# ---------------------------------------------------------------- the baseline (session 6, two-stage)
def kmeans_rbf(J, seed):
    r = np.random.default_rng(seed)
    pts = X[TR]
    mu = pts[r.choice(len(pts), J, replace=False)]
    for _ in range(100):
        a = ((pts[:, None] - mu[None]) ** 2).sum(-1).argmin(1)
        new = np.array([pts[a == j].mean(0) if (a == j).any() else mu[j] for j in range(J)])
        if np.allclose(new, mu):
            break
        mu = new
    dmax = max(np.linalg.norm(a - b) for a in mu for b in mu)
    return mu, np.full(J, dmax / np.sqrt(J))             # global width rule


def run(seed, verbose=False):
    ARCHIVE.clear()
    rng = np.random.default_rng(seed)
    g, best, history = pso(N=20, n=JMAX * GENES, lb=LB, ub=UB, T=100, W=0.7, C1=1.4, C2=1.4,
                           func=swarm_fitness, rng=rng, verbose=verbose)
    mu, sig = decode(g)
    return {"seed": seed, "units": len(mu), "mu": mu.tolist(), "sigma": sig.tolist(), "fitness": best,
            "val_acc": evaluate(mu, sig, VA)[1], "test_acc": evaluate(mu, sig, TE)[1],
            "history": history, "archive": list(ARCHIVE)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    print(f"two moons: {len(TR)} training, {len(VA)} validation, {len(TE)} test points\n")

    print("== Baseline: k-means centers + global width, output weights by least squares ==")
    base = {}
    for J in (2, 4, 8):
        accs = np.array([[evaluate(*kmeans_rbf(J, s), ix)[1] for ix in (VA, TE)] for s in range(10)])
        base[J] = accs.mean(0).tolist()
        print(f"J = {J}:  validation {accs[:, 0].mean():.3f}   test {accs[:, 1].mean():.3f}   (mean of 10 k-means starts)")

    print("\n== Neuroevolution: PSO evolves which units exist, their centers and their widths ==")
    runs = [run(s, verbose=(args.seeds == 1)) for s in range(args.seeds)]
    for r in runs:
        print(f"seed {r['seed']}: {r['units']} units   validation {r['val_acc']:.3f}   test {r['test_acc']:.3f}")
    if args.seeds > 1:
        u = np.array([r["units"] for r in runs]); te = np.array([r["test_acc"] for r in runs])
        va = np.array([r["val_acc"] for r in runs])
        print(f"\nover {len(runs)} runs: units {u.min()}-{u.max()} (median {np.median(u):g}),"
              f" validation {va.mean():.3f}, test {te.mean():.3f} (min {te.min():.3f}, max {te.max():.3f})")
    out = os.path.join(HERE, "results.json")
    with open(out, "w") as fh:
        json.dump({"baseline": base, "runs": runs}, fh)
    print(f"\nsaved {out}  (draw_figures.py turns it into pictures)")
