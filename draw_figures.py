#!/usr/bin/env python3
"""Figures for the neuroevolution deck, from results.json (run neuroevolve_rbf.py --seeds 10 first).

    python3 draw_figures.py      # writes SVG files into ../images/

Plain SVG, no plotting library.
"""
import json
import os

import numpy as np

import neuroevolve_rbf as N

OUT = os.path.join(N.HERE, "..", "images")
INK, MUTED, GRID = "#252530", "#6b6266", "#e4dbd2"
BURG, GOLD, BLUE, TEAL = "#8f173e", "#a8761c", "#2563b0", "#11806a"
CLS = [BLUE, GOLD]
FONT = "DejaVu Sans, Arial, sans-serif"
R = json.load(open(os.path.join(N.HERE, "results.json")))
SHOW_SEED = 4                                            # a typical run: 2 units, test 0.95


def T(x, y, s, size=16, col=INK, anchor="middle", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{col}" text-anchor="{anchor}" font-weight="{weight}" '
            f'stroke="#fffdf9" stroke-width="4" paint-order="stroke" stroke-linejoin="round">{s}</text>')


def save(name, w, h, body):
    with open(os.path.join(OUT, name + ".svg"), "w") as fh:
        fh.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
                 f'font-family="{FONT}"><rect width="{w}" height="{h}" fill="#fffdf9"/>{body}</svg>')
    print("wrote", name)


LIM = (-2.4, 2.4)


def plane(x0, y0, S, title, mu=None, sig=None, sets=("tr",), shade=True):
    """One square panel of the (standardized) input plane."""
    X = lambda v: x0 + (v - LIM[0]) / (LIM[1] - LIM[0]) * S
    Y = lambda v: y0 + S - (v - LIM[0]) / (LIM[1] - LIM[0]) * S
    cid = f"clip{int(x0)}"
    s = f'<defs><clipPath id="{cid}"><rect x="{x0}" y="{y0}" width="{S}" height="{S}"/></clipPath></defs><g clip-path="url(#{cid})">'
    if mu is not None and shade:
        W = N.fit_output(np.array(mu), np.array(sig))
        n = 70
        g = np.linspace(LIM[0], LIM[1], n + 1)
        c = (g[:-1] + g[1:]) / 2
        P = np.array([[a, b] for b in c[::-1] for a in c])
        o = N.hidden(P, np.array(mu), np.array(sig)) @ W
        k, conf = o.argmax(1), np.clip(np.abs(o[:, 1] - o[:, 0]), 0, 1)
        cw = S / n
        for i in range(n * n):
            r, q = divmod(i, n)
            s += (f'<rect x="{x0 + q * cw:.1f}" y="{y0 + r * cw:.1f}" width="{cw + 0.5:.1f}" height="{cw + 0.5:.1f}" '
                  f'fill="{CLS[k[i]]}" fill-opacity="{0.05 + 0.22 * conf[i]:.2f}"/>')
    s += f'<rect x="{x0}" y="{y0}" width="{S}" height="{S}" fill="none" stroke="{INK}" stroke-width="1.2"/>'
    ix = {"tr": N.TR, "va": N.VA, "te": N.TE}
    for name in sets:
        for i in ix[name]:
            px, py = X(N.X[i, 0]), Y(N.X[i, 1])
            if not (LIM[0] < N.X[i, 0] < LIM[1] and LIM[0] < N.X[i, 1] < LIM[1]):
                continue
            col = CLS[N.y[i]]
            if name == "tr":
                s += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.6" fill="{col}"/>'
            else:
                s += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.6" fill="#fffdf9" stroke="{col}" stroke-width="1.8"/>'
    if mu is not None:
        for (a, b), sg in zip(mu, sig):
            s += f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="{sg / (LIM[1] - LIM[0]) * S:.1f}" fill="none" stroke="{BURG}" stroke-width="2.2"/>'
            s += (f'<path d="M{X(a):.1f} {Y(b) - 9:.1f} L{X(a) + 9:.1f} {Y(b):.1f} L{X(a):.1f} {Y(b) + 9:.1f} L{X(a) - 9:.1f} {Y(b):.1f} Z" '
                  f'fill="{BURG}" stroke="#fffdf9" stroke-width="1.5"/>')
    s += "</g>"
    if mu is not None:
        for (a, b), sg in zip(mu, sig):
            if not (LIM[0] < a < LIM[1] and LIM[0] < b < LIM[1]):
                cx, cy = min(max(X(a), x0 + 14), x0 + S - 14), min(max(Y(b), y0 + 14), y0 + S - 14)
                s += f'<path d="M{cx:.1f} {cy - 9:.1f} L{cx + 9:.1f} {cy:.1f} L{cx:.1f} {cy + 9:.1f} L{cx - 9:.1f} {cy:.1f} Z" fill="#fffdf9" stroke="{BURG}" stroke-width="2"/>'
                s += T(cx + 14, cy + 5, f"center off the plot at ({a:.1f}, {b:.1f}), σ = {sg:.1f}", 13, BURG, "start")
    s += T(x0 + S / 2, y0 - 12, title, 17, INK, weight="bold")
    return s


def data_figure():
    S = 380
    s = plane(60, 50, S, "Two moons: 180 training points (filled)", sets=("tr",))
    s += plane(520, 50, S, "60 validation + 60 test points (hollow)", sets=("va", "te"))
    s += T(490, S + 92, "Two classes, two inputs, noisy: no straight line separates them.", 15, MUTED)
    save("moons-data", 980, S + 110, s)


def compare_figure():
    S = 380
    run = R["runs"][SHOW_SEED]
    mu8, sig8 = N.kmeans_rbf(8, 0)
    va8, te8 = N.evaluate(mu8, sig8, N.VA)[1], N.evaluate(mu8, sig8, N.TE)[1]
    s = plane(60, 50, S, f"k-means + global width: 8 units", mu8.tolist(), sig8.tolist(), sets=("tr",))
    s += T(60 + S / 2, S + 82, f"validation {va8:.3f} · test {te8:.3f}", 15, INK)
    s += plane(520, 50, S, f"evolved by PSO: {run['units']} units", run["mu"], run["sigma"], sets=("tr",))
    s += T(520 + S / 2, S + 82, f"validation {run['val_acc']:.3f} · test {run['test_acc']:.3f}", 15, INK)
    s += T(490, S + 110, "◆ center μ · circle radius σ · shading = predicted class", 14, MUTED)
    save("evolved-vs-kmeans", 980, S + 125, s)


def convergence_figure():
    W, H = 900, 400
    x0, y0, w, h = 90, 30, 760, 290
    hs = [r["history"] for r in R["runs"]]
    lo, hi = min(min(x) for x in hs), max(max(x) for x in hs)
    lo, hi = -0.25, -0.05
    X = lambda t: x0 + t / 100 * w
    Y = lambda f: y0 + h - (f - lo) / (hi - lo) * h
    s = ""
    for v in np.linspace(lo, hi, 5):
        s += f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x0 + w}" y2="{Y(v):.1f}" stroke="{GRID}"/>' + T(x0 - 8, Y(v) + 5, f"{v:.2f}", 13, MUTED, "end")
    for t in (0, 25, 50, 75, 100):
        s += T(X(t), y0 + h + 22, t, 13, MUTED)
    s += f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" stroke="{INK}"/><line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}" stroke="{INK}"/>'
    for i, hist in enumerate(hs):
        pts = " ".join(f"{X(t):.1f},{Y(max(lo, f)):.1f}" for t, f in enumerate(hist))
        col = BURG if i == SHOW_SEED else "#c9a9b3"
        s += f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{3 if i == SHOW_SEED else 1.6}"/>'
    s += T(x0 + w / 2, y0 + h + 48, "PSO iteration (20 particles each)", 15, INK)
    s += f'<text x="30" y="{y0 + h / 2}" font-size="15" fill="{INK}" text-anchor="middle" transform="rotate(-90 30 {y0 + h / 2})">best fitness = −(val. error + 0.01 · units)</text>'
    s += T(X(60), Y(-0.20), "10 independent runs; the highlighted run is the 2-unit network shown earlier", 14, MUTED)
    save("pso-convergence", W, H, s)


def tradeoff_figure():
    W, H = 900, 400
    x0, y0, w, h = 90, 30, 740, 290
    arch = [a for r in R["runs"] for a in r["archive"]]
    by_units = {}
    for u, err, mse in arch:
        by_units.setdefault(u, []).append(err)
    X = lambda u: x0 + (u - 0.5) / 8 * w
    Y = lambda e: y0 + h - e / 0.45 * h
    s = ""
    for v in (0, 0.1, 0.2, 0.3, 0.4):
        s += f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x0 + w}" y2="{Y(v):.1f}" stroke="{GRID}"/>' + T(x0 - 8, Y(v) + 5, f"{v:.1f}", 13, MUTED, "end")
    s += f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" stroke="{INK}"/><line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}" stroke="{INK}"/>'
    rng = np.random.default_rng(0)
    for u in sorted(by_units):
        s += T(X(u), y0 + h + 22, u, 13, MUTED)
        errs = np.array(by_units[u])
        pick = rng.choice(len(errs), min(len(errs), 400), replace=False)
        for e in errs[pick]:
            if e <= 0.45:
                s += f'<circle cx="{X(u) + rng.uniform(-14, 14):.1f}" cy="{Y(e):.1f}" r="2" fill="{MUTED}" fill-opacity="0.25"/>'
    best = [(u, min(by_units[u])) for u in sorted(by_units)]
    front, cur = [], 1
    for u, e in best:
        if e < cur:
            front.append((u, e)); cur = e
    s += f'<polyline points="{" ".join(f"{X(u):.1f},{Y(e):.1f}" for u, e in front)}" fill="none" stroke="{BURG}" stroke-width="3"/>'
    for u, e in front:
        s += f'<circle cx="{X(u):.1f}" cy="{Y(e):.1f}" r="7" fill="{BURG}" stroke="#fffdf9" stroke-width="2"/>'
        s += T(X(u) + 10, Y(e) - 10, f"{e:.3f}", 13, BURG, "start", "bold")
    s += T(x0 + w / 2, y0 + h + 48, "number of hidden units (structure)", 15, INK)
    s += f'<text x="30" y="{y0 + h / 2}" font-size="15" fill="{INK}" text-anchor="middle" transform="rotate(-90 30 {y0 + h / 2})">validation error</text>'
    s += T(X(5.6), Y(0.42), "gray: every network the swarms evaluated", 14, MUTED)
    s += T(X(5.6), Y(0.39), "burgundy: best error for each size (the trade-off front)", 14, BURG)
    save("size-vs-error", W, H, s)
    return front


def genome_figure():
    W, H = 980, 250
    s = T(W / 2, 28, "One particle = 8 slots × 4 genes = 32 numbers", 18, INK, weight="bold")
    ex = [(0.9, -1.0, 0.6, 0.7), (0.2, 0.4, 1.1, -0.5), (0.7, 1.1, -0.6, 0.4), (0.1, -0.2, 0.0, 0.1),
          (0.3, 0.9, 0.3, -1.2), (0.6, 0.1, 1.4, 0.2), (0.05, -1.5, -1.3, 0.8), (0.4, 1.6, 0.9, -0.3)]
    bw = 112
    for j, genes in enumerate(ex):
        x = 30 + j * (bw + 4)
        on = genes[0] > 0.5
        s += f'<rect x="{x}" y="50" width="{bw}" height="110" rx="6" fill="{"#f7e8ee" if on else "#f1ece6"}" stroke="{BURG if on else "#c9bdb0"}" stroke-width="{2 if on else 1}"/>'
        s += T(x + bw / 2, 72, f"unit {j + 1}", 14, BURG if on else MUTED, weight="bold")
        for k, (lab, val) in enumerate(zip(("on", "μ₁", "μ₂", "ln σ"), genes)):
            s += T(x + 10, 96 + k * 18, lab, 13, MUTED, "start") + T(x + bw - 10, 96 + k * 18, f"{val:+.2f}", 13, INK, "end")
        s += T(x + bw / 2, 182, "ON" if on else "off", 15, BURG if on else MUTED, weight="bold")
    s += T(W / 2, 222, "Units whose first gene is above 0.5 exist (here 3 of 8): the swarm searches structure and parameters together.", 15, MUTED)
    save("genome", W, H, s)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data_figure()
    genome_figure()
    compare_figure()
    convergence_figure()
    print("trade-off front:", tradeoff_figure())
