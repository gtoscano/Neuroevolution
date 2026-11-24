#!/usr/bin/env python3
"""Figures for the deep neuroevolution deck, from results_cnn.json (run evolve_cnn.py first).

    python3 draw_deep_figures.py      # writes SVG files into ../../images/   (plain Python, no torch needed)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "images")
INK, MUTED, GRID = "#252530", "#6b6266", "#e4dbd2"
BURG, GOLD, BLUE, TEAL = "#8f173e", "#a8761c", "#2563b0", "#11806a"
FONT = "DejaVu Sans, Arial, sans-serif"
R = json.load(open(os.path.join(HERE, "results_cnn.json")))
H = R["history"]


def T(x, y, s, size=15, col=INK, anchor="middle", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{col}" text-anchor="{anchor}" font-weight="{weight}" '
            f'stroke="#fffdf9" stroke-width="4" paint-order="stroke" stroke-linejoin="round">{s}</text>')


def save(name, w, h, body):
    with open(os.path.join(OUT, name + ".svg"), "w") as fh:
        fh.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
                 f'font-family="{FONT}"><rect width="{w}" height="{h}" fill="#fffdf9"/>{body}</svg>')
    print("wrote", name)


def axes(x0, y0, w, h, xt, yt, X, Y, xl, yl, fmt="{:g}"):
    s = ""
    for v in yt:
        s += f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x0 + w}" y2="{Y(v):.1f}" stroke="{GRID}"/>' + T(x0 - 8, Y(v) + 5, fmt.format(v), 13, MUTED, "end")
    for v in xt:
        s += T(X(v), y0 + h + 22, f"{v:,}" if isinstance(v, int) else v, 13, MUTED)
    s += f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" stroke="{INK}"/><line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}" stroke="{INK}"/>'
    s += T(x0 + w / 2, y0 + h + 48, xl, 15)
    s += f'<text x="{x0 - 58}" y="{y0 + h / 2}" font-size="15" fill="{INK}" text-anchor="middle" transform="rotate(-90 {x0 - 58} {y0 + h / 2})">{yl}</text>'
    return s


def progress():
    W, Hh = 940, 400
    x0, y0, w, h = 100, 30, 780, 290
    accs = [r["val_acc"] for r in H]
    lo = min(0.70, min(accs) - 0.01)
    hi = max(accs) + 0.01
    X = lambda i: x0 + (i - 1) / (len(H) - 1) * w
    Y = lambda a: y0 + h - (a - lo) / (hi - lo) * h
    yt = [round(lo + k * (hi - lo) / 4, 2) for k in range(5)]
    s = axes(x0, y0, w, h, [1, 8, 16, 24, len(H)], yt, X, Y, "network evaluated (8 initial, then one child per cycle)", "proxy validation accuracy", "{:.2f}")
    base = H[0]["val_acc"]
    s += f'<line x1="{x0}" y1="{Y(base):.1f}" x2="{x0 + w}" y2="{Y(base):.1f}" stroke="{GOLD}" stroke-width="2" stroke-dasharray="7 5"/>'
    s += T(x0 + w - 6, Y(base) + 18, f"mini-AlexNet {base:.3f}", 14, GOLD, "end", "bold")
    best, pts = 0, []
    for i, r in enumerate(H, 1):
        best = max(best, r["val_acc"])
        pts.append(f"{X(i):.1f},{Y(best):.1f}")
        col = MUTED if i <= 8 else BLUE
        s += f'<circle cx="{X(i):.1f}" cy="{Y(max(lo, r["val_acc"])):.1f}" r="5" fill="{col}" fill-opacity="0.7"/>'
    s += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{BURG}" stroke-width="3"/>'
    s += T(x0 + 10, y0 + 18, "dots: each network · line: best so far", 14, MUTED, "start")
    save("cnn-progress", W, Hh, s)


def tradeoff():
    W, Hh = 940, 400
    x0, y0, w, h = 100, 30, 780, 290
    pts = {}
    for r in H:
        pts[r["arch"]] = (r["params"], r["val_acc"])
    ps = [p for p, _ in pts.values()]
    accs = [a for _, a in pts.values()]
    import math
    lo_p, hi_p = 10 ** math.floor(math.log10(min(ps))), 10 ** math.ceil(math.log10(max(ps)))
    lo, hi = min(0.70, min(accs) - 0.01), max(accs) + 0.01
    X = lambda p: x0 + (math.log10(p) - math.log10(lo_p)) / (math.log10(hi_p) - math.log10(lo_p)) * w
    Y = lambda a: y0 + h - (a - lo) / (hi - lo) * h
    yt = [round(lo + k * (hi - lo) / 4, 2) for k in range(5)]
    s = axes(x0, y0, w, h, [], yt, X, Y, "trainable parameters (log scale)", "proxy validation accuracy", "{:.2f}")
    v = lo_p
    while v <= hi_p:
        lab = f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"
        s += T(X(v), y0 + h + 22, lab, 13, MUTED)
        s += f'<line x1="{X(v):.1f}" y1="{y0}" x2="{X(v):.1f}" y2="{y0 + h}" stroke="{GRID}"/>'
        v *= 10
    for p, a in pts.values():
        s += f'<circle cx="{X(p):.1f}" cy="{Y(max(lo, a)):.1f}" r="5" fill="{MUTED}" fill-opacity="0.6"/>'
    front, best = [], -1
    for p, a in sorted(pts.values()):
        if a > best:
            front.append((p, a)); best = a
    s += f'<polyline points="{" ".join(f"{X(p):.1f},{Y(a):.1f}" for p, a in front)}" fill="none" stroke="{BURG}" stroke-width="3"/>'
    for p, a in front:
        s += f'<circle cx="{X(p):.1f}" cy="{Y(a):.1f}" r="7" fill="{BURG}" stroke="#fffdf9" stroke-width="2"/>'
    p0, a0 = H[0]["params"], H[0]["val_acc"]
    s += f'<circle cx="{X(p0):.1f}" cy="{Y(a0):.1f}" r="9" fill="none" stroke="{GOLD}" stroke-width="3"/>'
    s += T(X(p0) + 12, Y(a0) + 22, "mini-AlexNet", 14, GOLD, "start", "bold")
    s += T(x0 + w - 6, y0 + h - 12, "burgundy: no network is both smaller and more accurate", 14, BURG, "end")
    save("cnn-tradeoff", W, Hh, s)
    return front


def architectures():
    W, Hh = 1000, 300
    rows = [("mini-AlexNet (start)", R["final"]["mini-AlexNet"]), ("evolved (best proxy score)", R["final"]["evolved"])]
    s = ""
    for k, (title, f) in enumerate(rows):
        y = 40 + k * 130
        s += T(20, y, f"{title}: {f['params']:,} parameters, test accuracy {f['test_acc']:.3f}", 16, INK, "start", "bold")
        arch = f["arch"]
        conv = arch[arch.index("[") + 1:arch.index("]")].split()
        fc = json.loads(arch[arch.index("fc") + 2:arch.index("drop")].strip())
        drop = arch[arch.index("drop") + 4:]
        x = 20
        for c in conv:
            pool = c.endswith("P")
            f_, k_ = c.rstrip("P").split("k")
            s += f'<rect x="{x}" y="{y + 16}" width="92" height="64" rx="6" fill="#f7e8ee" stroke="{BURG}" stroke-width="1.6"/>'
            s += T(x + 46, y + 42, f"conv {f_}", 14, BURG, weight="bold") + T(x + 46, y + 62, f"{k_}×{k_}", 13, INK)
            x += 98
            if pool:
                s += f'<rect x="{x}" y="{y + 30}" width="30" height="36" rx="4" fill="#eef1f8" stroke="{BLUE}"/>' + T(x + 15, y + 53, "P", 13, BLUE, weight="bold")
                x += 36
        for h_ in fc:
            s += f'<rect x="{x}" y="{y + 16}" width="78" height="64" rx="6" fill="#fbf3e4" stroke="{GOLD}" stroke-width="1.6"/>'
            s += T(x + 39, y + 42, f"FC {h_}", 14, GOLD, weight="bold") + T(x + 39, y + 62, f"drop {drop}", 12, MUTED)
            x += 84
        s += f'<rect x="{x}" y="{y + 16}" width="64" height="64" rx="6" fill="#eef6f3" stroke="{TEAL}" stroke-width="1.6"/>' + T(x + 32, y + 53, "10", 14, TEAL, weight="bold")
    s += T(W / 2, Hh - 10, "conv = convolution + ReLU · P = 2×2 max-pooling · FC = fully connected + ReLU · 10 = output logits", 13, MUTED)
    save("cnn-architectures", W, Hh, s)


def cnn_overview():
    """The mini-AlexNet drawn as a classic CNN diagram: feature maps shrink, channels grow, then dense layers."""
    W, Hh, cy = 1200, 330, 150
    stages = [  # (kind, channels, spatial size, top label, kernel, params, pooled after)
        ("in", 1, 28, "input", "", 0, False),
        ("conv", 32, 28, "conv 1", "5×5", 1 * 25 * 32 + 32, True),
        ("conv", 64, 14, "conv 2", "3×3", 32 * 9 * 64 + 64, True),
        ("conv", 64, 7, "conv 3", "3×3", 64 * 9 * 64 + 64, False),
        ("conv", 64, 7, "conv 4", "3×3", 64 * 9 * 64 + 64, False),
        ("conv", 32, 7, "conv 5", "3×3", 64 * 9 * 32 + 32, True),
        ("fc", 128, 0, "FC 6", "", 288 * 128 + 128, False),
        ("fc", 128, 0, "FC 7", "", 128 * 128 + 128, False),
        ("out", 10, 0, "output", "", 128 * 10 + 10, False),
    ]
    side = {28: 104, 14: 64, 7: 40}
    drawn = [(side[sz] + (ch * 0.55 if k == "conv" else 4)) if k in ("in", "conv") else 24 for k, ch, sz, *_ in stages]
    slots = [max(w + 36, 96) for w in drawn]
    gap = (W - 40 - sum(slots)) / (len(stages) - 1)
    s, x, total, prev_end = "", 20.0, 0, None
    for i, ((kind, ch, sz, lab, ker, p, pool), w, slot) in enumerate(zip(stages, drawn, slots)):
        total += p
        x0 = x + (slot - w) / 2
        if prev_end is not None:
            x1, x2 = prev_end, x0
            s += (f'<line x1="{x1 + 6:.1f}" y1="{cy}" x2="{x2 - 8:.1f}" y2="{cy}" stroke="{MUTED}" stroke-width="1.6"/>'
                  f'<path d="M{x2 - 9:.1f},{cy - 5} L{x2 - 2:.1f},{cy} L{x2 - 9:.1f},{cy + 5} z" fill="{MUTED}"/>')
            pk, pch, psz, plab, pker, pp, ppool = stages[i - 1]
            if ppool:
                s += T((x1 + x2) / 2, cy - 10, "pool", 12, BLUE, weight="bold")
            if pk == "conv" and kind == "fc":
                s += T((x1 + x2) / 2, cy + 22, "flatten", 12, BLUE, weight="bold")
        if kind in ("in", "conv"):
            a = side[sz]
            d = w - a
            fill, line = ("#efeae4", MUTED) if kind == "in" else ("#f7e8ee", BURG)
            y0 = cy - a / 2 + d / 2
            s += (f'<polygon points="{x0:.1f},{y0:.1f} {x0 + d:.1f},{y0 - d:.1f} {x0 + d + a:.1f},{y0 - d:.1f} {x0 + a:.1f},{y0:.1f}" fill="{fill}" stroke="{line}" stroke-width="1.4"/>'
                  f'<polygon points="{x0 + a:.1f},{y0:.1f} {x0 + a + d:.1f},{y0 - d:.1f} {x0 + a + d:.1f},{y0 + a - d:.1f} {x0 + a:.1f},{y0 + a:.1f}" fill="{fill}" stroke="{line}" stroke-width="1.4"/>'
                  f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{a}" height="{a}" fill="{"#f3efe9" if kind == "in" else "#fffdf9"}" stroke="{line}" stroke-width="1.6"/>')
            shape = f"{ch} × {sz}×{sz}"
        else:
            hgt = 150 if kind == "fc" else 46
            fill, line = ("#fbf3e4", GOLD) if kind == "fc" else ("#eef6f3", TEAL)
            s += f'<rect x="{x0:.1f}" y="{cy - hgt / 2:.1f}" width="{w}" height="{hgt}" rx="4" fill="{fill}" stroke="{line}" stroke-width="1.6"/>'
            shape = f"{ch} units" if kind == "fc" else "10 classes"
        col = {"in": INK, "conv": BURG, "fc": GOLD, "out": TEAL}[kind]
        mid = x0 + w / 2
        s += T(mid, cy - 98, lab, 15, col, weight="bold")
        if ker:
            s += T(mid, cy - 80, ker, 13, INK)
        s += T(mid, cy + 102, shape, 13, INK)
        s += T(mid, cy + 121, f"{p:,} weights" if p else "gray image", 12, MUTED)
        prev_end = x0 + w
        x += slot + gap
    s += T(W / 2, Hh - 22, f"8 trained layers · {total:,} weights, every one fitted by gradient descent · "
                         "pool = 2×2 max-pooling, halves the image", 14, MUTED)
    save("cnn-overview", W, Hh, s)
    return total


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("cnn-overview weights:", cnn_overview())
    progress()
    print("front:", tradeoff())
    architectures()
