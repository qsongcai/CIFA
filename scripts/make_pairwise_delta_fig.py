#!/usr/bin/env python3
# Pairwise per-domain accuracy difference (CIFA - SWAD). Seed count is
# auto-detected per domain (intersection of the two arms); PACS has 10 seeds.
# Produces an IEEE-style one-column dot plot (vector PDF + 300-dpi PNG).
import json, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "pdf.fonttype": 42,  # TrueType in PDF
    "ps.fonttype": 42,
})

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "CIFA-Repro", "results")
# PACS uses the authoritative unified-code batch; the other benchmarks have
# only one batch and are read from CIFA-Repro.
PACS_RES = os.path.join(ROOT, "baselines_20261004", "results_classic")
PACS_DOMS = ["art_painting", "cartoon", "photo", "sketch"]
summary = json.load(open(os.path.join(RES, "tables", "three_arm_recomputed.json")))

BENCH = [("pacs", "PACS"), ("vlcs", "VLCS"),
         ("officehome", "OfficeHome"), ("terra", "TerraIncognita")]

def short(bench, dom):
    m = {"art_painting": "Art", "cartoon": "Cartoon", "photo": "Photo",
         "sketch": "Sketch", "caltech": "Caltech", "labelme": "LabelMe",
         "pascal": "Pascal", "sun": "SUN", "art": "Art", "clipart": "Clipart",
         "product": "Product", "real_world": "Real"}
    key = dom.lower().replace(" ", "_")
    if key.startswith("location_"):
        return "L" + key.split("_")[1]
    return m.get(key, dom)

def res_root(bench):
    return PACS_RES if bench == "pacs" else RES

def load_acc(bench, method, dom, seed):
    root = res_root(bench)
    if bench == "pacs":
        pat = os.path.join(root, method, f"{method}_{dom}_seed{seed}.json")
    else:
        pat = os.path.join(root, bench, method, f"*_{dom}_seed{seed}.json")
    fs = glob.glob(pat)
    if not fs:
        return None
    return json.load(open(fs[0]))["target_acc"]

def domain_seeds(bench, dom):
    """Seeds present for a domain in both paired arms."""
    root = res_root(bench)
    def arm(method):
        out = set()
        if bench == "pacs":
            pat = os.path.join(root, method, f"{method}_{dom}_seed*.json")
        else:
            pat = os.path.join(root, bench, method, f"*_{dom}_seed*.json")
        for f in glob.glob(pat):
            m = re.search(r"seed(\d+)\.json$", os.path.basename(f))
            if m:
                out.add(int(m.group(1)))
        return out
    return sorted(arm("cifa") & arm("swad"))

rows = []          # (bench_label, dom_label, [deltas], mean, std)
for bkey, blabel in BENCH:
    doms = PACS_DOMS if bkey == "pacs" else list(summary[bkey]["cifa"]["domains"].keys())
    for dom in doms:
        deltas = []
        for s in domain_seeds(bkey, dom):
            c = load_acc(bkey, "cifa", dom, s)
            w = load_acc(bkey, "swad", dom, s)
            if c is not None and w is not None:
                deltas.append(100.0 * (c - w))
        rows.append((blabel, short(bkey, dom), np.array(deltas),
                     np.mean(deltas), np.std(deltas, ddof=1)))

# sanity print
print(f"{'Bench':<13}{'Domain':<12}{'mean':>7}{'std':>7}  n  pos")
for b, d, arr, m, sd in rows:
    print(f"{b:<13}{d:<12}{m:>7.2f}{sd:>7.2f}  {len(arr)}  {(arr>0).sum()}")

# ---- layout: 4 groups of 4 rows, with gaps between groups ----
n_per = 4
gap = 0.9
positions, labels, groups = [], [], []
idx = 0
for g in range(4):
    for k in range(n_per):
        positions.append(idx)
        labels.append(rows[g * n_per + k][1])
        idx += 1
    groups.append((g, idx - n_per - 0.5, idx - 0.5))
    idx += gap
positions = np.array(positions)

fig, ax = plt.subplots(figsize=(3.5, 5.3))

C_POS, C_NEG = "#1f6fb4", "#c0392b"
rng = np.random.default_rng(0)
for i, (b, d, arr, m, sd) in enumerate(rows):
    y = positions[i]
    # group shading
    # seed points with vertical jitter
    jit = rng.uniform(-0.16, 0.16, size=len(arr))
    for x, j in zip(arr, jit):
        ax.scatter(x, y + j, s=13,
                   color=C_POS if x >= 0 else C_NEG, alpha=0.55,
                   marker="o", linewidths=0, zorder=3)
    # error bar + mean
    ax.errorbar(m, y, xerr=sd, fmt="none", ecolor="black",
                elinewidth=0.9, capsize=2.4, zorder=4)
    ax.scatter(m, y, marker="D", s=20, facecolor="black",
               edgecolor="black", zorder=5)

# group bands + labels
xlo = -3.2
for g, y0, y1 in groups:
    ax.axhspan(y0, y1, color="#f2f2f2" if g % 2 == 0 else "#ffffff", zorder=0)
    ax.text(xlo + 0.12, (y0 + y1) / 2, BENCH[g][1],
            va="center", ha="left", fontsize=8, fontweight="bold",
            color="#444444", zorder=2)

ax.axvline(0, color="black", linewidth=1.0, zorder=2)

ax.set_yticks(positions)
ax.set_yticklabels(labels)
xlo, xhi = -3.2, 6.0
ax.set_xlim(xlo, xhi)
ax.set_ylim(positions[-1] + 0.7, positions[0] - 0.7)
ax.set_xlabel("Accuracy difference (CIFA $-$ SWAD), percentage points")
ax.tick_params(length=2)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#888888")
ax.spines["bottom"].set_color("#888888")

legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f7f7f",
           markersize=5, label="Individual seed"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="black",
           markersize=5.5, label="Mean over seeds"),
]
ax.legend(handles=legend, loc="center right", fontsize=7, frameon=True,
          framealpha=0.9, edgecolor="#cccccc")

fig.tight_layout()
out_pdf = os.path.join(os.path.dirname(RES), "fig_pairwise_delta.pdf")
out_png = os.path.join(os.path.dirname(RES), "fig_pairwise_delta.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print("saved:", out_pdf, out_png)
