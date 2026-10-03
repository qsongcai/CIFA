#!/usr/bin/env python3
# VLCS discriminant experiment: do feature-statistic mixing methods
# (MixStyle / DSU, with and without SWAD) help on VLCS?
#
# Independent unit = seed (n=10, seeds 0-9). Conventions match the existing
# PACS / paired analysis exactly:
#   * per-seed overall = macro mean over the 4 held-out target domains of
#     target_acc*100;
#   * per-seed own-worst = min over the 4 held-out target domains of
#     target_acc*100;
#   * per-domain = target_acc*100 on that held-out target, paired over seeds.
# Every contrast is a SEED-LEVEL paired difference. We report mean, sample
# SD, parametric t-95% CI, nonparametric paired bootstrap 95% CI
# (np.random.RandomState(20261002), B=20000, resampling seeds), paired-t p,
# Wilcoxon signed-rank p, positive-seed count and exact sign-test p.
import glob
import json
import os
import re
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "CIFA-Repro", "results")
DOMAINS = ["CALTECH", "LABELME", "PASCAL", "SUN"]
SEEDS = list(range(10))
B = 20000
BOOT_SEED = 20261002

# arm name -> (location kind, directory under results)
#   ("ds", sub)  -> results/vlcs/<sub>
#   ("erm", ...)  -> results/erm/vlcs
ARM_DIR = {
    "erm": ("erm", os.path.join("erm", "vlcs")),
    "cifa": ("ds", os.path.join("vlcs", "cifa")),
    "swad": ("ds", os.path.join("vlcs", "swad")),
    "mixstyle": ("ds", os.path.join("vlcs", "mixstyle")),
    "dsu": ("ds", os.path.join("vlcs", "dsu")),
    "mixstyle_swad": ("ds", os.path.join("vlcs", "mixstyle_swad")),
    "dsu_swad": ("ds", os.path.join("vlcs", "dsu_swad")),
}


def load_cell(arm, dom, seed):
    kind, sub = ARM_DIR[arm]
    pat = os.path.join(RES, sub, f"*_{dom}_seed{seed}.json")
    fs = glob.glob(pat)
    if not fs:
        return np.nan
    return json.load(open(fs[0]))["target_acc"] * 100.0


def arm_matrix(arm):
    """Return matrix [domain index, seed] of target_acc (%)."""
    m = np.full((len(DOMAINS), len(SEEDS)), np.nan)
    for i, d in enumerate(DOMAINS):
        for j, s in enumerate(SEEDS):
            m[i, j] = load_cell(arm, d, s)
    return m


def seed_overall_worst(mat):
    """Per-seed macro overall and own-worst (columns = seeds)."""
    overall = mat.mean(axis=0)          # over domains, per seed
    worst = mat.min(axis=0)
    return overall, worst


# one shared RNG, used in the fixed contrast order below -> deterministic
rng = np.random.RandomState(BOOT_SEED)


def boot_ci(diff):
    idx = rng.randint(0, len(diff), size=(B, len(diff)))
    means = np.asarray(diff)[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def report(diff):
    d = np.asarray(diff, float)
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    tci = np.array([m - tcrit * se, m + tcrit * se])
    bci = boot_ci(d)
    t_p = stats.ttest_1samp(d, 0.0).pvalue
    nz = d[d != 0]
    try:
        w_p = stats.wilcoxon(nz).pvalue if len(nz) else 1.0
    except ValueError:
        w_p = 1.0
    pos = int((d > 0).sum())
    s_p = stats.binomtest(max(pos, n - pos), n, 0.5,
                          alternative="two-sided").pvalue
    return {
        "n": n, "mean": float(m), "sd": float(sd),
        "t_ci": [float(tci[0]), float(tci[1])],
        "boot_ci": [float(bci[0]), float(bci[1])],
        "t_p": float(t_p), "wilcoxon_p": float(w_p),
        "pos": pos, "sign_p": float(s_p),
    }


def fmt(r):
    return (f"{r['mean']:+6.2f}±{r['sd']:4.2f} "
            f"bootCI[{r['boot_ci'][0]:+5.2f},{r['boot_ci'][1]:+5.2f}] "
            f"tCI[{r['t_ci'][0]:+5.2f},{r['t_ci'][1]:+5.2f}] "
            f"t={r['t_p']:.3f} W={r['wilcoxon_p']:.3f} "
            f"sign={r['sign_p']:.3f} pos {r['pos']}/{r['n']}")


# ---- load all arms ----
mats = {a: arm_matrix(a) for a in ARM_DIR}
ow = {a: seed_overall_worst(m) for a, m in mats.items()}

# absolute summary per arm (context): overall mean +- sd and worst mean +- sd
absolute = {}
for a in ARM_DIR:
    ov, wo = ow[a]
    absolute[a] = {
        "overall_mean": float(ov.mean()), "overall_sd": float(ov.std(ddof=1)),
        "worst_mean": float(wo.mean()), "worst_sd": float(wo.std(ddof=1)),
        "per_domain": {
            d: {"mean": float(mats[a][i].mean()),
                "sd": float(mats[a][i].std(ddof=1))}
            for i, d in enumerate(DOMAINS)
        },
    }

# ---- define contrasts ----
# net contribution of feature-stat mixing over the no-mixing baseline
net_contrasts = {
    "mixstyle - erm": ("mixstyle", "erm"),
    "dsu - erm": ("dsu", "erm"),
    "mixstyle_swad - swad": ("mixstyle_swad", "swad"),
    "dsu_swad - swad": ("dsu_swad", "swad"),
}
# CIFA vs each mixing arm
cifa_contrasts = {
    "cifa - mixstyle": ("cifa", "mixstyle"),
    "cifa - dsu": ("cifa", "dsu"),
    "cifa - mixstyle_swad": ("cifa", "mixstyle_swad"),
    "cifa - dsu_swad": ("cifa", "dsu_swad"),
}


def contrast_tables(contrasts):
    out = {}
    for name, (a, b) in contrasts.items():
        ao, aw = ow[a]
        bo, bw = ow[b]
        entry = {
            "overall": report(ao - bo),
            "worst": report(aw - bw),
            "per_domain": {},
        }
        for i, d in enumerate(DOMAINS):
            entry["per_domain"][d] = report(mats[a][i] - mats[b][i])
        out[name] = entry
    return out


net = contrast_tables(net_contrasts)
cif = contrast_tables(cifa_contrasts)


def print_block(title, tables):
    print("=" * 78)
    print(title)
    print("=" * 78)
    for name, e in tables.items():
        print(f"\n### {name}")
        print(f"  overall : {fmt(e['overall'])}")
        print(f"  ownworst: {fmt(e['worst'])}")
        for d in DOMAINS:
            print(f"  {d:8s}: {fmt(e['per_domain'][d])}")


print("\n############ VLCS ABSOLUTE (per-seed overall / own-worst mean±SD) ############")
for a in ["erm", "mixstyle", "dsu", "swad", "mixstyle_swad", "dsu_swad", "cifa"]:
    print(f"  {a:14s} overall {absolute[a]['overall_mean']:6.2f}±{absolute[a]['overall_sd']:4.2f}"
          f"   worst {absolute[a]['worst_mean']:6.2f}±{absolute[a]['worst_sd']:4.2f}")

print_block("NET CONTRIBUTION OF FEATURE-STAT MIXING (seed unit, n=10)", net)
print_block("CIFA vs FEATURE-STAT MIXING (seed unit, n=10)", cif)

out_path = os.path.join(HERE, "vlcs_discriminant_stats.json")
json.dump({
    "dataset": "vlcs", "unit": "seed", "n_seeds": 10, "seeds": SEEDS,
    "bootstrap": {"B": B, "seed": BOOT_SEED, "method": "paired seed resample"},
    "absolute": absolute,
    "net_contribution": net,
    "cifa_vs_mixing": cif,
}, open(out_path, "w"), indent=2)
print("\nsaved", out_path)

# also place a copy in the repro tables dir
tbl = os.path.join(RES, "tables", "vlcs_discriminant_stats.json")
json.dump({
    "dataset": "vlcs", "unit": "seed", "n_seeds": 10, "seeds": SEEDS,
    "bootstrap": {"B": B, "seed": BOOT_SEED, "method": "paired seed resample"},
    "absolute": absolute,
    "net_contribution": net,
    "cifa_vs_mixing": cif,
}, open(tbl, "w"), indent=2)
print("saved", tbl)
