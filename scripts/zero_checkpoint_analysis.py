#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R3-(9): zero-checkpoint SWAD behavior.

Fallback (train.py L225-226): the averaged weights are loaded only if
`averaged is not None and averaged.n > 0`; when the SWAD loss valley never
opens (n_checkpoints == 0), the run is evaluated with the LAST-EPOCH model
(no dense weight averaging). For CIFA that is a CIFA-augmented final model;
for the SWAD arm it is effectively a plain-ERM final model.

We (1) count zero-checkpoint runs per benchmark/arm, (2) recompute the paired
CIFA-SWAD comparison after removing every domain-seed cell where EITHER arm
has zero averaged checkpoints, and (3) compare against the all-cell result.
"""
import json, os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))

CFG = {
    "PACS(unified)": ("baselines_20261004/results_classic",
                      ["art_painting", "cartoon", "photo", "sketch"], range(10), "rn"),
    "PACS(reference)": ("CIFA-Repro/results/pacs",
                        ["art_painting", "cartoon", "photo", "sketch"], range(10), "rn"),
    "VLCS": ("CIFA-Repro/results/vlcs",
             ["CALTECH", "LABELME", "PASCAL", "SUN"], range(10), "rn"),
    "OfficeHome": ("CIFA-Repro/results/officehome",
                   ["Art", "Clipart", "Product", "Real World"], range(10), "rn"),
    "Terra": ("CIFA-Repro/results/terra",
              ["location_100", "location_38", "location_43", "location_46"], range(10), "rn"),
    "PACS ViT": ("vit_e2e_results",
                 ["art_painting", "cartoon", "photo", "sketch"], range(5), "vit"),
}


def path(root, m, d, s, kind):
    if kind == "vit":
        return os.path.join(ROOT, root, f"{m}_{d}_seed{s}_block6.json")
    return os.path.join(ROOT, root, m, f"{m}_{d}_seed{s}.json")


def load(root, m, d, s, kind):
    j = json.load(open(path(root, m, d, s, kind)))
    return 100.0 * j["target_acc"], int(j.get("swad_n_checkpoints", -1))


def t_tost(diff, margin=1.0):
    diff = np.asarray(diff, float)
    n = len(diff); mean, sd = diff.mean(), diff.std(ddof=1)
    se = sd / np.sqrt(n); tc = stats.t.ppf(0.95, n - 1)
    lo, hi = mean - tc * se, mean + tc * se
    p = max(stats.t.sf((mean + margin) / se, n - 1),
            stats.t.cdf((mean - margin) / se, n - 1))
    p2 = stats.ttest_1samp(diff, 0).pvalue
    return mean, sd, lo, hi, p2, p, int((diff > 0).sum()), int((diff < 0).sum())


print(f"{'benchmark':16s} {'arm':5s} {'zero/40':>9s} {'mean ckpt':>10s} {'median':>7s} {'max':>4s}")
summary = {}
for name, (root, doms, seeds, kind) in CFG.items():
    ntotal = len(doms) * len(list(seeds))
    zc = {}
    for arm in ("cifa", "swad"):
        z = []; ncks = []
        for d in doms:
            for s in seeds:
                _, nc = load(root, arm, d, s, kind)
                ncks.append(nc)
                if nc == 0:
                    z.append((d, s))
        zc[arm] = z
        print(f"{name:16s} {arm:5s} {len(z):4d}/{ntotal:<4d} {np.mean(ncks):10.2f} "
              f"{int(np.median(ncks)):7d} {max(ncks):4d}")
    # cell-level paired analysis
    def cells(keep):
        macro = []
        for s in seeds:
            diffs = []
            for d in doms:
                ca, cn = load(root, "cifa", d, s, kind)
                sa, sn = load(root, "swad", d, s, kind)
                ok = (cn > 0 and sn > 0) if keep else True
                if ok:
                    diffs.append(ca - sa)
            if diffs:
                macro.append(np.mean(diffs))
        return np.array(macro)
    all_m = cells(False)
    keep_m = cells(True)
    summary[name] = (all_m, keep_m, zc)
    print()

print("\n========== CIFA-SWAD overall macro: all cells vs only both-arms-averaged cells ==========")
print(f"{'benchmark':16s} {'set':>8s} {'n':>3s} {'Δ':>7s} {'90% CI':>18s} {'pair-p':>8s} {'±1 TOST':>8s} {'win/loss':>10s}")
for name, (all_m, keep_m, _) in summary.items():
    for lab, mm in [("all", all_m), ("kept", keep_m)]:
        if len(mm) < 3:
            print(f"{name:16s} {lab:>8s} n={len(mm)} (too few)")
            continue
        mean, sd, lo, hi, p2, p1, w, l = t_tost(mm, 1.0)
        print(f"{name:16s} {lab:>8s} {len(mm):3d} {mean:+7.2f}   [{lo:+6.2f},{hi:+6.2f}] {p2:8.3f} {p1:8.3f} {w:4d}/{l:<4d}")
    print()
