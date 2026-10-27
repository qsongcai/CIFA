#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-margin TOST (Schuirmann) for CIFA vs SWAD, corrected caliber.

TWO METRICS, following the manuscript exactly:
  * overall  = macro-average of the four target-domain target_acc values.
       - macro  : per-seed cross-domain mean (main table, n = #seeds)
       - pooled : every domain x seed pair (n = #domains x #seeds)
       - per-domain: within each target domain (n = #seeds)
  * worst    = OWN-WORST-DOMAIN: for each seed, each arm's accuracy on the
       target domain that the arm itself finds hardest, i.e. min over the four
       target-domain target_acc values, then the two arms' minima are differenced.
       This is a per-seed, cross-domain statistic (n = #seeds only); there is no
       pooled or per-domain version.
       (NOTE: the JSON field worst_holdout_acc is the worst *source holdout*
       accuracy used for SWAD selection and is NOT the reported worst-domain
       metric; it must not be used here.)

Margins: +/-0.5 / +/-1.0 / +/-2.0 percentage points.
Delta = CIFA - SWAD (points); positive favors CIFA.
"""
import json, os, csv
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
MARGINS = [0.5, 1.0, 2.0]


def rn_path(rt, m, d, s):
    return os.path.join(rt, m, f"{m}_{d}_seed{s}.json")


def vit_path(rt, m, d, s):
    return os.path.join(rt, f"{m}_{d}_seed{s}_block6.json")


def load(path, field):
    return 100.0 * json.load(open(path))[field]


def t_crit(n, cl=0.90):
    return stats.t.ppf(1 - (1 - cl) / 2, n - 1)


def tost(diff, margin):
    """diff: paired differences in points. Schuirmann two one-sided tests."""
    diff = np.asarray(diff, float)
    n = len(diff)
    mean, sd = diff.mean(), diff.std(ddof=1)
    se = sd / np.sqrt(n)
    tc = t_crit(n)
    lo, hi = mean - tc * se, mean + tc * se
    t_lower = (mean + margin) / se
    t_upper = (mean - margin) / se
    p_lower = stats.t.sf(t_lower, n - 1)
    p_upper = stats.t.cdf(t_upper, n - 1)
    p_tost = max(p_lower, p_upper)
    eq = (lo > -margin) and (hi < margin)
    return dict(mean=mean, sd=sd, n=n, lo=lo, hi=hi,
                p_tost=p_tost, equiv=eq)


def paired_t(diff):
    diff = np.asarray(diff, float)
    return stats.ttest_1samp(diff, 0.0).pvalue, int((diff > 0).sum()), int((diff < 0).sum())


def build_cfg(label, backbone, roots, kind, domains, seeds):
    """roots: list of (batch_name, root). kind: 'rn' or 'vit'.
    Returns dict method -> {(batch, domain, seed): overall acc}."""
    mats = {"cifa": {}, "swad": {}}
    for bname, rt in roots:
        for m in ("cifa", "swad"):
            for d in domains:
                for s in seeds:
                    p = vit_path(rt, m, d, s) if kind == "vit" else rn_path(rt, m, d, s)
                    mats[m][(bname, d, s)] = load(p, "target_acc")
    return mats


def add_config(rows, label, backbone, roots, kind, domains, seeds):
    M = build_cfg(label, backbone, roots, kind, domains, seeds)
    batches = [bn for bn, _ in roots]

    def val(m, b, d, s):
        return M[m][(b, d, s)]

    # ---- overall: paired differences per (batch, domain, seed) ----
    overall = {}
    for b in batches:
        for d in domains:
            for s in seeds:
                overall[(b, d, s)] = val("cifa", b, d, s) - val("swad", b, d, s)

    def emit(metric, scope, dom, diff):
        diff = np.asarray(diff, float)
        p2, w, l = paired_t(diff)
        r = dict(dataset=label, backbone=backbone, metric=metric, scope=scope,
                 domain=dom, n=len(diff), mean=diff.mean(), sd=diff.std(ddof=1),
                 p_pair=p2, wins=w, losses=l)
        for mg in MARGINS:
            t = tost(diff, mg)
            r[f"ci{mg}_lo"] = t["lo"]; r[f"ci{mg}_hi"] = t["hi"]
            r[f"p{mg}"] = t["p_tost"]; r[f"eq{mg}"] = "YES" if t["equiv"] else "NO"
        rows.append(r)

    # macro (per-seed cross-domain mean)
    emit("overall", "macro(n_seed)", "ALL",
         [np.mean([overall[(b, d, s)] for d in domains]) for b in batches for s in seeds])
    # pooled (domain x seed)
    emit("overall", "pooled(n_dom*seed)", "ALL",
         [overall[(b, d, s)] for b in batches for d in domains for s in seeds])
    # per-domain
    for d in domains:
        emit("overall", "per-domain", d,
             [overall[(b, d, s)] for b in batches for s in seeds])

    # ---- worst: own-worst-domain per seed (per-arm cross-target min) ----
    # pool across batches (2-batch -> n=20); one per-seed value per (batch, seed).
    worst_d = []
    for b in batches:
        for s in seeds:
            cifa_min = min(val("cifa", b, d, s) for d in domains)
            swad_min = min(val("swad", b, d, s) for d in domains)
            worst_d.append(cifa_min - swad_min)
    emit("worst", "own-worst(per-seed)", "ALL", worst_d)


CONFIGS = [
    ("PACS", "ResNet-50", [("uni", f"{ROOT}/baselines_20261004/results_classic")],
     "rn", ["art_painting", "cartoon", "photo", "sketch"], list(range(10))),
    ("PACS(2-batch)", "ResNet-50",
     [("ref", f"{ROOT}/CIFA-Repro/results/pacs"),
      ("uni", f"{ROOT}/baselines_20261004/results_classic")],
     "rn", ["art_painting", "cartoon", "photo", "sketch"], list(range(10))),
    ("VLCS", "ResNet-50", [("uni", f"{ROOT}/CIFA-Repro/results/vlcs")],
     "rn", ["CALTECH", "LABELME", "PASCAL", "SUN"], list(range(10))),
    ("OfficeHome", "ResNet-50", [("uni", f"{ROOT}/CIFA-Repro/results/officehome")],
     "rn", ["Art", "Clipart", "Product", "Real World"], list(range(10))),
    ("Terra", "ResNet-50", [("uni", f"{ROOT}/CIFA-Repro/results/terra")],
     "rn", ["location_100", "location_38", "location_43", "location_46"], list(range(10))),
    ("PACS", "ViT-B/16", [("uni", f"{ROOT}/vit_e2e_results")],
     "vit", ["art_painting", "cartoon", "photo", "sketch"], list(range(5))),
]

rows = []
for cfg in CONFIGS:
    add_config(rows, *cfg)

fields = ["dataset", "backbone", "metric", "scope", "domain", "n", "mean", "sd",
          "p_pair", "wins", "losses",
          "ci0.5_lo", "ci0.5_hi", "p0.5", "eq0.5",
          "ci1.0_lo", "ci1.0_hi", "p1.0", "eq1.0",
          "ci2.0_lo", "ci2.0_hi", "p2.0", "eq2.0"]
with open(os.path.join(ROOT, "tost_multi_margin_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)

# self-consistency: p-TOST criterion vs 90% CI criterion
mismatch = sum(1 for r in rows for mg in MARGINS
               if ((r[f"ci{mg}_lo"] > -mg and r[f"ci{mg}_hi"] < mg) != (r[f"p{mg}"] < 0.05)))
print(f"rows={len(rows)} checks={len(rows)*3} p-vs-CI mismatches={mismatch}\n")


def show(r):
    out = f"n={r['n']:2d} Δ={r['mean']:+.2f}({r['sd']:.2f}) pair-p={r['p_pair']:.3f} {r['wins']}/{r['losses']}"
    for mg in MARGINS:
        out += f" | ±{mg}: CI[{r[f'ci{mg}_lo']:+.2f},{r[f'ci{mg}_hi']:+.2f}] p={r[f'p{mg}']:.3f} {'YES' if r[f'eq{mg}']=='YES' else 'no'}"
    return out


ORDER = [("PACS", "ResNet-50"), ("PACS(2-batch)", "ResNet-50"), ("VLCS", "ResNet-50"),
         ("OfficeHome", "ResNet-50"), ("Terra", "ResNet-50"), ("PACS", "ViT-B/16")]
for label, bb in ORDER:
    print("=" * 112)
    print(f"{label}  ({bb})")
    print("=" * 112)
    for metric, scope in [("overall", "macro(n_seed)"), ("overall", "pooled(n_dom*seed)"),
                          ("worst", "own-worst(per-seed)")]:
        r = next(x for x in rows if x["dataset"] == label and x["backbone"] == bb
                 and x["metric"] == metric and x["scope"] == scope)
        print(f"  [{metric:7s} {scope:20s}] {show(r)}")
    pd = [x for x in rows if x["dataset"] == label and x["backbone"] == bb
          and x["metric"] == "overall" and x["scope"] == "per-domain"]
    for r in pd:
        print(f"    per-domain {r['domain']:14s} {show(r)}")
    print()
