#!/usr/bin/env python3
"""Recompute the JSON audit tables straight from the per-run result files.

Outputs (under results/tables/):
  * three_arm_recomputed.json -- per-domain and macro overall mean+-std
    (ddof=1) for ERM / SWAD / CIFA on every benchmark;
  * summary_pacs.json -- paired CIFA-SWAD summary for PACS, including
    per-seed own-worst-domain rows and augmentation diagnostics.

PACS is aggregated over the seeds actually present (n=10: seeds 0-9);
the other benchmarks remain n=5. Pure standard library, deterministic.

    python scripts/recompute_json_tables.py
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "cifa"))
from datasets import DATASETS  # noqa: E402

DATASET_ORDER = ["pacs", "vlcs", "officehome", "terra"]


def mean(xs):
    return sum(xs) / len(xs)


def sample_std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def classify(path):
    parts = path.split(os.sep)
    if "erm" in parts:
        ds = next((p for p in parts if p in DATASET_ORDER), None)
        return ("erm", ds)
    ds = next((p for p in parts if p in DATASET_ORDER), None)
    method = parts[-2] if parts[-2] in ("cifa", "swad") else None
    return (method, ds)


def load_tree():
    """main[ds][method][domain][seed] -> acc (%) ; plus raw records."""
    main = {d: {} for d in DATASET_ORDER}
    raw = {d: {"cifa": {}, "swad": {}} for d in DATASET_ORDER}
    for fn in glob.glob(os.path.join(RES, "**", "*.json"), recursive=True):
        if "ablation" in fn.split(os.sep) or "tables" in fn.split(os.sep):
            continue
        with open(fn) as f:
            r = json.load(f)
        if "target_acc" not in r:
            continue
        method, ds = classify(fn)
        if ds is None or method is None:
            continue
        main[ds].setdefault(method, {}).setdefault(
            r["target_domain"], {})[r["seed"]] = r["target_acc"] * 100
        if method in ("cifa", "swad"):
            raw[ds][method][(r["target_domain"], r["seed"])] = r
    return main, raw


def arm_summary(cells, domains):
    """Per-domain mean/std and per-seed macro overall mean/std."""
    seeds = sorted({s for d in domains for s in cells.get(d, {})})
    out_domains = {}
    for d in domains:
        vals = [cells[d][s] for s in sorted(cells.get(d, {}))]
        m, sd = mean(vals), sample_std(vals)
        out_domains[d] = {"mean": m, "std": sd, "cell": f"{m:.2f}±{sd:.2f}"}
    per_seed = [mean([cells[d][s] for d in domains]) for s in seeds]
    m, sd = mean(per_seed), sample_std(per_seed)
    return {"overall_mean": m, "overall_std": sd, "overall": f"{m:.2f}±{sd:.2f}",
            "domains": out_domains, "_seeds": seeds}


def build_three_arm(main):
    out = {}
    for ds in DATASET_ORDER:
        domains = DATASETS[ds]["domains"]
        out[ds] = {}
        for method in ("erm", "swad", "cifa"):
            if method in main[ds]:
                s = arm_summary(main[ds][method], domains)
                out[ds][method] = {k: v for k, v in s.items() if not k.startswith("_")}
    return out


def build_pacs_summary(main, raw):
    domains = DATASETS["pacs"]["domains"]
    cifa_cells, swad_cells = main["pacs"]["cifa"], main["pacs"]["swad"]
    seeds = sorted({s for d in domains for s in cifa_cells.get(d, {})})

    per_domain = {}
    cifa_means, swad_means = [], []
    all_deltas = []
    for d in domains:
        deltas = [cifa_cells[d][s] - swad_cells[d][s] for s in seeds]
        cvals = [cifa_cells[d][s] for s in seeds]
        svals = [swad_cells[d][s] for s in seeds]
        cm, sm = mean(cvals), mean(svals)
        cifa_means.append(cm)
        swad_means.append(sm)
        all_deltas += deltas
        per_domain[d] = {
            "swad_mean": sm,
            "swad_std": sample_std(svals),
            "cifa_mean": cm,
            "cifa_std": sample_std(cvals),
            "delta_mean": mean(deltas),
            "delta_std": sample_std(deltas),
            "pos_seeds": sum(x > 0 for x in deltas),
            "deltas": deltas,
        }

    cifa_macro, swad_macro = mean(cifa_means), mean(swad_means)

    worst_rows = []
    for s in seeds:
        cworst_dom = min(domains, key=lambda d: cifa_cells[d][s])
        sworst_dom = min(domains, key=lambda d: swad_cells[d][s])
        cw, sw = cifa_cells[cworst_dom][s], swad_cells[sworst_dom][s]
        worst_rows.append({
            "seed": s,
            "swad_worst_domain": sworst_dom,
            "swad_worst_acc": sw,
            "cifa_worst_acc": cw,
            "delta": cw - sw,
            "cifa_worst_domain": cworst_dom,
        })

    # augmentation diagnostics over CIFA cells that carry them
    gates, trigs, kls = [], [], []
    for (dom, s), r in sorted(raw["pacs"]["cifa"].items()):
        if r.get("aug_mean_gate") is not None:
            gates.append(r["aug_mean_gate"])
        if r.get("aug_trigger_fraction") is not None:
            trigs.append(r["aug_trigger_fraction"])
        if r.get("aug_mean_consistency_kl") is not None:
            kls.append(r["aug_mean_consistency_kl"])

    hardest = min(domains, key=lambda d: per_domain[d]["swad_mean"])
    return {
        "dataset": "pacs",
        "seeds": seeds,
        "summary": per_domain,
        "swad_macro": swad_macro,
        "cifa_macro": cifa_macro,
        "delta_macro": cifa_macro - swad_macro,
        "pooled_delta_mean": mean(all_deltas),
        "pooled_delta_std": sample_std(all_deltas),
        "positive_cells": sum(x > 0 for x in all_deltas),
        "n_cells": len(all_deltas),
        "worst_domain_per_seed": worst_rows,
        "gate_min": min(gates),
        "gate_mean": mean(gates),
        "trigger_mean": mean(trigs),
        "kl_mean": mean(kls),
        "n_diag_cells": len(gates),
        "hardest_domain": hardest,
    }


def write(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print("wrote", os.path.relpath(path, ROOT))


def main():
    main_tree, raw = load_tree()
    three_arm = build_three_arm(main_tree)
    write(os.path.join(RES, "tables", "three_arm_recomputed.json"), three_arm)
    pacs = build_pacs_summary(main_tree, raw)
    write(os.path.join(RES, "tables", "summary_pacs.json"), pacs)

    # console summary
    print("\nPACS n=%d: CIFA %.2f SWAD %.2f delta %+.2f | worst %+.2f (%d/%d)"
          % (len(pacs["seeds"]), pacs["cifa_macro"], pacs["swad_macro"],
             pacs["delta_macro"],
             mean([r["delta"] for r in pacs["worst_domain_per_seed"]]),
             sum(r["delta"] > 0 for r in pacs["worst_domain_per_seed"]),
             len(pacs["seeds"])))


if __name__ == "__main__":
    main()
