"""Summarize kappa*r^2 / kink analysis JSONs across domains and seeds.

Usage: python kappa_summary.py analysis/kappa_*.json
"""
import argparse
import glob
import json
import math
from collections import defaultdict


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def show_one(r):
    print("=" * 70)
    print(f"{r['target_domain']:>14} seed={r['seed']}  "
          f"captured={r['n_captured_epochs']} window={r['window_epochs']} "
          f"(n={r['n_valley_points']})")
    print(f"  r2={r['r2']:.5f}  r={r['r']:.5f}")
    print(f"  rms_gap={r['rms_gap_h_ens_minus_h_bar']:.5f}  "
          f"rms_logits={r['rms_logits']:.4f}  "
          f"gap/logits={r['gap_over_logits']:.5f}  kappa_eff={r['kappa_eff']:.5f}")
    print(f"  kink_frac@a1={r['kink_sample_fraction_at_alpha1']:.4f}  "
          f"frac_linear_piece={r['fraction_sample_direction_with_linear_piece']:.4f}")
    print(f"  rms_gap LINEAR={r['rms_gap_linear_samples']:.6f}  "
          f"KINK={r['rms_gap_kink_samples']:.6f}  "
          f"SMOOTH(adaptive)={r['rms_smooth_curvature_gap_adaptive']:.6f}")
    lc = r.get("linear_cell_global_radius")
    ratio = r.get("swad_window_to_linear_cell_ratio")
    print(f"  linear-cell: alpha<={r.get('linear_cell_alpha')}  "
          f"global_radius={lc:.2e}" if lc is not None
          else "  linear-cell: none (all flips even at finest radius)",
          f"  window/linear-cell={ratio:.2e}" if ratio else "")
    print("  alpha: kink_frac | mean_flip_units | interior_quad")
    for a, v in r["alpha_scan"].items():
        q = v.get("rms_interior_quadratic_on_linear_piece")
        qs = f"{q:.6f}" if q is not None else "  --    "
        print(f"    {a:>6}: {v['cross_kink_fraction']:.4f} | "
              f"{v['mean_flip_units']:12.2f} | {qs}")
    print(f"  loglog slope(gap vs r) = {r['loglog_slope_gap_vs_r']}")
    print("  nested radius sweep:")
    for s in r["radius_sweep"]:
        print(f"    n={s['n_points']:2d} r={s['r']:.4f} "
              f"rms_gap={s['rms_gap']:.5f} gap/logits={s['gap_over_logits']:.5f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*")
    args = p.parse_args()
    files = args.files or sorted(glob.glob("analysis/kappa_*.json"))
    rows = [json.load(open(f)) for f in files]
    rows.sort(key=lambda r: (r["target_domain"], r["seed"]))
    for r in rows:
        show_one(r)

    # aggregate by domain then overall
    keys = [
        ("r", "r"),
        ("gap/logits", "gap_over_logits"),
        ("kappa_eff", "kappa_eff"),
        ("kink_frac@a1", "kink_sample_fraction_at_alpha1"),
        ("frac_linear_piece", "fraction_sample_direction_with_linear_piece"),
        ("SMOOTH(adaptive)", "rms_smooth_curvature_gap_adaptive"),
        ("win/linear", "swad_window_to_linear_cell_ratio"),
    ]
    by_dom = defaultdict(list)
    for r in rows:
        by_dom[r["target_domain"]].append(r)

    print("=" * 70)
    print("AGGREGATE  mean +/- SD  (n seeds)")
    hdr = f"{'domain':>14} | " + " | ".join(f"{k:>18}" for k, _ in keys)
    print(hdr)
    for dom, rs in sorted(by_dom.items()):
        vals = []
        for _, kk in keys:
            xs = [r[kk] for r in rs if r[kk] is not None]
            vals.append(f"{mean(xs):>9.5f}+-{sd(xs):.4f}")
        print(f"{dom:>14} | " + " | ".join(f"{v:>18}" for v in vals))
    if len(by_dom) > 1:
        vals = []
        for _, kk in keys:
            xs = [r[kk] for r in rows if r[kk] is not None]
            vals.append(f"{mean(xs):>9.5f}+-{sd(xs):.4f}")
        print(f"{'OVERALL':>14} | " + " | ".join(f"{v:>18}" for v in vals))


if __name__ == "__main__":
    main()
