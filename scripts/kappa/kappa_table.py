"""Per-run compact table for the 12 R1-6 kappa runs (CSV + console)."""
import csv
import glob
import json
import sys

FLIP = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0]


def frac(r, a):
    return r["alpha_scan"][str(a)]["cross_kink_fraction"]


def units(r, a):
    return r["alpha_scan"][str(a)]["mean_flip_units"]


def first_full_flip(r):
    for a in FLIP:
        if frac(r, a) >= 0.999:
            return a
    return None


rows = []
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1
                          else "analysis/kappa_*.json")):
    rows.append(json.load(open(f)))
rows.sort(key=lambda r: (r["target_domain"], r["seed"]))

hdr = ["domain", "seed", "win_ep", "n_pts", "r", "gap/logits", "kappa_eff",
       "loglog_slope", "flip@1e-9", "flip@1e-8", "flip@1e-7",
       "units@1e-7", "first_full_flip_a",
       "gap_n2", "gap_n4", "gap_largest"]
out = []
for r in rows:
    sw = r["radius_sweep"]
    g2 = next((s["gap_over_logits"] for s in sw if s["n_points"] == 2), None)
    g4 = next((s["gap_over_logits"] for s in sw if s["n_points"] == r["n_valley_points"]), None)
    gl = sw[-1]["gap_over_logits"] if sw else None
    out.append([
        r["target_domain"], r["seed"],
        "-".join(map(str, r["window_epochs"])), r["n_valley_points"],
        round(r["r"], 4), round(r["gap_over_logits"], 4),
        round(r["kappa_eff"], 4), round(r["loglog_slope_gap_vs_r"], 3),
        round(frac(r, 1e-9), 4), round(frac(r, 1e-8), 4),
        round(frac(r, 1e-7), 4), round(units(r, 1e-7), 1),
        first_full_flip(r),
        round(g2, 4), round(g4, 4), round(gl, 4)])

print(",".join(hdr))
for row in out:
    print(",".join(str(x) for x in row))

with open("kappa_per_run.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(hdr)
    w.writerows(out)
print("\nsaved kappa_per_run.csv", file=sys.stderr)

# summary stats over the 12 runs for a few key scalars
import statistics as st
for col, idx in [("r", 4), ("gap/logits", 5), ("kappa_eff", 6),
                 ("loglog_slope", 7)]:
    xs = [row[idx] for row in out]
    print(f"{col:>14}: mean={st.mean(xs):.4f} sd={st.pstdev(xs):.4f} "
          f"min={min(xs):.4f} max={max(xs):.4f}", file=sys.stderr)
