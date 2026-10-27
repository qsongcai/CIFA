"""Aggregate the 12 float64 reinforcement runs (PACS, 4 domains x 3 seeds)."""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results_f64")


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = st.mean(vals)
    s = st.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s, min(vals), max(vals)


def fmt(t):
    if not t:
        return "NA"
    if abs(t[0]) < 0.01 or abs(t[0]) >= 1e4:
        return f"{t[0]:.3e}±{t[1]:.1e} [{t[2]:.2e},{t[3]:.2e}]"
    return f"{t[0]:.4f}±{t[1]:.4f} [{t[2]:.4f},{t[3]:.4f}]"


def interpolated_cell(d, target=0.5):
    """Log-alpha where flip fraction crosses target, via log interpolation."""
    import math
    r = d["relu"]
    keys = sorted(r.keys(), key=lambda s: float(s))
    pts = [(float(k), r[k]["flip_fraction"]) for k in keys]
    for i in range(len(pts) - 1):
        a0, f0 = pts[i]
        a1, f1 = pts[i + 1]
        if f0 < target <= f1:
            la = math.log(a0) + (math.log(a1) - math.log(a0)) * \
                (target - f0) / (f1 - f0)
            return math.exp(la) * d["mean_dt"]
    return None


def fit_slope_native(pts):
    import math
    pts = [(math.log(a), math.log(v)) for a, v in pts if v]
    if len(pts) < 2:
        return None
    mx = sum(a for a, _ in pts) / len(pts)
    my = sum(b for _, b in pts) / len(pts)
    den = sum((a - mx) ** 2 for a, _ in pts)
    return sum((a - mx) * (b - my) for a, b in pts) / den if den else None


def main():
    files = sorted(glob.glob(os.path.join(RES, "f64_kappa_*.json")))
    rows = []
    for f in files:
        d = json.load(open(f))
        rows.append(d)
    print(f"loaded {len(rows)} runs\n")

    # per-run derived quantities
    derived = []
    for d in rows:
        dt = d["mean_dt"]
        r = d["relu"]
        n_noflip_1e7 = r["1e-07"]["N_over_logits_noflip"]
        n_noflip_1e6 = r["1e-06"]["N_over_logits_noflip"]
        eps_1e7 = 1e-7 * dt
        c2 = n_noflip_1e7 / eps_1e7 ** 2 if n_noflip_1e7 else None
        cell_eps = interpolated_cell(d, 0.5)
        cell_quad = c2 * cell_eps ** 2 if (c2 and cell_eps) else None
        win = d["window_gap_N_over_2_at_alpha1"]
        # kink slope on every alpha with ANY flipped samples (1e-7..1e-5)
        fpts = [(float(a), r[a]["N_over_logits_flip"])
                for a in r if float(a) <= 1e-5 and r[a]["N_over_logits_flip"]]
        s_flip = fit_slope_native(fpts)
        # GELU slope on resolved points
        if d["gelu"]:
            gpts = [(float(a), d["gelu"][a]["N_over_logits_gelu"])
                    for a in d["gelu"]
                    if d["gelu"][a]["N_over_logits_gelu"]
                    and d["gelu"][a]["N_over_logits_gelu"] >
                    10 * d["empirical_floor_N_over_logits"]]
            s_gelu = fit_slope_native(gpts)
        else:
            s_gelu = None
        derived.append({
            "d": d, "cell_eps": cell_eps, "cell_quad": cell_quad,
            "c2": c2, "n_noflip_1e6": n_noflip_1e6,
            "ratio": win / cell_quad if cell_quad else None,
            "s_flip": s_flip, "s_gelu": s_gelu,
        })

    print(f"{'domain':<14}{'seed':<5}{'cell_eps':<10}{'noflip@1e-6':<12}"
          f"{'cell_quad':<11}{'win_gap':<9}{'ratio':<10}{'s_flip':<8}{'s_gelu':<7}")
    for z in derived:
        d = z["d"]
        print(f"{d['target_domain']:<14}{d['seed']:<5}{z['cell_eps']:<10.2e}"
              f"{z['n_noflip_1e6'] or 0:<12.2e}{z['cell_quad'] or 0:<11.2e}"
              f"{d['window_gap_N_over_2_at_alpha1']:<9.4f}"
              f"{z['ratio'] or 0:<10.2e}{(z['s_flip'] or 0):<8.2f}"
              f"{(z['s_gelu'] or 0):<7.2f}")

    # aggregate summaries
    print("\n=== aggregate across 12 runs (mean ± sd [min,max]) ===")
    def col(get):
        return ms([get(d) for d in rows])

    def colz(get):
        return ms([get(z) for z in derived])

    summary = {
        "dt": col(lambda d: d["mean_dt"]),
        "floor_N_over_logits": col(lambda d: d["empirical_floor_N_over_logits"]),
        "flip_fraction_1e-8": col(lambda d: d["relu"]["1e-08"]["flip_fraction"]),
        "flip_fraction_1e-7": col(lambda d: d["relu"]["1e-07"]["flip_fraction"]),
        "flip_fraction_1e-6": col(lambda d: d["relu"]["1e-06"]["flip_fraction"]),
        "flip_fraction_1e-5": col(lambda d: d["relu"]["1e-05"]["flip_fraction"]),
        "cell_radius_eps_interp": colz(lambda z: z["cell_eps"]),
        "noflip_N_measured_at_1e-6": colz(lambda z: z["n_noflip_1e6"]),
        "cell_quad_bound_interp": colz(lambda z: z["cell_quad"]),
        "c2_upper_bound": colz(lambda z: z["c2"]),
        "window_gap_N2": col(lambda d: d["window_gap_N_over_2_at_alpha1"]),
        "window_to_cell_quad_ratio_interp": colz(lambda z: z["ratio"]),
        "slope_flip_kink": colz(lambda z: z["s_flip"]),
        "slope_gelu_smooth": colz(lambda z: z["s_gelu"]),
        "gelu_1e-5_to_floor": col(
            lambda d: (d["gelu"]["1e-05"]["N_over_logits_gelu"]
                       / d["empirical_floor_N_over_logits"])
            if d["gelu"] else None),
        "flip_N_at_1e-7": col(lambda d: d["relu"]["1e-07"]["N_over_logits_flip"]),
        "noflip_N_at_1e-7": col(lambda d: d["relu"]["1e-07"]["N_over_logits_noflip"]),
        "flip_over_noflip_ratio_at_1e-7": col(
            lambda d: (d["relu"]["1e-07"]["N_over_logits_flip"]
                       / d["relu"]["1e-07"]["N_over_logits_noflip"])
            if d["relu"]["1e-07"]["N_over_logits_flip"]
            and d["relu"]["1e-07"]["N_over_logits_noflip"] else None),
    }
    for k, t in summary.items():
        print(f"  {k:<36} {fmt(t)}")

    out = {
        "n_runs": len(rows),
        "summary": {k: ({"mean": t[0], "sd": t[1], "min": t[2], "max": t[3]}
                       if t else None) for k, t in summary.items()},
        "per_run": [{"domain": d["target_domain"], "seed": d["seed"],
                     "dt": d["mean_dt"]} for d in rows],
    }
    outp = os.path.join(RES, "f64_summary.json")
    json.dump(out, open(outp, "w"), indent=2)
    print("\nsaved", outp)


if __name__ == "__main__":
    main()
