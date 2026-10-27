"""Aggregate the 12 float32-vs-float64 quantization-noise control runs."""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results_quant")


def ms(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    return st.mean(vals), st.pstdev(vals)


def main():
    files = sorted(glob.glob(os.path.join(RES, "quant_*.json")))
    rows = [json.load(open(f)) for f in files]
    print(f"loaded {len(rows)} runs\n")

    alphas = sorted({a["alpha"] for d in rows for a in d["param_audit"]})

    # ---------- Part A: parameter-space quantization audit ----------
    print("=== Part A: parameter-space quantization audit (mean ± sd over 12) ===")
    print(f"{'alpha':<9}{'per-coord/ULP':<15}{'f32 norm×':<16}{'f32 cos':<14}"
          f"{'f32 %changed':<14}{'f64 norm×':<12}{'f64 cos':<10}")
    audit = {}
    for al in alphas:
        def col(field):
            return ms([next(a[field] for a in d["param_audit"]
                            if a["alpha"] == al) for d in rows])
        ratio = ms([next(a["requested_per_coord_rms"] / a["f32_ULP_at_1"]
                         for a in d["param_audit"] if a["alpha"] == al)
                    for d in rows])
        n32, c32, ch32 = col("f32_applied_over_requested_norm"), \
            col("f32_direction_cos_to_v"), col("f32_fraction_params_changed")
        n64, c64 = col("f64_applied_over_requested_norm"), \
            col("f64_direction_cos_to_v")
        audit[al] = dict(coord_over_ULP=ratio, f32_norm=n32, f32_cos=c32,
                         f32_changed=ch32, f64_norm=n64, f64_cos=c64)
        print(f"{al:<9.0e}{ratio[0]:<15.3e}"
              f"{n32[0]:.3f}±{n32[1]:<10.3f}{c32[0]:.3f}±{c32[1]:<9.3f}"
              f"{ch32[0]*100:.2f}%±{ch32[1]*100:<8.2f}"
              f"{n64[0]:.4f}      {c64[0]:.4f}")

    # ---------- Part B: network flips f32 vs f64 ----------
    print("\n=== Part B: network activation flips (mean ± sd over 12) ===")
    print(f"{'alpha':<9}{'f32 %flip':<18}{'f64 %flip':<18}"
          f"{'f32 units':<16}{'f64 units':<14}{'units f32/f64':<14}")
    net = {}
    for al in alphas:
        key = f"{al:.0e}" if al != 0.001 else "0.001"
        def col(prec, field):
            return ms([d[prec][key][field] for d in rows])
        f32p, f64p = col("f32_flips", "sample_flip_fraction"), \
            col("f64_flips", "sample_flip_fraction")
        f32u, f64u = col("f32_flips", "mean_flip_units_per_sample_per_dir"), \
            col("f64_flips", "mean_flip_units_per_sample_per_dir")
        infl = f32u[0] / f64u[0] if f64u[0] > 1e-9 else float("inf")
        net[al] = dict(f32_flip=f32p, f64_flip=f64p, f32_units=f32u,
                       f64_units=f64u, inflation=infl)
        print(f"{al:<9.0e}{f32p[0]*100:6.2f}%±{f32p[1]*100:<9.2f}"
              f"{f64p[0]*100:6.2f}%±{f64p[1]*100:<9.2f}"
              f"{f32u[0]:7.2f}±{f32u[1]:<8.2f}{f64u[0]:7.3f}±{f64u[1]:<7.3f}"
              f"{infl:<14.1f}")

    # headline
    print("\n=== headline ===")
    for al in (1e-7, 1e-3):
        z = net[al]
        print(f"  alpha={al:.0e}: f32 reports {z['f32_flip'][0]*100:.1f}% "
              f"images / {z['f32_units'][0]:.1f} units, f64 reports "
              f"{z['f64_flip'][0]*100:.1f}% / {z['f64_units'][0]:.2f} units "
              f"(unit-count inflation {z['inflation']:.0f}x)")

    out = {
        "partA": {str(k): {kk: vv for kk, vv in v.items()}
                  for k, v in audit.items()},
        "partB": {str(k): {kk: (list(vv) if isinstance(vv, tuple) else vv)
                           for kk, vv in v.items()} for k, v in net.items()},
    }
    json.dump(out, open(os.path.join(RES, "quant_summary.json"), "w"),
              indent=2, default=str)
    print("\nsaved quant_summary.json")


if __name__ == "__main__":
    main()
