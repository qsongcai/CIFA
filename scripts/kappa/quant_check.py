"""R1-6 float32 quantization-noise control for the activation-flip scan.

The float32 analysis (kappa_analyze.py) reported a sharp activation-pattern
transition at alpha=1e-7 (global eps~1.8e-7) and called that the "linear-cell
diameter".  The float64 run instead puts the true transition at alpha=1e-6..
1e-5.  This script proves the float32 transition is a ROUND-OFF artifact:

The window unit direction v is normalized over P~23M parameters, so each
coordinate of the requested perturbation is eps*v_i ~ eps/sqrt(P).  At
alpha=1e-7, eps~1.8e-7 and eps*v_i ~ 4e-11, which is ~3000x SMALLER than the
float32 unit-in-the-last-place of a weight of order 1 (ULP~1.2e-7).  When the
perturbation is applied in float32 (p.copy_(base + eps*unit)), the smooth
displacement cannot be represented: it becomes sparse, random +/-ULP
quantization jumps whose total magnitude and direction are unrelated to the
requested window direction.  Those jumps (not the true geometry) flip the
activations.  In float64 (ULP~2.2e-16) the same perturbation is represented
exactly.

Part A (parameter space, no network):
  - norm and direction fidelity of the actually-applied float32 displacement
  - fraction of parameters that change at all
  - same quantities for the float64 path (must be exact)
Part B (real ResNet forward, n=256):
  - per-sample flip fraction and mean flipping units at each alpha, compared
    between an exact float32 reproduction of kappa_analyze and the float64
    path.

Usage:
  python quant_check.py --ckpt ckpt/kappa_art_painting_seed0.pt \
      --out analysis/quant_art_painting_seed0.json
"""
import argparse
import json
import math

import torch
import torch.nn as nn

from datasets import DATASETS
from kappa_analyze import (build_model, get_batch, delta_from_sd, norm2,
                          FlipCollector)

ALPHAS = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-3]
BATCH = 64


def quant_audit(base, unit, dt, alphas):
    """Part A: compare requested vs actually-applied displacement."""
    rows = []
    P = sum(t.numel() for t in base.values())
    # aggregate over all parameters
    total_v2 = sum((v.double() ** 2).sum().item() for v in unit.values())
    for a in alphas:
        eps = a * dt
        req_sq = eps ** 2 * total_v2
        # float32 path: the exact expression kappa_analyze evaluates
        f32_applied_sq = 0.0
        f32_dot = 0.0
        changed = 0
        for n in base:
            b = base[n]
            pert = (b + eps * unit[n].float()).float()      # float32 add
            d = (pert.double() - b.double())
            f32_applied_sq += (d ** 2).sum().item()
            f32_dot += (d * unit[n].double()).sum().item()
            changed += (pert != b).sum().item()
        # float64 path
        f64_applied_sq = 0.0
        f64_dot = 0.0
        changed64 = 0
        for n in base:
            b = base[n].double()
            pert = b + eps * unit[n].double()
            d = pert - b
            f64_applied_sq += (d ** 2).sum().item()
            f64_dot += (d * unit[n].double()).sum().item()
            changed64 += (pert != b).sum().item()

        def cos(dot, sq):
            return dot / math.sqrt(sq * total_v2) if sq > 0 else 0.0

        rows.append({
            "alpha": a,
            "requested_eps": eps,
            "requested_per_coord_rms": eps / math.sqrt(P),
            "f32_ULP_at_1": 1.1920928955078125e-7,
            "f32_applied_over_requested_norm": math.sqrt(f32_applied_sq / req_sq),
            "f32_direction_cos_to_v": cos(f32_dot, f32_applied_sq),
            "f32_fraction_params_changed": changed / P,
            "f64_applied_over_requested_norm": math.sqrt(f64_applied_sq / req_sq),
            "f64_direction_cos_to_v": cos(f64_dot, f64_applied_sq),
            "f64_fraction_params_changed": changed64 / P,
        })
    return rows


def forward_collect(model, X, y, cfg, device, dtype, collector):
    """Run forward in chunks; snapshot or count via collector modes."""
    outs, flips = [], []
    for i in range(0, X.size(0), BATCH):
        xb = X[i:i + BATCH].to(device).to(dtype)
        yb = y[i:i + BATCH].to(device)
        b = xb.size(0)
        with torch.no_grad():
            if collector.mode == "snapshot":
                lg, _, _ = model.forward_path(xb, yb, None, augment=False,
                                              cfg=cfg)
            elif collector.mode == "count":
                # chunk-aligned hit offset
                chunk_idx = i // BATCH
                total = (X.size(0) + BATCH - 1) // BATCH
                M = len(collector.baseline) // total
                collector.flips = torch.zeros(b, dtype=torch.long,
                                              device=device)
                collector.hit = chunk_idx * M
                lg, _, _ = model.forward_path(xb, yb, None, augment=False,
                                              cfg=cfg)
                flips.append(collector.flips.clone())
            else:
                lg, _, _ = model.forward_path(xb, yb, None, augment=False,
                                              cfg=cfg)
        outs.append(lg.detach().to(torch.float64).cpu())
    return torch.cat(outs, 0), (torch.cat(flips, 0) if flips else None)


def network_flips(ckpt, X, y, device, dtype, win_ep):
    """Part B: flips for a model in the given dtype (float32 or float64)."""
    model, cfg = build_model(ckpt, device)
    model.to(dtype).eval()
    names = [n for n, _ in model.named_parameters()]
    base = {n: t.to(device).to(dtype) for n, t in ckpt["theta_bar"].items()}
    collector = FlipCollector(model)

    def setp(unit, eps):
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(base[n] if unit is None else
                        base[n] + eps * unit[n])

    setp(None, 0.0)
    collector.mode = "snapshot"
    collector.baseline, collector.hit = [], 0
    forward_collect(model, X, y, cfg, device, dtype, collector)

    rows = {}
    for e in win_ep:
        d = delta_from_sd(ckpt["theta_bar"], ckpt["traj"][e], names, device)
        dt = norm2(d) ** 0.5
        unit = {n: (d[n] / dt).to(dtype) for n in names}
        for a in ALPHAS:
            eps = a * dt
            setp(unit, eps)
            collector.mode = "count"
            _, fp = forward_collect(model, X, y, cfg, device, dtype, collector)
            setp(unit, -eps)
            collector.mode = "count"
            _, fm = forward_collect(model, X, y, cfg, device, dtype, collector)
            flip = (fp > 0) | (fm > 0)
            units = (fp + fm).float()
            rows.setdefault(a, {"flip_n": 0, "unit_sum": 0.0, "N": 0})
            rows[a]["flip_n"] += int(flip.sum())
            rows[a]["unit_sum"] += units.sum().item()
            rows[a]["N"] += X.size(0)
    T = len(win_ep)
    out = {str(a): {"sample_flip_fraction": r["flip_n"] / r["N"],
                    "mean_flip_units_per_sample_per_dir": r["unit_sum"] /
                    r["N"]} for a, r in rows.items()}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.data_dir is None:
        args.data_dir = DATASETS[ckpt["dataset"]]["default_dir"]
    X, y = get_batch(ckpt, args.data_dir, "holdout", args.n_samples,
                     args.workers)
    win_ep = sorted(int(e) for e in ckpt["window_epochs"])
    names = [n for n in ckpt["theta_bar"].keys()]
    e0 = win_ep[0]
    d = delta_from_sd(ckpt["theta_bar"], ckpt["traj"][e0], names, device)
    dt = norm2(d) ** 0.5
    unit = {n: (d[n] / dt) for n in names}
    # Part A is a pure parameter-space audit: do it on CPU in float32/float64.
    unit_cpu = {n: unit[n].float().cpu() for n in names}

    print("[quant] Part A: parameter-space quantization audit", flush=True)
    partA = quant_audit(ckpt["theta_bar"], unit_cpu, dt, ALPHAS)
    for r in partA:
        print(f"  a={r['alpha']:.0e} req/coord={r['requested_per_coord_rms']:.1e}"
              f"  f32: norm={r['f32_applied_over_requested_norm']:.2f}x "
              f"cos={r['f32_direction_cos_to_v']:.3f} "
              f"changed={r['f32_fraction_params_changed']:.2e} | "
              f"f64: norm={r['f64_applied_over_requested_norm']:.4f}x "
              f"cos={r['f64_direction_cos_to_v']:.4f}")

    print("[quant] Part B: float32 network flips", flush=True)
    f32 = network_flips(ckpt, X, y, device, torch.float32, win_ep)
    print("[quant] Part B: float64 network flips", flush=True)
    f64 = network_flips(ckpt, X, y, device, torch.float64, win_ep)
    print("  alpha | f32 flip% | f32 units | f64 flip% | f64 units")
    for a in ALPHAS:
        print(f"  {a:.0e} | {f32[str(a)]['sample_flip_fraction']:.4f} | "
              f"{f32[str(a)]['mean_flip_units_per_sample_per_dir']:.2f} | "
              f"{f64[str(a)]['sample_flip_fraction']:.4f} | "
              f"{f64[str(a)]['mean_flip_units_per_sample_per_dir']:.3f}")

    res = {"dataset": ckpt["dataset"], "target_domain": ckpt["target_domain"],
           "seed": ckpt["seed"], "mean_dt": dt,
           "param_audit": partA, "f32_flips": f32, "f64_flips": f64}
    txt = json.dumps(res, indent=2)
    if args.out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w").write(txt)
        print("saved", args.out)


if __name__ == "__main__":
    main()
