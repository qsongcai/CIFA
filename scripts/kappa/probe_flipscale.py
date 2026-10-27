"""Probe: at what relative radius does the ReLU flip rate collapse?

Real SWAD windows cross kinks at every alpha we can resolve for the HVP. This
probe scans very fine radii (flip counts only, no HVP -> no finite-difference
noise floor issue) to locate the linear-piece scale, and reports the number of
flipping units per sample (to distinguish harmless single-unit flips from
dense kink crossings). Reuses the checkpoint + FlipCollector from kappa_analyze.
"""
import argparse
import sys

import numpy as np
import torch

from kappa_analyze import (build_model, param_names, delta_from_sd, norm2,
                           apply_params, FlipCollector, get_batch, _forward)

FINE = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2,
        3e-2, 0.1, 0.3, 1.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--n_samples", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    X, y = get_batch(ckpt, args.data_dir, "holdout", args.n_samples, 2)
    model, cfg = build_model(ckpt, device)
    names = param_names(model)
    B = X.size(0)
    X, y = X.to(device), y.to(device)

    base = {n: ckpt["theta_bar"][n].to(device) for n in names}
    traj = ckpt["traj"]
    win_ep = sorted(e for e in ckpt["window_epochs"] if e in traj)
    if len(win_ep) < 2:
        win_ep = sorted(traj.keys())

    units = []
    for e in win_ep:
        d = delta_from_sd(ckpt["theta_bar"], traj[e], names, device)
        d2 = norm2(d)
        dt = d2 ** 0.5
        units.append((e, {n: (d[n] / dt) for n in names}, dt))

    collector = FlipCollector(model)
    apply_params(model, base, None, 0.0)
    H0, _ = _forward(model, X, y, cfg, collector, "snapshot", B, device)

    print(f"ckpt={args.ckpt}  domain={ckpt['target_domain']} seed={ckpt['seed']}")
    print(f"window={win_ep}  ||theta_t-theta_bar|| (r per point)="
          f"{[round(u[2],4) for u in units]}")
    print(f"{'alpha':>10} {'eps_global':>11} {'frac_any_flip':>14} "
          f"{'mean_units':>11} {'median_units':>12} {'p90_units':>10}")
    for a in FINE:
        any_flip = torch.zeros(B, dtype=torch.bool, device=device)
        tot_units = torch.zeros(B, dtype=torch.float, device=device)
        for e, unit, dt in units:
            eps = a * dt
            apply_params(model, base, unit, eps)
            _, fp = _forward(model, X, y, cfg, collector, "count", B, device)
            apply_params(model, base, unit, -eps)
            _, fm = _forward(model, X, y, cfg, collector, "count", B, device)
            any_flip |= (fp > 0) | (fm > 0)
            tot_units += (fp + fm).float()
        tot_units /= len(units)
        frac = any_flip.float().mean().item()
        u = tot_units.detach().cpu().numpy()
        print(f"{a:>10.0e} {a*units[0][2]:>11.2e} {frac:>14.4f} "
              f"{u.mean():>11.2f} {np.median(u):>12.2f} "
              f"{np.quantile(u,0.9):>10.2f}")
    collector.remove()


if __name__ == "__main__":
    main()
