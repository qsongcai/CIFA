"""R1-6 empirical curvature (kappa*r^2) and kink analysis for the SWAD window.

Consumes a checkpoint written by `train.py --kappa_ckpt <path>.pt`, which stores
theta_bar (the SWAD-averaged weights) and the per-epoch theta_t captured around
the detected valley.

On a fixed batch of in-domain hold-out (or target) images it reports:

  1. Geometry          r^2 = (1/T) sum_t ||theta_t - theta_bar||^2
                       and the measured predictor gap ||h_ens - h_bar||.
  2. Effective curvature  kappa_eff = 2*rms(h_ens - h_bar)/r^2  (mixes kink).
  3. Smooth-vs-kink split.  A ReLU net is affine in theta within a fixed
     activation pattern, so the interior Hessian is exactly zero.  We snapshot
     the activation signs once at theta_bar, then at theta_bar +/- eps*v_t
     count, per sample, the number of ReLU units whose sign flips.  A sample
     with no flip on either side stays in the same linear piece and its
     central-difference second directional derivative must be ~0 (smooth
     curvature).  Samples that flip cross a kink (residual epsilon_kink).  The
     interior Hessian is estimated at the largest epsilon that avoids a flip
     (most accurate finite difference inside one linear piece).
  4. Radius sweep.  Nested contiguous windows around the true SWAD window give
     r from small to large; the smooth interior term should stay ~0 while the
     net gap is driven by kink crossings.

Memory note: only the baseline activation signs are kept (one snapshot). Each
perturbed forward accumulates a per-sample flip count inside the forward hook
and never materializes the full activation mask, so memory is bounded by one
forward pass plus the baseline.

All numbers are empirical and use the *clean* forward path (augment=False).
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from datasets import DATASETS, get_dataset
from networks import CIFAResNet

# Relative radii eps = alpha*||theta_t-theta_bar|| at which flips are counted.
# Fine values (1e-9..) locate the linear-cell diameter of the piecewise-linear
# ReLU network; the HVP subset is the radii where a finite difference is above
# the float noise floor and we also estimate the interior quadratic term.
FLIP_ALPHAS = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
               1e-2, 0.1, 0.5, 1.0]
HVP_ALPHAS = [1e-3, 1e-2, 0.1, 0.5, 1.0]
# KNOWN LIMITATION (2026-10-25, float64): every HVP radius is >= 1e-3, at which
# 100% of samples have already crossed a kink (float64 cell radius ~1.47e-6).
# Hence the float32 "rms_interior_quadratic_on_linear_piece = 0 (12/12)" result
# is "no in-cell sample measurable at these radii", NOT a measured zero. The
# in-cell term is resolved in float64 by float64_hvp.py (non-zero ~1.2e-13).


def build_model(ckpt, device):
    cfg = ckpt["hparams"]
    model = CIFAResNet(
        backbone=cfg.get("backbone", "resnet50"),
        pretrained=False,
        num_classes=ckpt["num_classes"],
        layers=tuple(cfg.get("cifa_layers", ("layer1", "layer2"))),
        freeze_bn=cfg.get("freeze_bn", True)).to(device)
    model.load_state_dict(ckpt["theta_bar"])
    model.eval()
    return model, cfg


def param_names(model):
    return [name for name, _ in model.named_parameters()]


def delta_from_sd(base_sd, sd, names, device):
    """Per-parameter theta_t - theta_bar (learnable parameters only)."""
    return {n: (sd[n].to(device) - base_sd[n].to(device)) for n in names}


def norm2(delta):
    return sum((d ** 2).sum().item() for d in delta.values())


def apply_params(model, base, unit, eps):
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(base[name] if unit is None else base[name] + eps * unit[name])


class FlipCollector:
    """Snapshot baseline ReLU signs once; later count per-sample flips.

    A single nn.ReLU module is invoked several times per bottleneck, so we
    index by forward-hit order (the order is identical on every forward for a
    fixed model, input shape, and augment=False).
    """

    def __init__(self, model):
        self.relus = [m for m in model.modules() if isinstance(m, nn.ReLU)]
        self.handles = [m.register_forward_hook(self._hook) for m in self.relus]
        self.mode = "idle"
        self.baseline = []
        self.flips = None
        self.hit = 0

    def _hook(self, module, inp, out):
        pos = (out > 0)
        if self.mode == "snapshot":
            self.baseline.append(pos.detach())
        elif self.mode == "count":
            base = self.baseline[self.hit]
            self.flips += (pos.detach() != base).reshape(pos.size(0), -1).sum(1)
            self.hit += 1
        # idle: nothing stored

    def begin_snapshot(self):
        self.baseline = []
        self.hit = 0
        self.mode = "snapshot"

    def begin_count(self, batch, device):
        self.flips = torch.zeros(batch, dtype=torch.long, device=device)
        self.hit = 0
        self.mode = "count"

    def idle(self):
        self.mode = "idle"

    def remove(self):
        for h in self.handles:
            h.remove()


def _forward(model, X, y, cfg, collector, mode, B, device):
    """Clean forward; mode is snapshot|count|idle. Returns (logits, flips)."""
    with torch.no_grad():
        if mode == "snapshot":
            collector.begin_snapshot()
        elif mode == "count":
            collector.begin_count(B, device)
        else:
            collector.idle()
        logits, _, _ = model.forward_path(X, y, None, augment=False, cfg=cfg)
        flips = collector.flips.clone() if mode == "count" else None
    return logits.detach(), flips


def get_batch(ckpt, data_dir, split, n_samples, workers):
    data = get_dataset(
        ckpt["dataset"], data_dir, ckpt["target_domain"],
        holdout_fraction=ckpt["hparams"].get("holdout_fraction", 0.2),
        seed=ckpt["seed"])
    from torch.utils.data import ConcatDataset, DataLoader
    if split == "target":
        ds = data["target"]
    else:
        ds = ConcatDataset(list(data["holdout_by_domain"].values()))
    loader = DataLoader(ds, batch_size=n_samples, shuffle=False,
                        num_workers=workers)
    X, y = next(iter(loader))
    return X, y


def analyze(ckpt, X, y, device, model=None, cfg=None, max_points=16):
    if model is None:
        model, cfg = build_model(ckpt, device)
    else:
        cfg = cfg or {}
    names = param_names(model)
    B = X.size(0)
    X, y = X.to(device), y.to(device)

    base = {n: ckpt["theta_bar"][n].to(device) for n in names}
    traj = ckpt["traj"]
    win_ep = [e for e in ckpt["window_epochs"] if e in traj]
    if len(win_ep) < 2:
        win_ep = sorted(traj.keys())
    win_ep = sorted(win_ep)

    collector = FlipCollector(model)

    # ---- reference at theta_bar: snapshot baseline signs ----
    apply_params(model, base, None, 0.0)
    H0, _ = _forward(model, X, y, cfg, collector, "snapshot", B, device)
    K = H0.size(1)

    # ---- per-window-point: real logits, geometry, finite-difference HVP ----
    Ht = {}
    per_point = []
    for e in win_ep:
        d = delta_from_sd(ckpt["theta_bar"], traj[e], names, device)
        d2 = norm2(d)
        dt = d2 ** 0.5
        unit = {n: (d[n] / dt) for n in names}

        # real theta_t logits (no masks needed)
        apply_params(model, base, unit, dt)
        Ht[e], _ = _forward(model, X, y, cfg, collector, "idle", B, device)

        # two perturbed forwards per radius give BOTH logits and flip counts
        cross_a, units_a, quad_a = {}, {}, {}
        for a in FLIP_ALPHAS:
            eps = a * dt
            apply_params(model, base, unit, eps)
            Hp, fp = _forward(model, X, y, cfg, collector, "count", B, device)
            apply_params(model, base, unit, -eps)
            Hm, fm = _forward(model, X, y, cfg, collector, "count", B, device)
            cross_a[a] = (fp > 0) | (fm > 0)
            units_a[a] = (fp + fm).float()             # flipping units / sample
            if a in HVP_ALPHAS:
                D2 = (Hp - 2 * H0 + Hm) / (eps ** 2)   # v^T H v [B,K]
                quad_a[a] = (0.5 * d2 * D2).detach()   # [B,K]
        per_point.append({"cross": cross_a, "units": units_a, "quad": quad_a})

    # ---- ensemble gap and effective curvature at the true SWAD window ----
    T = len(win_ep)
    Hens = torch.stack([Ht[e] for e in win_ep], 0).mean(0)
    gap = Hens - H0
    r2 = sum(norm2(delta_from_sd(ckpt["theta_bar"], traj[e], names, device))
             for e in win_ep) / T
    rms_gap = torch.sqrt((gap ** 2).mean()).item()
    rms_logits = torch.sqrt((H0 ** 2).mean()).item()
    kappa_eff = 2 * rms_gap / r2 if r2 > 0 else float("nan")

    cross_at = {a: torch.zeros(B, dtype=torch.bool, device=device)
                for a in FLIP_ALPHAS}
    units_sum = {a: torch.zeros(B, device=device) for a in FLIP_ALPHAS}
    for pp in per_point:
        for a in FLIP_ALPHAS:
            cross_at[a] |= pp["cross"][a]
            units_sum[a] += pp["units"][a]
    units_mean = {a: units_sum[a] / T for a in FLIP_ALPHAS}

    # kink membership at the true step (alpha=1)
    kink_samples = cross_at[1.0]
    linear_samples = ~kink_samples
    rms_gap_linear = torch.sqrt((gap[linear_samples] ** 2).mean()).item() \
        if linear_samples.any() else 0.0
    rms_gap_kink = torch.sqrt((gap[kink_samples] ** 2).mean()).item() \
        if kink_samples.any() else 0.0

    # radius scan: fraction crossing a kink, number of flipping units, and
    # (only on HVP radii above the noise floor) the interior quadratic term
    alpha_tbl = {}
    for a in FLIP_ALPHAS:
        keep = ~cross_at[a]
        entry = {
            "cross_kink_fraction": cross_at[a].float().mean().item(),
            "mean_flip_units": units_mean[a].mean().item(),
            "median_flip_units": units_mean[a].median().item(),
        }
        if a in HVP_ALPHAS:
            vals = torch.stack([pp["quad"][a] for pp in per_point], 0).mean(0)
            entry["rms_interior_quadratic_on_linear_piece"] = (
                torch.sqrt((vals[keep] ** 2).mean()).item()
                if keep.any() else 0.0)
        else:
            entry["rms_interior_quadratic_on_linear_piece"] = None
        alpha_tbl[str(a)] = entry

    # Activation-cell diameter (FLOAT32, KNOWN TO BE UNRELIABLE below ~1e-3).
    # CORRECTION (2026-10-25, float64 reinforcement): inside a fixed activation
    # pattern a ReLU net is MULTI-AFFINE, not affine, in the parameters
    # (f = W2 D W1 x; a joint direction has a non-zero eps^2 cross-layer term
    # V2 D V1 x, so the in-cell Hessian is a non-zero constant; only a
    # single-layer direction has zero Hessian). Moreover this float32 scan
    # cannot resolve the cell: at alpha<=1e-3 the per-coordinate move
    # eps/sqrt(P) is far below the float32 ULP, so the reported "flip" is
    # quantization noise (quant_check: alpha=1e-7 unit counts inflated ~1e3x).
    # Float64 gives the true cell radius ~1.47e-6 and an in-cell quadratic
    # term ~1.2e-13 (relative to logits), i.e. non-zero but negligible vs the
    # window gap (~0.08). See float64_hvp.py / results_f64 / results_quant.
    # The SWAD window r vs the float64 cell radius (~1e6 ratio) still shows
    # the averaging window lies deep in the kink regime rather than a cell.
    linear_alpha = None
    for a in FLIP_ALPHAS:
        if cross_at[a].float().mean().item() < 1e-9:
            linear_alpha = a
    linear_eps_global = linear_alpha * (r2 ** 0.5) if linear_alpha is not None \
        else None
    window_to_linear_ratio = ((r2 ** 0.5) / linear_eps_global) \
        if linear_eps_global else None

    # adaptive smooth curvature: largest epsilon that stays in one linear piece
    smooth_gap = torch.zeros(B, K, device=device)
    have_linear, total_bt = 0, 0
    for pp in per_point:
        for b in range(B):
            amax = None
            for a in reversed(HVP_ALPHAS):
                if not pp["cross"][a][b]:
                    amax = a
                    break
            if amax is not None:
                smooth_gap[b] += pp["quad"][amax][b]
                have_linear += 1
            total_bt += 1
    smooth_gap /= T
    rms_smooth_gap_adaptive = torch.sqrt((smooth_gap ** 2).mean()).item()
    frac_linear_segments = have_linear / max(1, total_bt)

    # ---- radius sweep: nested contiguous windows (logits only, idle) ----
    all_ep = sorted(traj.keys())
    center = sum(win_ep) / len(win_ep)
    sweep = []
    max_r = max(abs(e - center) for e in all_ep)
    for k in range(0, int(max_r) + 1):
        sub = [e for e in all_ep if abs(e - center) <= k]
        if len(sub) < 2:
            continue
        wbar = {n: torch.stack([traj[e][n].to(device) for e in sub], 0).mean(0)
                for n in names}
        r2w = 0.0
        for e in sub:
            dd = {n: (traj[e][n].to(device) - wbar[n]) for n in names}
            r2w += norm2(dd)
        r2w /= len(sub)
        apply_params(model, wbar, None, 0.0)
        Hw, _ = _forward(model, X, y, cfg, collector, "idle", B, device)
        Hlist = []
        for e in sub:
            if e in Ht:
                Hlist.append(Ht[e])
            else:
                apply_params(model, wbar, None, 0.0)
                with torch.no_grad():
                    for nm, p in model.named_parameters():
                        p.copy_(traj[e][nm].to(device))
                lg, _ = _forward(model, X, y, cfg, collector, "idle", B, device)
                Hlist.append(lg)
        g = torch.stack(Hlist, 0).mean(0) - Hw
        rms_g = torch.sqrt((g ** 2).mean()).item()
        sweep.append({
            "n_points": len(sub),
            "epochs": sub,
            "r2": r2w,
            "r": r2w ** 0.5,
            "rms_gap": rms_g,
            "gap_over_logits": rms_g / rms_logits if rms_logits > 0 else None,
        })

    apply_params(model, base, None, 0.0)
    collector.idle()
    collector.remove()

    slope = None
    valid = [s for s in sweep if s["rms_gap"] > 1e-12 and s["r"] > 0]
    if len(valid) >= 3:
        lr = [math.log(s["r"]) for s in valid]
        lg = [math.log(s["rms_gap"]) for s in valid]
        mr, mg = sum(lr) / len(lr), sum(lg) / len(lg)
        num = sum((a - mr) * (b - mg) for a, b in zip(lr, lg))
        den = sum((a - mr) ** 2 for a in lr)
        slope = num / den if den > 0 else None

    return {
        "dataset": ckpt["dataset"],
        "target_domain": ckpt["target_domain"],
        "seed": ckpt["seed"],
        "n_batch": B,
        "n_valley_points": T,
        "window_epochs": win_ep,
        "valley_start": ckpt["valley_start"],
        "valley_end": ckpt["valley_end"],
        "n_captured_epochs": len(traj),
        "r2": r2,
        "r": r2 ** 0.5,
        "rms_gap_h_ens_minus_h_bar": rms_gap,
        "rms_logits": rms_logits,
        "gap_over_logits": rms_gap / rms_logits if rms_logits > 0 else None,
        "kappa_eff": kappa_eff,
        "kink_sample_fraction_at_alpha1": kink_samples.float().mean().item(),
        "fraction_sample_direction_with_linear_piece": frac_linear_segments,
        "rms_gap_linear_samples": rms_gap_linear,
        "rms_gap_kink_samples": rms_gap_kink,
        "rms_smooth_curvature_gap_adaptive": rms_smooth_gap_adaptive,
        "linear_cell_alpha": linear_alpha,
        "linear_cell_global_radius": linear_eps_global,
        "swad_window_to_linear_cell_ratio": window_to_linear_ratio,
        "alpha_scan": alpha_tbl,
        "radius_sweep": sweep,
        "loglog_slope_gap_vs_r": slope,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--split", default="holdout", choices=["holdout", "target"])
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.data_dir is None:
        args.data_dir = DATASETS[ckpt["dataset"]]["default_dir"]
    X, y = get_batch(ckpt, args.data_dir, args.split, args.n_samples,
                     args.workers)
    res = analyze(ckpt, X, y, device)
    txt = json.dumps(res, indent=2)
    print(txt)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(txt)
        print("saved", args.out)


if __name__ == "__main__":
    main()
