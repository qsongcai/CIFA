"""R1-6 float64 reinforcement: DIRECT numerical test of the in-cell geometry.

The float32 analysis (kappa_analyze.py) located the activation-pattern cell
diameter at global eps ~ 2e-7, but there eps^2 ~ 4e-14 is below the float32
round-off floor (~1e-7), so the interior could only be argued analytically.
In float64 (eps_mach ~ 2.2e-16) the second central difference at eps~1e-8
has a floor of only ~eps_mach*|H|, which lets us read the in-cell terms
directly.

CORRECT geometry (the earlier "Hessian is strictly zero inside a linear
piece" claim was WRONG; smoke_float64.py proves the corrected form).  For a
fixed input x, a ReLU net inside a fixed activation pattern D is
MULTI-AFFINE in the parameters, not affine:

  f(x; W1, W2) = W2 D W1 x,
  f(theta+eps*v) = f(theta)
      + eps [V2 D W1 x + W2 D V1 x] + eps^2 [V2 D V1 x].
Along a single-layer direction the second difference is exactly 0; along a
JOINT direction (the ones SWAD moves along) it is eps^2 times a constant
in-cell Hessian (and an L-layer net has terms up to eps^L).  We report
    N(eps) = f(theta+eps*v) - 2 f(theta) + f(theta-eps*v)
normalized by RMS logit magnitude, with three scaling fingerprints:

  (1) ReLU, samples that do NOT flip (inside one activation-pattern cell):
          N = c2*eps^2 (multi-affine interior term), which at eps ~ 1e-8 is
          indistinguishable from the float64 floor (~1e-16); we report a
          rigorous UPPER BOUND on c2 and extrapolate it to the cell radius.
  (2) ReLU, samples that DO cross a kink:
          N ~ ||grad jump|| * eps  -> LINEAR growth (slope 1). Positive
          control from the same net, batch, and direction.
  (3) GELU network with identical weights (C^2 everywhere, no kinks):
          N = eps^2 * v^T H v     -> QUADRATIC growth (slope 2). Calibrates
          that the float64 pipeline does resolve genuine curvature.

The conclusion is the separation of scales: the in-cell multi-affine term
at the cell radius (~2e-7) is bounded at ~1e-14 of the logits, while the
window-scale (eps ~ 1.7) kink bias is ~0.07 — twelve orders of magnitude
larger, and bounded/saturating (radius-scan slope 0.38).

Usage:
  python float64_hvp.py --ckpt ckpt/kappa_art_painting_seed0.pt \
      --out analysis/f64_kappa_art_painting_seed0.json
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from datasets import DATASETS
from kappa_analyze import (build_model, get_batch,
                          delta_from_sd, norm2, FlipCollector)

# Relative radii eps = alpha * ||theta_t - theta_bar||.
# ReLU scans also include alpha=1.0 (the actual SWAD window step = the
# symmetric ensemble Jensen gap N/2, comparable to the float32 gap/logits).
ALPHAS = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1.0]
GELU_ALPHAS = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]   # small-scale scaling only
BATCH = 64
GELU_DIRS = 2          # GELU calibration: only first/last window direction
EPS_MACH = torch.finfo(torch.float64).eps   # 2.220446049250313e-16


def replace_relu_with_gelu(root):
    """Recursively replace every nn.ReLU (also inside Sequential) with GELU."""
    for parent in root.modules():
        for cname, child in list(parent.named_children()):
            if isinstance(child, nn.ReLU):
                setattr(parent, cname, nn.GELU())
            else:
                replace_relu_with_gelu(child)


def double_state(sd, device):
    return {n: t.to(device).double() for n, t in sd.items()}


def set_params(model, base, unit, eps):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if unit is None:
                p.copy_(base[name])
            else:
                p.copy_(base[name] + eps * unit[name])


def rms(t):
    return torch.sqrt((t ** 2).mean()).item()


def batched_forward(model, X, y, cfg, device, hook_mode, collector=None):
    """Forward in chunks of BATCH (float64 memory). Returns logits [N,K],
    and per-sample flips [N] when hook_mode='count'."""
    outs, flips = [], []
    N = X.size(0)
    for i in range(0, N, BATCH):
        xb = X[i:i + BATCH].to(device).double()
        yb = y[i:i + BATCH].to(device)
        b = xb.size(0)
        with torch.no_grad():
            if hook_mode == "snapshot":
                # baseline must ACCUMULATE across chunks: begin once, later
                # chunks keep appending (begin_snapshot() would wipe it).
                if i == 0:
                    collector.begin_snapshot()
                lg, _, _ = model.forward_path(xb, yb, None, augment=False,
                                              cfg=cfg)
            elif hook_mode == "count":
                # index into the concatenated baseline: each chunk's first
                # ReLU hit starts at chunk_index * M (begin_count resets hit
                # to 0 and must NOT be used for later chunks).
                chunk_idx = i // BATCH
                total_chunks = (N + BATCH - 1) // BATCH
                M = len(collector.baseline) // total_chunks
                collector.flips = torch.zeros(b, dtype=torch.long,
                                              device=device)
                collector.hit = chunk_idx * M
                collector.mode = "count"
                lg, _, _ = model.forward_path(xb, yb, None, augment=False,
                                              cfg=cfg)
                flips.append(collector.flips.clone())
            else:
                lg, _, _ = model.forward_path(xb, yb, None, augment=False,
                                              cfg=cfg)
        outs.append(lg.detach().double().cpu())
    if hook_mode == "count":
        return torch.cat(outs, 0), torch.cat(flips, 0)
    return torch.cat(outs, 0), None


def _new_acc():
    # per alpha: sum of squared normalized second differences + element
    # counts, split by flip group; plus sample counts for the flip fraction.
    return {a: {"noflip_sumsq": 0.0, "noflip_n": 0,
                "flip_sumsq": 0.0, "flip_n": 0,
                "all_sumsq": 0.0, "all_n": 0,
                "flip_samples": 0, "total_samples": 0} for a in ALPHAS}


def _accumulate(acc, a, Nmat, flip):
    n2 = (Nmat ** 2).sum(1)                       # per sample [N]
    K = Nmat.size(1)
    acc[a]["noflip_sumsq"] += n2[~flip].sum().item()
    acc[a]["noflip_n"] += int((~flip).sum()) * K
    acc[a]["flip_sumsq"] += n2[flip].sum().item()
    acc[a]["flip_n"] += int(flip.sum()) * K
    acc[a]["all_sumsq"] += n2.sum().item()
    acc[a]["all_n"] += Nmat.numel()
    acc[a]["flip_samples"] += int(flip.sum())
    acc[a]["total_samples"] += Nmat.size(0)


def _table(acc, h0_rms):
    out = {}
    for a in ALPHAS:
        e = acc[a]

        def rr(sumsq, n):
            return math.sqrt(sumsq / n) / h0_rms if n else None

        out[str(a)] = {
            "flip_fraction": (e["flip_samples"] / e["total_samples"]
                              if e["total_samples"] else None),
            "N_over_logits_noflip": rr(e["noflip_sumsq"], e["noflip_n"]),
            "N_over_logits_flip": rr(e["flip_sumsq"], e["flip_n"]),
            "N_over_logits_all": rr(e["all_sumsq"], e["all_n"]),
        }
    return out


def run_relu(ckpt, X, y, device, win_ep):
    """Float64 second differences on the real ReLU network, split by flip."""
    model, cfg = build_model(ckpt, device)
    model.double().eval()
    names = [n for n, _ in model.named_parameters()]
    base = double_state(ckpt["theta_bar"], device)
    collector = FlipCollector(model)

    dirs = []
    for e in win_ep:
        d = delta_from_sd(ckpt["theta_bar"], ckpt["traj"][e], names, device)
        dt = norm2(d) ** 0.5
        dirs.append((e, dt, {n: (d[n].double() / dt) for n in names}))

    set_params(model, base, None, 0.0)
    H0, _ = batched_forward(model, X, y, cfg, device, "snapshot", collector)
    h0_rms = rms(H0)

    acc = _new_acc()
    dts = []
    for e, dt, unit in dirs:
        dts.append(dt)
        for a in ALPHAS:
            eps = a * dt
            set_params(model, base, unit, eps)
            Hp, fp = batched_forward(model, X, y, cfg, device, "count",
                                     collector)
            set_params(model, base, unit, -eps)
            Hm, fm = batched_forward(model, X, y, cfg, device, "count",
                                     collector)
            Nmat = (Hp - 2 * H0 + Hm).double()
            flip = ((fp > 0) | (fm > 0)).cpu()
            _accumulate(acc, a, Nmat, flip)
    collector.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return _table(acc, h0_rms), sum(dts) / len(dts), h0_rms


def run_gelu(ckpt, X, y, device, win_ep):
    """Float64 calibration on the same weights with GELU (everywhere C^2)."""
    model, cfg = build_model(ckpt, device)
    replace_relu_with_gelu(model)
    model.double().eval()
    names = [n for n, _ in model.named_parameters()]
    base = double_state(ckpt["theta_bar"], device)

    use_ep = [win_ep[0], win_ep[-1]] if len(win_ep) >= 2 else list(win_ep)
    dirs = []
    for e in use_ep:
        d = delta_from_sd(ckpt["theta_bar"], ckpt["traj"][e], names, device)
        dt = norm2(d) ** 0.5
        dirs.append((dt, {n: (d[n].double() / dt) for n in names}))

    set_params(model, base, None, 0.0)
    H0, _ = batched_forward(model, X, y, cfg, device, "idle")
    h0_rms = rms(H0)
    acc = {a: {"sumsq": 0.0, "n": 0} for a in GELU_ALPHAS}
    dts = []
    for dt, unit in dirs:
        dts.append(dt)
        for a in GELU_ALPHAS:
            eps = a * dt
            set_params(model, base, unit, eps)
            Hp, _ = batched_forward(model, X, y, cfg, device, "idle")
            set_params(model, base, unit, -eps)
            Hm, _ = batched_forward(model, X, y, cfg, device, "idle")
            Nmat = (Hp - 2 * H0 + Hm).double()
            acc[a]["sumsq"] += (Nmat ** 2).sum().item()
            acc[a]["n"] += Nmat.numel()
    table = {str(a): {"N_over_logits_gelu":
                      math.sqrt(acc[a]["sumsq"] / acc[a]["n"]) / h0_rms
                      if acc[a]["n"] else None} for a in GELU_ALPHAS}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return table, sum(dts) / len(dts), h0_rms


def fit_slope(xs, ys):
    """log-log slope of y vs x (x = alpha, i.e. eps up to a constant dt)."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys)
           if y is not None and y > 0]
    if len(pts) < 2:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    num = sum((a - mx) * (b - my) for a, b in pts)
    den = sum((a - mx) ** 2 for a, _ in pts)
    return num / den if den > 0 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--split", default="holdout", choices=["holdout", "target"])
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--skip_gelu", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.data_dir is None:
        args.data_dir = DATASETS[ckpt["dataset"]]["default_dir"]
    X, y = get_batch(ckpt, args.data_dir, args.split, args.n_samples,
                     args.workers)

    win_ep = sorted(int(e) for e in ckpt["window_epochs"])
    assert len(win_ep) >= 2, ckpt["window_epochs"]

    print(f"[f64] device={device} dtype=float64 eps_mach={EPS_MACH:.3e} "
          f"window={win_ep}", flush=True)
    relu_tbl, dt_relu, _ = run_relu(ckpt, X, y, device, win_ep)
    print("[f64] ReLU done", flush=True)
    gelu_tbl = None
    if not args.skip_gelu:
        gelu_tbl, _, _ = run_gelu(ckpt, X, y, device, win_ep)
        print("[f64] GELU done", flush=True)

    # ---- scaling fingerprints ----
    # noflip points need a majority of samples still in the same cell
    # (alpha=1e-8 has 85-96% non-flipped, plenty to estimate the in-cell RMS);
    # flip points are alphas where a majority of samples cross a kink.
    noflip_pts, flip_pts = [], []
    for a in ALPHAS:
        e = relu_tbl[str(a)]
        ff = e["flip_fraction"]
        if ff is not None and ff <= 0.5 and e["N_over_logits_noflip"]:
            noflip_pts.append((a, e["N_over_logits_noflip"]))
        if ff is not None and ff >= 0.5 and e["N_over_logits_flip"]:
            flip_pts.append((a, e["N_over_logits_flip"]))
    slope_noflip = fit_slope([x for x, _ in noflip_pts],
                             [z for _, z in noflip_pts])
    # kink slope: only the small-alpha flipped points (1e-7..1e-5), before any
    # large-scale saturation; alpha=1 is the actual window, not a scaling fit.
    small_flip = [(a, z) for a, z in flip_pts if a <= 1e-5]
    slope_flip = fit_slope([x for x, _ in small_flip],
                           [z for _, z in small_flip])

    floors = [relu_tbl[str(a)]["N_over_logits_noflip"] for a in ALPHAS
              if relu_tbl[str(a)]["N_over_logits_noflip"]]
    floor_emp = min(floors) if floors else EPS_MACH
    slope_gelu = None
    if gelu_tbl:
        keep = [(a, gelu_tbl[str(a)]["N_over_logits_gelu"])
                for a in GELU_ALPHAS
                if gelu_tbl[str(a)]["N_over_logits_gelu"] is not None
                and gelu_tbl[str(a)]["N_over_logits_gelu"]
                > 3 * max(floor_emp, EPS_MACH)]
        slope_gelu = fit_slope([x for x, _ in keep],
                               [z for _, z in keep])

    # ---- in-cell multi-affine upper bound, extrapolated to the cell radius
    # tightest non-flipped radius: largest alpha with most samples in-cell
    in_cell = [(a, n) for a, n in noflip_pts
               if relu_tbl[str(a)]["flip_fraction"] is not None
               and relu_tbl[str(a)]["flip_fraction"] <= 0.5]
    c2_bounds = [(a, n / (a * dt_relu) ** 2) for a, n in in_cell]
    # smallest c2 bound = tightest (largest radius), but the floor contaminates
    # it; we report the alpha=1e-8 point as the conservative bound by default.
    a_cell = next((a for a in ALPHAS
                   if relu_tbl[str(a)]["flip_fraction"] is not None
                   and relu_tbl[str(a)]["flip_fraction"] >= 0.5), None)
    c2_bound = None
    cell_quad_bound = None
    cell_eps = None
    if in_cell and a_cell:
        a_tight, n_tight = max(in_cell, key=lambda t: t[0])
        c2_bound = n_tight / (a_tight * dt_relu) ** 2
        # cell radius upper bound: the first alpha where half the samples flip
        cell_eps = a_cell * dt_relu
        cell_quad_bound = c2_bound * cell_eps ** 2

    kink_1e7 = relu_tbl[str(1e-7)]["N_over_logits_flip"]
    # actual window (alpha=1): symmetric ensemble Jensen gap is N/2, directly
    # comparable to the float32 gap/logits (~0.073)
    n1 = relu_tbl[str(1.0)]["N_over_logits_all"]
    window_gap = n1 / 2.0 if n1 else None
    gelu_1e5 = gelu_tbl[str(1e-5)]["N_over_logits_gelu"] if gelu_tbl else None

    res = {
        "dataset": ckpt["dataset"],
        "target_domain": ckpt["target_domain"],
        "seed": ckpt["seed"],
        "dtype": "float64",
        "eps_mach": EPS_MACH,
        "n_samples": int(X.size(0)),
        "window_epochs": win_ep,
        "mean_dt": dt_relu,
        "relu": relu_tbl,
        "gelu": gelu_tbl,
        "scaling_slopes": {
            "relu_noflip_floor": slope_noflip,
            "relu_flip_kink_expected_1": slope_flip,
            "gelu_smooth_expected_2": slope_gelu,
        },
        "empirical_floor_N_over_logits": floor_emp,
        "in_cell_multi_affine": {
            "tightest_in_cell_alpha": (max(in_cell, key=lambda t: t[0])[0]
                                       if in_cell else None),
            "c2_upper_bound": c2_bound,
            "cell_boundary_alpha": a_cell,
            "cell_radius_eps_upper": cell_eps,
            "cell_quadratic_term_bound_N_over_logits": cell_quad_bound,
        },
        "kink_N_over_logits_at_1e-7": kink_1e7,
        "kink_to_cell_quad_ratio_at_1e-7": (
            kink_1e7 / cell_quad_bound
            if kink_1e7 and cell_quad_bound else None),
        "window_gap_N_over_2_at_alpha1": window_gap,
        "window_to_cell_quad_ratio": (
            window_gap / cell_quad_bound
            if window_gap and cell_quad_bound else None),
        "gelu_N_over_logits_at_1e-5": gelu_1e5,
        "gelu_to_floor_ratio_at_1e-5": (gelu_1e5 / floor_emp
                                        if gelu_1e5 and floor_emp else None),
    }
    txt = json.dumps(res, indent=2)
    print(txt)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(txt)
        print("saved", args.out)


if __name__ == "__main__":
    main()
