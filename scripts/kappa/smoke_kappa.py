"""CPU smoke test for the R1-6 curvature/kink analysis.

Uses a tiny, controlled ReLU MLP (with the same forward_path interface as
CIFAResNet) and symmetric window points theta_t = theta_bar +/- scale*u so that
theta_bar is the *exact* centroid.  A ReLU network is affine in theta within a
fixed activation pattern, therefore:

  * small scale (no activation flips) -> ensemble gap ~ 0, interior central-
    difference Hessian ~ 0 (the smooth curvature vanishes);
  * large scale (some ReLU units cross their boundary) -> those samples are
    flagged by the activation-mask flip detector, and their gap is non-zero
    (epsilon_kink), while samples still in one linear piece keep gap ~ 0.

This validates the exact logic used in kappa_analyze.py without a GPU or data.
"""
import copy
import math

import torch
import torch.nn as nn

from kappa_analyze import analyze


class TinyMLP(nn.Module):
    def __init__(self, d=6, h=16, k=4):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(h, k)

    def forward_path(self, x, y=None, tracker=None, augment=False, cfg=None):
        z = self.relu(self.fc1(x))
        return self.fc2(z), z, {}


def unit_dirs(base_sd, names, gen):
    out = []
    for _ in range(2):
        d = {n: torch.randn(base_sd[n].shape, generator=gen) for n in names}
        norm = math.sqrt(sum((v ** 2).sum().item() for v in d.values()))
        out.append({n: v / norm for n, v in d.items()})
    return out


def make_ckpt(base_sd, a, b, scale, names):
    def shifted(sign, u):
        sd = {n: base_sd[n].clone() for n in base_sd}
        for n in names:
            sd[n] = sd[n] + sign * scale * u[n]
        return sd
    traj = {1: shifted(+1, a), 2: shifted(-1, a),
            3: shifted(+1, b), 4: shifted(-1, b)}
    return {
        "dataset": "pacs", "target_domain": "smoke", "seed": 0,
        "variant": "full", "hparams": {"cifa_moments_eps": 1e-6},
        "num_classes": 4,
        "valley_start": 1, "valley_end": 4,
        "window_epochs": [1, 2, 3, 4],
        "val_losses": [1.0, 0.9, 0.9, 0.9],
        "theta_bar": {n: t.clone() for n, t in base_sd.items()},
        "traj": traj,
    }


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(123)
    device = torch.device("cpu")
    model = TinyMLP()
    model.eval()
    X = torch.randn(64, 6, generator=torch.Generator().manual_seed(7))
    y = torch.zeros(64, dtype=torch.long)

    names = [n for n, _ in model.named_parameters()]
    base_sd = copy.deepcopy(model.state_dict())
    a, b = unit_dirs(base_sd, names, gen)

    print("scale | kinkfrac(alpha=1) | rms_gap | smooth(adaptive) | "
          "gap_linear | gap_kink | frac_linear | sweep slope")
    rows = []
    found_kink = None
    for scale in [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.7, 1.5, 3.0]:
        model.load_state_dict(base_sd)
        ckpt = make_ckpt(base_sd, a, b, scale, names)
        r = analyze(ckpt, X, y, device, model=model,
                    cfg={"cifa_moments_eps": 1e-6})
        sm = r["rms_smooth_curvature_gap_adaptive"]
        rows.append((scale, r))
        print(f"{scale:6.3g} | {r['kink_sample_fraction_at_alpha1']:17.3f} | "
              f"{r['rms_gap_h_ens_minus_h_bar']:8.2e} | {sm:15.2e} | "
              f"{r['rms_gap_linear_samples']:9.2e} | "
              f"{r['rms_gap_kink_samples']:8.2e} | "
              f"{r['fraction_sample_direction_with_linear_piece']:11.2f} | "
              f"{r['loglog_slope_gap_vs_r'] if r['loglog_slope_gap_vs_r'] is not None else float('nan'):6.2f}")
        if found_kink is None and r["kink_sample_fraction_at_alpha1"] > 0:
            found_kink = r

    # ---- assertions ----
    small = rows[0][1]
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(("PASS" if cond else "FAIL"), "-", msg)
        ok = ok and cond

    check(small["kink_sample_fraction_at_alpha1"] == 0.0,
          "small-scale: no ReLU flips (all samples stay in one linear piece)")
    check(small["rms_gap_h_ens_minus_h_bar"] < 1e-5,
          "small-scale: ensemble gap ~0 (affine-in-theta + exact centroid)")
    check(small["rms_smooth_curvature_gap_adaptive"] < 1e-4,
          "small-scale: adaptive interior Hessian ~0 (smooth kappa=0)")
    check(found_kink is not None,
          "large-scale: some samples cross a ReLU boundary (kink detected)")
    if found_kink is not None:
        check(found_kink["rms_gap_kink_samples"]
              > found_kink["rms_gap_linear_samples"],
              "large-scale: kink samples carry the gap, linear-piece samples ~0")
        check(found_kink["rms_gap_h_ens_minus_h_bar"] > 10 *
              small["rms_gap_h_ens_minus_h_bar"],
          "large-scale gap is orders of magnitude above the linear baseline")

    print("\nSMOKE", "PASSED" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
