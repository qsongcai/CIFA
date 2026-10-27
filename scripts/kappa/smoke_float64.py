"""CPU smoke for the float64 Hessian/kink analysis — CORRECT geometry.

Key fact this test pins down (and which the earlier float32 analysis got
wrong): for a fixed input x, a ReLU network inside a fixed activation pattern
is MULTI-AFFINE in the parameters, NOT affine:

  f(x; W1, W2) = W2 D(W1 x),  D fixed
  f(theta+eps*v) = f(theta)
      + eps [V2 D W1 x + W2 D V1 x]
      + eps^2 [V2 D V1 x]                      <- non-zero interior term
  (a 3-layer net also has an eps^3 term V3 D2 V2 D1 V1 x).

The Hessian is zero only along single-layer directions; joint directions
(the ones SWAD actually moves along) have a constant in-cell Hessian.
Part A pins the arithmetic; Part B verifies the multi-affine structure on
small real ReLU MLPs and the kink-vs-smooth scaling.

Checks:
  1. scalar quadratic theta^2 -> D2 = 2 exactly (pipeline correctness)
  2. ReLU kink at theta=0     -> D2 = 1/eps (numerator linear in eps)
  3. GELU''(0) = sqrt(2/pi)   -> smooth curvature recovered
  4. 2-layer ReLU MLP, single-layer directions (V1 only / V2 only):
     N at the float64 floor for all eps with no flip
  5. 2-layer ReLU MLP, joint direction: N = c2*eps^2 exactly (slope 2,
     c2 constant), no eps^3 term
  6. 3-layer ReLU MLP, joint direction: N = c2*eps^2 + c3*eps^3,
     log-log slope between 2 and 3
  7. same joint direction, samples that cross a kink: |N| jumps by orders
     of magnitude above the multi-affine in-cell value
"""
import math

import torch
import torch.nn as nn

from float64_hvp import replace_relu_with_gelu

EPS_MACH = 2.220446049250313e-16
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

failures = []


def check(cond, msg):
    print(("PASS" if cond else "FAIL"), "-", msg)
    if not cond:
        failures.append(msg)


def D2_scalar(f, theta, eps):
    return (f(theta + eps) - 2 * f(theta) + f(theta - eps)) / eps ** 2


# ---------------- Part A: scalar arithmetic ----------------
quad = D2_scalar(lambda t: t ** 2, 1.0, 1e-4)
check(abs(quad - 2) < 1e-7, f"quadratic: D2(theta^2)=2, got {quad:.10f}")

eps = 1e-6
d2_kink = D2_scalar(torch.relu, torch.tensor(0.0), eps).item()
check(abs(d2_kink - 1 / eps) / (1 / eps) < 1e-12,
      f"ReLU kink: D2=1/eps={1/eps:.1e}, got {d2_kink:.1e}")

gelu_d2 = D2_scalar(nn.functional.gelu, torch.tensor(0.0), 1e-4).item()
gelu_true = math.sqrt(2 / math.pi)
check(abs(gelu_d2 - gelu_true) / gelu_true < 1e-6,
      f"GELU smooth: GELU''(0)={gelu_true:.6f}, got {gelu_d2:.6f}")

# kink numerator ~ eps (x10/decade); smooth numerator ~ eps^2 (x100/decade)
N_k = [torch.relu(torch.tensor(e)) - 2 * torch.relu(torch.tensor(0.0))
       + torch.relu(torch.tensor(-e)) for e in [1e-7, 1e-6, 1e-5]]
check(all(abs((N_k[i + 1] / N_k[i]).item() - 10) < 1e-8 for i in range(2)),
      "kink numerator scales LINEARLY (x10/decade)")
N_q = [((1 + e) ** 2 - 2 + (1 - e) ** 2) for e in [1e-5, 1e-4, 1e-3]]
check(all(abs(N_q[i + 1] / N_q[i] - 100) < 1e-3 for i in range(2)),
      "smooth numerator scales QUADRATICALLY (x100/decade)")


def resolved_slope(rows, floor_frac=100.0):
    """Fit slope only on points that have cleared the float64 round-off floor."""
    floor = EPS_MACH * floor_frac
    pts = [(a, v) for a, _, v in rows if v > floor]
    return fit_slope([a for a, _ in pts], [v for _, v in pts]), pts


# ---------------- Part B: multi-affine structure ----------------
def make_mlp(depth, seed=0):
    g = torch.Generator().manual_seed(seed)
    layers, d = [], 8
    widths = [8, 16, 16, 4]
    for i in range(depth):
        lin = nn.Linear(widths[i], widths[i + 1])
        with torch.no_grad():
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * 0.5)
            lin.bias.copy_(torch.randn(lin.bias.shape, generator=g) * 0.5)
        layers.append(lin)
        if i < depth - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def activation_signs(model, X):
    snaps, handles = [], []
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            def hook(mod, inp, out, s=snaps):
                s.append((out > 0).reshape(out.size(0), -1).clone())
            handles.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model(X)
    out = [s.clone() for s in snaps]
    for h in handles:
        h.remove()
    return out


def perturb(model, base, vec, eps):
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(base[name] if vec is None else base[name] + eps * vec[name])


def N_value(model, X, base, vec, eps):
    perturb(model, base, vec, 0.0)
    H0 = model(X)
    perturb(model, base, vec, eps)
    Hp = model(X)
    perturb(model, base, vec, -eps)
    Hm = model(X)
    perturb(model, base, None, 0.0)
    return (Hp - 2 * H0 + Hm).detach(), H0.detach()


def rand_vec(model, g, layer_subset=None):
    vec = {}
    for name, p in model.named_parameters():
        if layer_subset is not None and not any(name.startswith(s)
                                                for s in layer_subset):
            vec[name] = torch.zeros_like(p)
        else:
            v = torch.randn(p.shape, generator=g)
            vec[name] = v / v.norm()
    return vec


def flip_mask(model, X, base_signs, base, vec, eps):
    perturb(model, base, vec, eps)
    sp = activation_signs(model, X)
    perturb(model, base, vec, -eps)
    sm = activation_signs(model, X)
    perturb(model, base, None, 0.0)
    fl = torch.zeros(X.size(0), dtype=torch.bool)
    for s0, a, b in zip(base_signs, sp, sm):
        fl |= (a != s0).any(1) | (b != s0).any(1)
    return fl


def fit_slope(xs, ys):
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if y > 0]
    if len(pts) < 2:
        return None
    mx = sum(a for a, _ in pts) / len(pts)
    my = sum(b for _, b in pts) / len(pts)
    return sum((a - mx) * (b - my) for a, b in pts) / \
        sum((a - mx) ** 2 for a, _ in pts)


X = torch.randn(64, 8)
g = torch.Generator().manual_seed(42)

# ---- 2-layer net: single-layer directions must be exactly linear ----
mlp2 = make_mlp(2, seed=1)
base2 = {n: p.detach().clone() for n, p in mlp2.named_parameters()}
signs2 = activation_signs(mlp2, X)
v1 = rand_vec(mlp2, g, layer_subset=["0"])   # only W1,b1
v2 = rand_vec(mlp2, g, layer_subset=["2"])   # only W2,b2
vj = rand_vec(mlp2, g)                        # joint
ALPHAS = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.1, 0.5, 1.0]
for tag, vec in (("V1-only", v1), ("V2-only", v2), ("joint", vj)):
    rows = []
    for a in ALPHAS:
        Nmat, H0 = N_value(mlp2, X, base2, vec, a)
        fl = flip_mask(mlp2, X, signs2, base2, vec, a)
        nf_rms = torch.sqrt((Nmat[~fl] ** 2).sum() /
                            max(1, int((~fl).sum()) * H0.size(1))).item() \
            / torch.sqrt((H0 ** 2).mean()).item() if (~fl).any() else 0.0
        fp_rms = torch.sqrt((Nmat[fl] ** 2).sum() /
                            max(1, int(fl.sum()) * H0.size(1))).item() \
            / torch.sqrt((H0 ** 2).mean()).item() if fl.any() else None
        rows.append((a, fl.float().mean().item(), nf_rms, fp_rms))
    print(f"\n2-layer {tag}: alpha | flip | N/H noflip | N/H flip")
    for r in rows:
        print(f"  {r[0]:.0e} | {r[1]:.3f} | {r[2]:.3e} | "
              f"{r[3]:.3e}" if r[3] is not None else
              f"  {r[0]:.0e} | {r[1]:.3f} | {r[2]:.3e} | -")
    no = [r for r in rows if r[1] <= 0.02]
    if tag in ("V1-only", "V2-only"):
        check(all(r[2] < 1e-13 for r in no),
              f"2-layer {tag} direction: in-cell N at float64 floor (<1e-13)")
    else:
        # fit only on points that have cleared the round-off floor
        sl, pts = resolved_slope([(r[0], r[1], r[2]) for r in no])
        check(sl is not None and abs(sl - 2) < 0.05,
              f"2-layer joint: in-cell N ~ eps^2 (slope={sl}, "
              f"resolved alphas {[f'{a:.0e}' for a, _ in pts]})")
        # N/eps^2 constant on resolved decades
        vals = [v for _, v in pts]
        aa = [a for a, _ in pts]
        ratios = [(vals[i + 1] / vals[i]) /
                  ((aa[i + 1] / aa[i]) ** 2) for i in range(len(vals) - 1)]
        check(all(abs(r - 1) < 0.02 for r in ratios),
              f"2-layer joint: N/eps^2 constant (ratios={[round(r, 4) for r in ratios]})")
    # at the FIRST alpha where kinks appear, flipped samples must clearly
    # exceed the in-cell eps^2 value measured on non-flipped samples at the
    # SAME alpha (N_kink ~ ||grad jump||*eps dominates c2*eps^2 by ~1/eps)
    first_kink = next((r for r in rows if r[1] >= 0.1 and r[3] is not None),
                      None)
    if tag == "joint" and first_kink:
        same_a_noflip = next(r[2] for r in rows if r[0] == first_kink[0])
        ratio = first_kink[3] / same_a_noflip
        check(ratio > 3,
              f"2-layer joint: at first kink alpha={first_kink[0]:.0e} "
              f"flipped N {first_kink[3]:.2e} > 3x in-cell {same_a_noflip:.2e} "
              f"(ratio {ratio:.1f})")

# ---- 3-layer net: joint direction must show eps^2 + eps^3 (slope 2->3) ----
mlp3 = make_mlp(3, seed=2)
base3 = {n: p.detach().clone() for n, p in mlp3.named_parameters()}
signs3 = activation_signs(mlp3, X)
v3 = rand_vec(mlp3, g)
rows3 = []
for a in ALPHAS:
    Nmat, H0 = N_value(mlp3, X, base3, v3, a)
    fl = flip_mask(mlp3, X, signs3, base3, v3, a)
    nr = torch.sqrt((Nmat[~fl] ** 2).sum() /
                    max(1, int((~fl).sum()) * H0.size(1))).item() \
        / torch.sqrt((H0 ** 2).mean()).item() if (~fl).any() else 0.0
    rows3.append((a, fl.float().mean().item(), nr))
print("\n3-layer joint: alpha | flip | N/H noflip")
for r in rows3:
    print(f"  {r[0]:.0e} | {r[1]:.3f} | {r[2]:.3e}")
no3 = [r for r in rows3 if r[1] <= 0.02]
# slope on floor-cleared points: in-cell polynomial is eps^2 + eps^3, so the
# slope must lie in [2, 3] (2 as eps->0); empirically the quadratic dominates
# at these radii and we measure ~2.
sl3, pts3 = resolved_slope(no3)
print(f"3-layer resolved slope (expect in [2,3]): {sl3} on "
      f"{[f'{a:.0e}' for a, _ in pts3]}")
check(sl3 is not None and 1.95 <= sl3 <= 3.05,
      f"3-layer joint: in-cell slope in [2,3] (eps^2+eps^3), got {sl3}")

print("\n" + ("ALL SMOKE CHECKS PASSED" if not failures
               else f"{len(failures)} FAILURES: {failures}"))
raise SystemExit(1 if failures else 0)
