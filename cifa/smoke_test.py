"""CPU smoke test for CIFA (no dataset, no pretrained download).

Run:  python smoke_test.py
Checks:
  1. instance moments on a known tensor
  2. BaselineTracker update / EMA / fallback
  3. deviation-mixing operator (both mix and gaussian modes), baseline fixed
  4. end-to-end train steps with clean+aug forward, loss, backward, update
  5. SWAD valley detection, dense averaging, load averaged weights
"""
import torch
import torch.nn.functional as F

from config import RESNET_CHANNELS, hparams_registry
from cifa_aug import BaselineTracker, augment_layer, instance_moments, _derangement
from networks import CIFAResNet
from swad import AveragedModel, LossValley

DEVICE = torch.device("cpu")
torch.manual_seed(0)


def check(cond, msg):
    if not cond:
        raise AssertionError("FAIL: " + msg)
    print("  ok -", msg)


print("[1] instance moments")
Z = torch.ones(2, 3, 4, 4)
Z[0] = 0.0
mu, sigma = instance_moments(Z, 1e-5)
check(abs(mu[0].mean().item()) < 1e-6 and abs(mu[1].mean().item() - 1.0) < 1e-6,
      "channel mean computed correctly")
check(sigma[0].mean().item() < 5e-3, "constant channel has ~zero std (sqrt(eps))")

print("[2] BaselineTracker")
K, C = 7, 8
tracker = BaselineTracker(K, {"l1": C}, momentum=0.9, device=DEVICE)
y = torch.tensor([0, 0, 1])
mu_b = torch.tensor([[1.0] * C, [1.0] * C, [3.0] * C])
sg_b = torch.ones(3, C) * 0.5
tracker.update("l1", mu_b, sg_b, y)
check(tracker.seen["l1"][0] == 1 and tracker.seen["l1"][1] == 1,
      "class prototypes marked seen")
check(abs(tracker.proto_mu["l1"][0].mean().item() - 1.0) < 1e-6 and
      abs(tracker.proto_mu["l1"][1].mean().item() - 3.0) < 1e-6,
      "class-prototype mean correct")
# EMA update for class 0
tracker.update("l1", torch.ones(1, C) * 2.0, torch.ones(1, C) * 0.5,
                torch.tensor([0]))
check(abs(tracker.proto_mu["l1"][0].mean().item() - 1.1) < 1e-6,
      "EMA update applied (0.9*1 + 0.1*2 = 1.1)")
bm, bs = tracker.get_baseline("l1", torch.tensor([0, 5]))
check(abs(bm[0].mean().item() - 1.1) < 1e-6, "seen class uses prototype")
check(abs(bm[1].mean().item() - tracker.global_mu["l1"].mean().item()) < 1e-6,
      "unseen class falls back to global")

print("[3] deviation-mixing operator")
B, Cc, H, W = 4, 8, 6, 6
Zx = torch.randn(B, Cc, H, W)
yy = torch.tensor([0, 0, 1, 1])
tr = BaselineTracker(7, {"l1": Cc}, device=DEVICE)
mu0, sg0 = instance_moments(Zx, 1e-5)
tr.update("l1", mu0.flatten(1), sg0.flatten(1), yy)
cfg = hparams_registry()
cfg["cifa_mix_mode"] = "mix"
Za = augment_layer("l1", Zx, yy, tr, cfg)
check(Za.shape == Zx.shape, "aug output shape preserved")
# after augmentation, instance moments should sit near (baseline + mixed dev),
# i.e. baseline component is retained: check mean close to baseline on average
mua, sga = instance_moments(Za, 1e-5)
base_mu, _ = tr.get_baseline("l1", yy)
check((mua.flatten(1).mean() - base_mu.mean()).abs().item() < 0.5,
      "augmented means stay anchored to content baseline")
perm = _derangement(16, DEVICE)
check((perm != torch.arange(16)).all(), "derangement has no fixed points")
cfg["cifa_mix_mode"] = "gaussian"
Zg = augment_layer("l1", Zx, yy, tr, cfg)
check(Zg.shape == Zx.shape and torch.isfinite(Zg).all(),
      "gaussian mode produces finite output")

print("[4] end-to-end train steps (small resnet18, no pretrained)")
model = CIFAResNet("resnet18", pretrained=False, num_classes=7,
                   layers=("layer1", "layer2"), freeze_bn=True).to(DEVICE)
ch = {n: RESNET_CHANNELS["resnet18"][n] for n in ("layer1", "layer2")}
tracker = BaselineTracker(7, ch, momentum=0.9, device=DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
cfg = hparams_registry()
cfg["cifa_prob"] = 1.0   # force augmentation every step
w0 = model.fc.weight.detach().clone()
for it in range(4):
    x = torch.randn(4, 3, 64, 64)
    yb = torch.randint(0, 7, (4,))
    model.train()
    lc, la = model(x, yb, tracker, cfg)
    check(la is not None, f"aug view present (step {it})")
    loss_a = F.cross_entropy(la, yb)
    with torch.no_grad():
        pc = F.softmax(lc, 1)
        cons = (pc * (F.log_softmax(lc, 1) - F.log_softmax(la, 1))).sum(1).mean()
    risks = torch.stack([F.cross_entropy(lc, yb).detach(), loss_a.detach()])
    q = F.softmax(cfg["cifa_soft_eta"] * risks, 0)
    loss = q[0] * F.cross_entropy(lc, yb) + q[1] * loss_a + \
        cfg["cifa_consistency_coef"] * cons
    check(torch.isfinite(loss), f"finite loss (step {it})")
    opt.zero_grad(); loss.backward(); opt.step()
check((model.fc.weight.detach() - w0).abs().sum().item() > 0,
      "parameters updated by optimizer")

# gradient connectivity: augmented view must reach layers BEFORE the aug point
model.zero_grad()
model.train()
cfg["cifa_prob"] = 1.0
xg = torch.randn(4, 3, 64, 64)
yg = torch.randint(0, 7, (4,))
_, la_g = model(xg, yg, tracker, cfg)
F.cross_entropy(la_g, yg).backward()
g_stem = model.stem[0].weight.grad
check(g_stem is not None and g_stem.abs().sum().item() > 0,
      "augmented-view gradient propagates through the augmentation point to earlier layers")
model.zero_grad()

# no-aug path
cfg["cifa_enabled"] = False
model.eval()
with torch.no_grad():
    x = torch.randn(4, 3, 64, 64)
    yb = torch.randint(0, 7, (4,))
    lc, la = model(x, yb, tracker, cfg)
check(la is None, "aug view absent when disabled / eval")

print("[5] SWAD valley + dense averaging")
valley = LossValley(n_converge=3, n_tolerance=6, tolerance_ratio=0.3,
                    tolerance_epoch=5)
averaged = None
# decreasing then plateau, then cap by tolerance_epoch
losses = [3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
for i, l in enumerate(losses):
    start, end = valley.update(i, l)
    if valley.need_average(i):
        if averaged is None:
            averaged = AveragedModel(model)
        averaged.update(model)
check(start >= 0, "valley start detected")
check(end >= 0, "valley end detected (tolerance_epoch cap)")
check(averaged is not None and averaged.n > 0, "dense average accumulated")
model.load_state_dict(averaged.state_dict())
model.eval()
with torch.no_grad():
    out = model.forward_path(torch.randn(2, 3, 64, 64),
                             torch.tensor([0, 1]), None, False, cfg)[0]
check(out.shape == (2, 7) and torch.isfinite(out).all(),
      "averaged model runs and outputs logits")

print("\nALL SMOKE TESTS PASSED")
