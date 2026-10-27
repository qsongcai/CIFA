"""
End-to-end ViT-B/16 CIFA experiment on PACS (leave-one-domain-out).

Independent script — does NOT modify or import CIFA-Repro.
Implements the same CIFA mechanism (content-baseline / style-deviation
decomposition + deviation mixing) applied to a ViT intermediate block,
plus SWAD weight averaging and ERM baseline.

ViT feature at block L has shape [B, N, D] (N=197 = 196 patches + 1 CLS).
We exclude the CLS token from moment statistics, divide D into G groups,
and compute per-group mean/std across the N-1 patch tokens.

Usage:
    python vit_cifa_e2e.py --target_domain sketch --method cifa --seed 0
    python vit_cifa_e2e.py --target_domain sketch --method swad --seed 0
    python vit_cifa_e2e.py --target_domain sketch --method erm  --seed 0
    # layer ablation:
    python vit_cifa_e2e.py --target_domain sketch --method cifa --seed 0 --vit_block 3
"""
import argparse
import json
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image

# ============================================================
# 1. SWAD components (identical logic to CIFA-Repro/swad.py)
# ============================================================

class LossValley:
    def __init__(self, n_converge=3, n_tolerance=6, tolerance_ratio=0.3):
        self.n_converge = n_converge
        self.n_tolerance = n_tolerance
        self.tolerance_ratio = tolerance_ratio
        self.reset()

    def reset(self):
        self.losses = []
        self.start = -1
        self.end = -1
        self.mean = None
        self.above = 0

    def _is_converged(self):
        if len(self.losses) < self.n_converge:
            return False
        window = self.losses[-self.n_converge:]
        return window[0] <= min(window[1:])

    def _is_tolerance(self, idx, loss):
        if self.mean is None:
            segment = self.losses[self.start:]
            self.mean = sum(segment) / len(segment)
        if loss > self.mean * (1.0 + self.tolerance_ratio):
            self.above += 1
        else:
            self.above = 0
            self.mean = None
        if self.above >= self.n_tolerance:
            self.end = idx
        return self.end

    def update(self, idx, loss):
        self.losses.append(loss)
        if self.start < 0 and self._is_converged():
            self.start = idx - self.n_converge + 1
        if self.start >= 0 and self.end < 0:
            self._is_tolerance(idx, loss)
        return self.start, self.end

    def need_average(self, idx):
        return self.start >= 0 and (self.end < 0 or idx <= self.end)


class AveragedModel:
    def __init__(self, model):
        self.n = 0
        self.module = deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        avg_params = dict(self.module.named_parameters())
        for name, p in model.named_parameters():
            avg_params[name].add_((p.detach() - avg_params[name]) / self.n)
        for name, b in model.named_buffers():
            self.module.get_buffer(name).copy_(b)

    def state_dict(self):
        return self.module.state_dict()


# ============================================================
# 2. CIFA components for ViT grouped moments
# ============================================================

def vit_instance_moments(Z, num_groups, eps):
    """Grouped mean/std of ViT feature Z [B, N, D], excluding CLS token.

    Returns mu, sigma each [B, G] where G = num_groups.
    Statistics are computed across the N-1 patch tokens within each group.
    """
    B, N, D = Z.shape
    G = num_groups
    assert D % G == 0, f"feat_dim {D} not divisible by num_groups {G}"
    group_dim = D // G
    # exclude CLS (first token): [B, N-1, D]
    patches = Z[:, 1:, :]
    # reshape to [B, N-1, G, group_dim]
    x = patches.reshape(B, N - 1, G, group_dim)
    # mean/std across patch tokens (dim=1) and within group (dim=3)
    mu = x.mean(dim=(1, 3))       # [B, G]
    var = ((x - mu.unsqueeze(1).unsqueeze(3)) ** 2).mean(dim=(1, 3))
    sigma = torch.sqrt(var + eps)  # [B, G]
    return mu, sigma


class ViTBaselineTracker:
    """Per-class EMA prototypes of grouped moments at the ViT block."""

    def __init__(self, num_classes, num_groups, momentum=0.9, device="cpu"):
        self.momentum = momentum
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.proto_mu = torch.zeros(num_classes, num_groups, device=device)
        self.proto_sigma = torch.zeros(num_classes, num_groups, device=device)
        self.seen = torch.zeros(num_classes, device=device)
        self.global_mu = torch.zeros(num_groups, device=device)
        self.global_sigma = torch.zeros(num_groups, device=device)
        self.global_seen = 0

    @torch.no_grad()
    def update(self, mu, sigma, y):
        """mu,sigma: [B,G] detached clean moments; y: [B]."""
        classes = torch.unique(y)
        for k in classes:
            idx = (y == k)
            bm = mu[idx].mean(dim=0)
            bs = sigma[idx].mean(dim=0)
            kk = int(k)
            if self.seen[kk] < 0.5:
                self.proto_mu[kk] = bm
                self.proto_sigma[kk] = bs
            else:
                m = self.momentum
                self.proto_mu[kk].mul_(m).add_(bm, alpha=1 - m)
                self.proto_sigma[kk].mul_(m).add_(bs, alpha=1 - m)
            self.seen[kk] = 1.0
        gm = mu.mean(dim=0)
        gs = sigma.mean(dim=0)
        if self.global_seen == 0:
            self.global_mu = gm
            self.global_sigma = gs
        else:
            m = self.momentum
            self.global_mu.mul_(m).add_(gm, alpha=1 - m)
            self.global_sigma.mul_(m).add_(gs, alpha=1 - m)
        self.global_seen = 1

    @torch.no_grad()
    def get_baseline(self, y):
        b_mu = self.proto_mu[y].clone()
        b_sigma = self.proto_sigma[y].clone()
        unseen = (self.seen[y] < 0.5)
        if unseen.any():
            b_mu[unseen] = self.global_mu
            b_sigma[unseen] = self.global_sigma
        return b_mu, b_sigma


def _derangement(B, device):
    if B <= 1:
        return torch.randperm(B, device=device)
    while True:
        perm = torch.randperm(B, device=device)
        if (perm != torch.arange(B, device=device)).all():
            return perm


def vit_augment(Z, y, tracker, num_groups, eps, sigma_min,
                lam_min=0.1, lam_max=0.9):
    """Apply CIFA deviation-mixing to ViT feature Z [B, N, D].

    Only patch tokens are modified; CLS token is passed through unchanged.
    Statistics are constants (no_grad); gradient flows through the
    normalized residual, exactly as in the ResNet version.
    """
    B, N, D = Z.shape
    G = num_groups
    group_dim = D // G

    with torch.no_grad():
        mu, sigma = vit_instance_moments(Z, G, eps)  # [B, G]
        b_mu, b_sigma = tracker.get_baseline(y)       # [B, G]
        d_mu = mu - b_mu
        d_sigma = sigma - b_sigma

        perm = _derangement(B, Z.device)
        d_mu_p = d_mu[perm]
        d_sigma_p = d_sigma[perm]
        lam = torch.rand(B, 1, device=Z.device) * (lam_max - lam_min) + lam_min
        da_mu = lam * d_mu + (1.0 - lam) * d_mu_p
        da_sigma = lam * d_sigma + (1.0 - lam) * d_sigma_p

        new_mu = b_mu + da_mu            # [B, G]
        new_sigma = (b_sigma + da_sigma).clamp_min(sigma_min)  # [B, G]

    # normalized residual keeps gradient path: [B, N-1, G, gd]
    patches = Z[:, 1:, :]
    x = patches.reshape(B, N - 1, G, group_dim)
    mu_exp = mu.unsqueeze(1).unsqueeze(3)       # [B,1,G,1]
    sigma_exp = sigma.unsqueeze(1).unsqueeze(3)  # [B,1,G,1]
    x_norm = (x - mu_exp) / sigma_exp

    new_mu_exp = new_mu.unsqueeze(1).unsqueeze(3)
    new_sigma_exp = new_sigma.unsqueeze(1).unsqueeze(3)
    x_aug = new_sigma_exp * x_norm + new_mu_exp
    patches_aug = x_aug.reshape(B, N - 1, D)

    # CLS token unchanged
    Z_aug = torch.cat([Z[:, :1, :], patches_aug], dim=1)
    return Z_aug


# ============================================================
# 3. ViT model with intermediate-block hook
# ============================================================

class ViTCIFA(nn.Module):
    """ViT-B/16 wrapper (torchvision) that extracts/reinjects at block L.

    Uses torchvision.models.vit_b_16 (ImageNet-1K pretrained, cached locally).
    The forward pass runs patch_embed + blocks 0..L, optionally applies CIFA
    to the block-L output, then runs blocks L+1..end and the head.
    """

    def __init__(self, num_classes=7, vit_block=6, pretrained=True):
        super().__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)
        self.embed_dim = self.vit.hidden_dim  # 768
        self.num_blocks = len(self.vit.encoder.layers)  # 12
        assert 0 <= vit_block < self.num_blocks
        self.vit_block = vit_block
        # replace the classification head
        self.vit.heads = nn.Linear(self.embed_dim, num_classes)

    def _forward_to_block(self, x, block_idx):
        """Run patch embed + pos + blocks 0..block_idx, return [B, N, D]."""
        # torchvision ViT forward_features up to encoder layers
        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)
        for i in range(block_idx + 1):
            x = self.vit.encoder.layers[i](x)
        return x

    def _forward_from_block(self, x, block_idx):
        """Run blocks block_idx+1..end + ln + head, return logits."""
        for i in range(block_idx + 1, self.num_blocks):
            x = self.vit.encoder.layers[i](x)
        x = self.vit.encoder.ln(x)
        # CLS token classification
        x = x[:, 0]
        return self.vit.heads(x)

    def forward_path(self, x, y, tracker, augment, cfg):
        """One full forward pass; collect clean moments or augment at block L."""
        feat = self._forward_to_block(x, self.vit_block)
        if augment:
            feat = vit_augment(
                feat, y, tracker,
                num_groups=cfg["num_groups"],
                eps=cfg["moments_eps"],
                sigma_min=cfg["sigma_min"],
                lam_min=cfg["lambda_min"],
                lam_max=cfg["lambda_max"])
            moments = None
        else:
            with torch.no_grad():
                mu, sigma = vit_instance_moments(
                    feat, cfg["num_groups"], cfg["moments_eps"])
            moments = (mu, sigma)
        logits = self._forward_from_block(feat, self.vit_block)
        return logits, moments

    def forward(self, x, y, tracker, cfg):
        """Clean path (updates baseline) + optional augmented path."""
        logits_c, moments = self.forward_path(x, y, tracker, augment=False, cfg=cfg)
        if tracker is not None and moments is not None:
            mu, sigma = moments
            tracker.update(mu.detach(), sigma.detach(), y)

        logits_a = None
        if self.training and cfg["cifa_enabled"] and tracker is not None:
            if random.random() < cfg["cifa_prob"]:
                logits_a, _ = self.forward_path(x, y, tracker, augment=True, cfg=cfg)
        return logits_c, logits_a


# ============================================================
# 4. PACS data loading (DomainBed LODO protocol, self-contained)
# ============================================================

PACS_DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
VLCS_DOMAINS = ["CALTECH", "LABELME", "PASCAL", "SUN"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImageList(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        return self.transform(img), label


def _split_indices(n, holdout_fraction, seed):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_holdout = int(round(n * holdout_fraction))
    return idx[n_holdout:], idx[:n_holdout]


def get_pacs_loo(data_dir, target_domain, holdout_fraction=0.2, seed=0):
    from torchvision import datasets
    source_domains = [d for d in PACS_DOMAINS if d != target_domain]
    train_samples = []
    holdout_by_domain = {}
    classes = None

    for di, domain in enumerate(source_domains):
        folder = datasets.ImageFolder(os.path.join(data_dir, domain))
        if classes is None:
            classes = folder.classes
        tr_idx, va_idx = _split_indices(
            len(folder.samples), holdout_fraction, seed + 1000 * di)
        train_samples += [folder.samples[i] for i in tr_idx]
        holdout_by_domain[domain] = ImageList(
            [folder.samples[i] for i in va_idx], test_transform())

    target_folder = datasets.ImageFolder(os.path.join(data_dir, target_domain))
    target = ImageList(target_folder.samples, test_transform())

    rng = random.Random(seed + 777)
    rng.shuffle(train_samples)

    return {
        "train_dataset": ImageList(train_samples, train_transform()),
        "holdout_by_domain": holdout_by_domain,
        "target": target,
        "classes": classes,
    }


def get_vlcs_loo(data_dir, target_domain, holdout_fraction=0.2, seed=0):
    """VLCS LODO: source domains use 'train' split, target uses 'test' split."""
    from torchvision import datasets
    source_domains = [d for d in VLCS_DOMAINS if d != target_domain]
    train_samples = []
    holdout_by_domain = {}
    classes = None

    for di, domain in enumerate(source_domains):
        folder = datasets.ImageFolder(os.path.join(data_dir, domain, "train"))
        if classes is None:
            classes = folder.classes
        tr_idx, va_idx = _split_indices(
            len(folder.samples), holdout_fraction, seed + 1000 * di)
        train_samples += [folder.samples[i] for i in tr_idx]
        holdout_by_domain[domain] = ImageList(
            [folder.samples[i] for i in va_idx], test_transform())

    target_folder = datasets.ImageFolder(os.path.join(data_dir, target_domain, "test"))
    target = ImageList(target_folder.samples, test_transform())

    rng = random.Random(seed + 777)
    rng.shuffle(train_samples)

    return {
        "train_dataset": ImageList(train_samples, train_transform()),
        "holdout_by_domain": holdout_by_domain,
        "target": target,
        "classes": classes,
    }


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


# ============================================================
# 5. Training
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, dataset, cfg, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model.forward_path(x, y, None, augment=False, cfg=cfg)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n


def combine_views(loss_c, logits_c, logits_a, y, cfg):
    """Soft worst-case aggregation + content-consistency guard."""
    loss_a = F.cross_entropy(logits_a, y)
    with torch.no_grad():
        p_c = F.softmax(logits_c, 1)
        logp_c = F.log_softmax(logits_c, 1)
        logp_a = F.log_softmax(logits_a, 1)
        cons = (p_c * (logp_c - logp_a)).sum(1).mean()

    risks = torch.stack([loss_c.detach(), loss_a.detach()])
    q = F.softmax(cfg["soft_eta"] * risks, dim=0)

    if cfg["use_guard"]:
        gate = (cfg["consistency_delta"] /
                (cons.detach() + 1e-8)).clamp(max=1.0)
        loss = q[0] * loss_c + gate * (
            q[1] * loss_a + cfg["consistency_coef"] * cons)
    else:
        gate = torch.ones((), device=logits_c.device)
        loss = q[0] * loss_c + q[1] * loss_a
    return loss, gate, cons


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="pacs", choices=["pacs", "vlcs"])
    p.add_argument("--data_dir", default=None)
    p.add_argument("--target_domain", required=True)
    p.add_argument("--method", required=True, choices=["erm", "swad", "cifa"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="vit_e2e_results")
    p.add_argument("--vit_block", type=int, default=6,
                   help="block index (0-11) at which to apply CIFA")
    # optimization
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--holdout_fraction", type=float, default=0.2)
    # CIFA
    p.add_argument("--num_groups", type=int, default=32)
    p.add_argument("--cifa_prob", type=float, default=0.5)
    p.add_argument("--moments_eps", type=float, default=1e-5)
    p.add_argument("--sigma_min", type=float, default=1e-3)
    p.add_argument("--lambda_min", type=float, default=0.1)
    p.add_argument("--lambda_max", type=float, default=0.9)
    p.add_argument("--baseline_momentum", type=float, default=0.9)
    p.add_argument("--soft_eta", type=float, default=0.5)
    p.add_argument("--consistency_coef", type=float, default=1.0)
    p.add_argument("--consistency_delta", type=float, default=2.0)
    p.add_argument("--use_guard", action="store_true", default=True)
    p.add_argument("--no_guard", action="store_true")
    # SWAD
    p.add_argument("--swad_n_converge", type=int, default=3)
    p.add_argument("--swad_n_tolerance", type=int, default=6)
    p.add_argument("--swad_tolerance_ratio", type=float, default=0.3)
    p.add_argument("--eval_every", type=int, default=200,
                   help="holdout eval interval in steps")
    args = p.parse_args()

    if args.no_guard:
        args.use_guard = False

    # Dataset-specific defaults
    if args.data_dir is None:
        if args.dataset == "pacs":
            args.data_dir = "/mnt/sdd1/datasets_qsong/PACS/Homework3-PACS-master/PACS"
        elif args.dataset == "vlcs":
            args.data_dir = "/home/qsong/qsong/data/VLCS"

    cfg = vars(args)
    cfg["cifa_enabled"] = (args.method == "cifa")
    cfg["swad_enabled"] = (args.method in ("swad", "cifa"))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== ViT E2E: dataset={args.dataset} method={args.method} "
          f"target={args.target_domain} seed={args.seed} block={args.vit_block} ===", flush=True)
    print(f"Device: {device}", flush=True)

    if args.dataset == "pacs":
        data = get_pacs_loo(args.data_dir, args.target_domain,
                            holdout_fraction=args.holdout_fraction, seed=args.seed)
    elif args.dataset == "vlcs":
        data = get_vlcs_loo(args.data_dir, args.target_domain,
                            holdout_fraction=args.holdout_fraction, seed=args.seed)
    num_classes = len(data["classes"])
    print(f"Train: {len(data['train_dataset'])}, "
          f"Target: {len(data['target'])}, classes={num_classes}", flush=True)

    train_loader = DataLoader(
        data["train_dataset"], batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=8, pin_memory=True)
    steps_per_epoch = max(1, len(data["train_dataset"]) // args.batch_size)

    model = ViTCIFA(
        num_classes=num_classes, vit_block=args.vit_block,
        pretrained=True).to(device)

    tracker = None
    if cfg["cifa_enabled"]:
        tracker = ViTBaselineTracker(
            num_classes, args.num_groups,
            momentum=args.baseline_momentum, device=device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    valley = None
    averaged = None
    if cfg["swad_enabled"]:
        valley = LossValley(
            n_converge=args.swad_n_converge,
            n_tolerance=args.swad_n_tolerance,
            tolerance_ratio=args.swad_tolerance_ratio)

    merged_holdout = ConcatDataset(list(data["holdout_by_domain"].values()))
    diag = {"n_aug": 0, "cons_sum": 0.0, "gate_sum": 0.0}

    model.train()
    loader_it = infinite_loader(train_loader)
    eval_counter = 0

    for step in range(1, args.steps + 1):
        x, y = next(loader_it)
        x, y = x.to(device), y.to(device)

        logits_c, logits_a = model(x, y, tracker, cfg)
        loss_c = F.cross_entropy(logits_c, y)

        if logits_a is not None:
            loss, gate, cons = combine_views(loss_c, logits_c, logits_a, y, cfg)
            diag["n_aug"] += 1
            diag["cons_sum"] += cons.item()
            diag["gate_sum"] += gate.item()
        else:
            loss = loss_c

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"[step {step}/{args.steps}] loss={loss.item():.4f}", flush=True)

        # periodic holdout eval for SWAD
        if step % args.eval_every == 0:
            eval_counter += 1
            if cfg["swad_enabled"]:
                val_loss, _ = evaluate(model, merged_holdout, cfg, device)
                start, end = valley.update(eval_counter, val_loss)
                if valley.need_average(eval_counter):
                    if averaged is None:
                        averaged = AveragedModel(model)
                    averaged.update(model)
                model.train()

    # final model selection
    if cfg["swad_enabled"] and averaged is not None and averaged.n > 0:
        model.load_state_dict(averaged.state_dict())
        print(f"SWAD averaged {averaged.n} checkpoints", flush=True)

    # evaluate
    holdout_acc = {}
    for dom, ds in data["holdout_by_domain"].items():
        _, acc = evaluate(model, ds, cfg, device)
        holdout_acc[dom] = acc
    worst_holdout = min(holdout_acc.values())

    target_loss, target_acc = evaluate(model, data["target"], cfg, device)
    print(f"target={args.target_domain} seed={args.seed} "
          f"acc={target_acc*100:.2f} worst_holdout={worst_holdout*100:.2f}",
          flush=True)

    if diag["n_aug"] > 0:
        mean_cons = diag["cons_sum"] / diag["n_aug"]
        mean_gate = diag["gate_sum"] / diag["n_aug"]
        print(f"[diag] aug triggered {diag['n_aug']}/{args.steps} "
              f"({diag['n_aug']/args.steps:.0%}), "
              f"mean KL={mean_cons:.3f}, mean gate={mean_gate:.3f}", flush=True)
    else:
        mean_cons = mean_gate = None

    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        "dataset": args.dataset,
        "method": args.method,
        "target_domain": args.target_domain,
        "seed": args.seed,
        "vit_block": args.vit_block,
        "backbone": "ViT-B16_224_in1k_e2e",
        "target_acc": target_acc,
        "target_loss": target_loss,
        "holdout_acc": holdout_acc,
        "worst_holdout_acc": worst_holdout,
        "swad_n_checkpoints": averaged.n if averaged is not None else 0,
        "aug_trigger_fraction": diag["n_aug"] / args.steps,
        "aug_mean_consistency_kl": mean_cons,
        "aug_mean_gate": mean_gate,
        "steps": args.steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
    }
    fn = os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.method}_{args.target_domain}_seed{args.seed}_block{args.vit_block}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", fn, flush=True)


if __name__ == "__main__":
    main()
