"""
R3-(4) Feature-space Trajectory-Augmentation Alignment Verification.

Extends trajectory_alignment.py with feature-space metrics:
  - Instance-moment (channel-wise mean/std) distribution alignment
  - Feature-space nearest-neighbor coverage ratio
  - CKA (centered kernel alignment) structural similarity

Hypothesis: SWAD-window checkpoints' feature distributions at layer1/layer2
are aligned with CIFA augmentation-induced feature distributions.

Output: feature_alignment_{dataset}_{domain}_seed{seed}.json
"""
import argparse
import json
import os
import random
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from config import RESNET_CHANNELS, hparams_registry, merge_overrides
from datasets import DATASETS, get_dataset, infinite_loader
from networks import CIFAResNet
from cifa_aug import BaselineTracker, instance_moments
from swad import AveragedModel, LossValley


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Feature extraction
# ============================================================

@torch.no_grad()
def extract_instance_moments(model, dataset, cfg, device, layers=("layer1", "layer2"),
                             batch_size=128, augment=False, tracker=None, seed_offset=0):
    """Extract per-sample channel-wise moments (mean, std) at specified layers.

    Returns dict: layer_name -> tensor [N, 2*C] (concat of mean and std)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    eps = cfg["cifa_moments_eps"]

    was_training = model.training
    if augment:
        model.train()
    else:
        model.eval()

    all_moments = {name: [] for name in layers}

    if augment:
        import random as pyrandom
        torch.manual_seed(seed_offset)
        pyrandom.seed(seed_offset + 10000)

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            h = model.stem(x)
            block_names = ["layer1", "layer2", "layer3", "layer4"]
            for bname in block_names:
                h = getattr(model, bname)(h)
                if bname in layers:
                    if augment and bname in model.aug_layers:
                        # Need to apply augmentation manually
                        from cifa_aug import augment_layer
                        h = augment_layer(bname, h, y, tracker, cfg)
                    mu, sigma = instance_moments(h, eps)
                    # mu, sigma: [B, C, 1, 1] -> [B, C]
                    mu = mu.flatten(1)
                    sigma = sigma.flatten(1)
                    all_moments[bname].append(torch.cat([mu, sigma], dim=1).cpu())

    if augment:
        # For multiple augmentation views, caller loops and calls this repeatedly
        model.train(was_training)
    else:
        model.train(was_training)

    return {name: torch.cat(parts, dim=0) for name, parts in all_moments.items()}


def extract_moments_multi_aug(model, dataset, cfg, device, n_aug,
                              layers=("layer1", "layer2"), tracker=None):
    """Extract instance moments for n_aug random augmentation views.

    Returns dict: layer -> tensor [n_aug, N, 2*C]
    """
    results = {name: [] for name in layers}
    for i in range(n_aug):
        moments = extract_instance_moments(
            model, dataset, cfg, device, layers=layers,
            augment=True, tracker=tracker, seed_offset=20000 + i)
        for name in layers:
            results[name].append(moments[name])
    return {name: torch.stack(parts, dim=0) for name, parts in results.items()}


# ============================================================
# Feature alignment metrics
# ============================================================

def feature_nn_ratio(moments_set1, moments_set2):
    """Nearest-neighbor coverage ratio in feature-moment space.

    moments_set1: [K1, N, D], moments_set2: [K2, N, D]
    Uses cosine distance on per-sample feature vectors.

    Returns: ratio (cross-NN / mean within-set NN), lower=more aligned
    """
    K1, N, D = moments_set1.shape
    K2 = moments_set2.shape[0]

    # Normalize for cosine distance
    def normalize(x):
        return F.normalize(x, dim=-1)

    s1 = normalize(moments_set1)  # [K1, N, D]
    s2 = normalize(moments_set2)  # [K2, N, D]

    # Pairwise cosine similarity, averaged over samples
    # [K1, K2, N] -> mean over N
    cross_sim = torch.einsum('ind,jnd->ijn', s1, s2).mean(dim=-1)  # [K1, K2]
    cross_dist = 1 - cross_sim
    nn_cross = cross_dist.min(dim=1).values.mean().item()

    # Within-set distances
    within1_sim = torch.einsum('ind,jnd->ijn', s1, s1).mean(dim=-1)
    within1_dist = 1 - within1_sim
    within1_dist.fill_diagonal_(float('inf'))
    nn_within1 = within1_dist.min(dim=1).values.mean().item()

    within2_sim = torch.einsum('ind,jnd->ijn', s2, s2).mean(dim=-1)
    within2_dist = 1 - within2_sim
    within2_dist.fill_diagonal_(float('inf'))
    nn_within2 = within2_dist.min(dim=1).values.mean().item()

    ratio = nn_cross / ((nn_within1 + nn_within2) / 2 + 1e-10)
    return nn_cross, nn_within1, nn_within2, ratio


def feature_moment_kl(moments_set1, moments_set2, n_channels):
    """KL-like divergence between mean feature-moment distributions.

    Treats each channel's mean and std as approximate Gaussian statistics,
    computes symmetric KL between the two sets' average moments.

    Returns: mean symmetric KL across channels
    """
    # Split into mean and std parts
    # moments: [K, N, 2*C] where first C = mean, last C = std
    m1 = moments_set1[:, :, :n_channels].mean(dim=0)   # [N, C]
    s1 = moments_set1[:, :, n_channels:].mean(dim=0)
    m2 = moments_set2[:, :, :n_channels].mean(dim=0)
    s2 = moments_set2[:, :, n_channels:].mean(dim=0)

    # Convert std to variance, add floor
    v1 = s1.pow(2) + 1e-6
    v2 = s2.pow(2) + 1e-6

    # Symmetric KL between Gaussians (per sample, per channel), then average
    # KL(N1||N2) = log(s2/s1) + (v1 + (m1-m2)^2)/(2*v2) - 0.5
    kl_12 = torch.log(s2 / s1 + 1e-10) + (v1 + (m1 - m2).pow(2)) / (2 * v2) - 0.5
    kl_21 = torch.log(s1 / s2 + 1e-10) + (v2 + (m2 - m1).pow(2)) / (2 * v1) - 0.5
    sym_kl = (kl_12 + kl_21) / 2

    return sym_kl.mean().item()


def cka_similarity(features1, features2):
    """Linear CKA between two feature matrices.

    features1: [N, D1], features2: [N, D2]
    Returns CKA value in [0, 1], higher=more similar.
    """
    # Center
    f1 = features1 - features1.mean(dim=0, keepdim=True)
    f2 = features2 - features2.mean(dim=0, keepdim=True)

    # Gram matrices
    g1 = f1 @ f1.T
    g2 = f2 @ f2.T

    # CKA = ||g1 g2||_F^2 / (||g1||_F^2 * ||g2||_F^2)
    numerator = (g1 * g2).sum().pow(2)
    denominator = (g1 * g1).sum() * (g2 * g2).sum() + 1e-10
    return (numerator / denominator).sqrt().item()


def feature_cka_comparison(moments_set1, moments_set2):
    """Compare CKA cross-set vs within-set.

    Returns: cross_cka, within_cka1, within_cka2
    """
    K1 = moments_set1.shape[0]
    K2 = moments_set2.shape[0]

    # Use average over samples as the feature representation per view
    # moments: [K, N, D] -> treat each view's per-sample features,
    # compute CKA between views across N samples
    def mean_pairwise_cka(feat_set):
        K = feat_set.shape[0]
        ckas = []
        for i in range(K):
            for j in range(i + 1, K):
                ckas.append(cka_similarity(feat_set[i], feat_set[j]))
        return np.mean(ckas) if ckas else 0.0

    # Cross-set CKA
    cross_ckas = []
    for i in range(K1):
        for j in range(K2):
            cross_ckas.append(cka_similarity(moments_set1[i], moments_set2[j]))
    cross_cka = np.mean(cross_ckas)

    within_cka1 = mean_pairwise_cka(moments_set1)
    within_cka2 = mean_pairwise_cka(moments_set2)

    return cross_cka, within_cka1, within_cka2


# ============================================================
# Training (reuse from trajectory_alignment)
# ============================================================

@torch.no_grad()
def evaluate(model, dataset, cfg, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _, _ = model.forward_path(x, y, None, augment=False, cfg=cfg)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n


def train_and_save_checkpoints(args, method="cifa"):
    """Train model and save SWAD-window (or late-training) checkpoints."""
    cfg = merge_overrides(hparams_registry(), {})
    if method == "erm":
        cfg["cifa_enabled"] = False
        cfg["swad"] = False
    elif method == "swad":
        cfg["cifa_enabled"] = False
        cfg["swad"] = True

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = get_dataset(args.dataset, args.data_dir, args.target_domain,
                       holdout_fraction=cfg["holdout_fraction"], seed=args.seed)
    num_classes = len(data["classes"])

    train_loader = DataLoader(
        data["train_dataset"], batch_size=cfg["batch_size"], shuffle=True,
        drop_last=True, num_workers=args.workers, pin_memory=True)
    steps_per_epoch = max(1, len(data["train_dataset"]) // cfg["batch_size"])

    model = CIFAResNet(
        backbone=cfg["backbone"], pretrained=cfg["pretrained"],
        num_classes=num_classes, layers=cfg["cifa_layers"],
        freeze_bn=cfg["freeze_bn"]).to(device)

    channels = {n: RESNET_CHANNELS[cfg["backbone"]][n] for n in cfg["cifa_layers"]}
    tracker = BaselineTracker(num_classes, channels,
                              momentum=cfg["cifa_baseline_momentum"], device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])

    valley = LossValley(
        n_converge=cfg["swad_n_converge"],
        n_tolerance=cfg["swad_n_tolerance"],
        tolerance_ratio=cfg["swad_tolerance_ratio"],
        tolerance_epoch=cfg["swad_tolerance_epoch"]) if cfg["swad"] else None
    averaged = None

    merged_holdout = ConcatDataset(list(data["holdout_by_domain"].values()))

    checkpoints = []
    model.train()
    loader_it = infinite_loader(train_loader)
    epoch = 0

    for step in range(1, cfg["steps"] + 1):
        x, y = next(loader_it)
        x, y = x.to(device), y.to(device)

        logits_c, logits_a = model(x, y, tracker, cfg)
        loss_c = F.cross_entropy(logits_c, y)

        if logits_a is not None:
            loss_a = F.cross_entropy(logits_a, y)
            with torch.no_grad():
                p_c = F.softmax(logits_c, 1)
                logp_c = F.log_softmax(logits_c, 1)
                logp_a = F.log_softmax(logits_a, 1)
                cons = (p_c * (logp_c - logp_a)).sum(1).mean()
            risks = torch.stack([loss_c.detach(), loss_a.detach()])
            q = F.softmax(cfg["cifa_soft_eta"] * risks, dim=0)
            gate = (cfg["cifa_consistency_delta"] / (cons.detach() + 1e-8)).clamp(max=1.0)
            loss = q[0] * loss_c + gate * (q[1] * loss_a + cfg["cifa_consistency_coef"] * cons)
        else:
            loss = loss_c

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % steps_per_epoch == 0:
            epoch += 1
            if cfg["swad"]:
                val_loss, _ = evaluate(model, merged_holdout, cfg, device)
                start, end = valley.update(epoch, val_loss)
                if valley.need_average(epoch):
                    if averaged is None:
                        averaged = AveragedModel(model)
                    averaged.update(model)
                    checkpoints.append((epoch, copy.deepcopy(model.state_dict())))
                model.train()
            elif method == "erm":
                if epoch > cfg["steps"] // steps_per_epoch - 15:
                    checkpoints.append((epoch, copy.deepcopy(model.state_dict())))

    if cfg["swad"] and averaged is not None and averaged.n > 0:
        model.load_state_dict(averaged.state_dict())

    print(f"[{method}] Saved {len(checkpoints)} checkpoints, "
          f"SWAD averaged {averaged.n if averaged else 0} checkpoints", flush=True)

    return model, tracker, checkpoints, data, cfg, device


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="pacs")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--target_domain", default="sketch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n_aug", type=int, default=20)
    p.add_argument("--output_dir", default="feature_alignment_results")
    args = p.parse_args()

    spec = DATASETS[args.dataset]
    if args.data_dir is None:
        args.data_dir = spec["default_dir"]

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Train CIFA
    print("=" * 60, flush=True)
    print("Training CIFA...", flush=True)
    print("=" * 60, flush=True)
    cifa_model, cifa_tracker, cifa_ckpts, data, cfg, device = train_and_save_checkpoints(
        args, method="cifa")

    # 2. Train ERM
    print("\n" + "=" * 60, flush=True)
    print("Training ERM control...", flush=True)
    print("=" * 60, flush=True)
    erm_model, _, erm_ckpts, _, _, _ = train_and_save_checkpoints(args, method="erm")

    val_dataset = ConcatDataset(list(data["holdout_by_domain"].values()))
    print(f"\nValidation set size: {len(val_dataset)}", flush=True)

    layers = ("layer1", "layer2")

    # 3. Extract SWAD checkpoint moments
    print("\n" + "=" * 60, flush=True)
    print("Extracting SWAD checkpoint feature moments...", flush=True)
    print("=" * 60, flush=True)
    swad_moments = {name: [] for name in layers}
    for epoch, sd in cifa_ckpts:
        cifa_model.load_state_dict(sd)
        moments = extract_instance_moments(
            cifa_model, val_dataset, cfg, device, layers=layers)
        for name in layers:
            swad_moments[name].append(moments[name])
    swad_moments = {name: torch.stack(parts, dim=0) for name, parts in swad_moments.items()}
    for name in layers:
        print(f"  {name}: {swad_moments[name].shape}", flush=True)

    # Restore averaged model
    avg_sd = copy.deepcopy(cifa_ckpts[0][1])
    for name in avg_sd:
        avg_sd[name] = sum(sd[name] for _, sd in cifa_ckpts) / len(cifa_ckpts)
    cifa_model.load_state_dict(avg_sd)

    # 4. Extract augmentation moments
    print("\n" + "=" * 60, flush=True)
    print(f"Extracting {args.n_aug} augmentation view moments...", flush=True)
    print("=" * 60, flush=True)
    aug_moments = extract_moments_multi_aug(
        cifa_model, val_dataset, cfg, device, args.n_aug,
        layers=layers, tracker=cifa_tracker)
    for name in layers:
        print(f"  {name}: {aug_moments[name].shape}", flush=True)

    # 5. Extract ERM checkpoint moments
    print("\n" + "=" * 60, flush=True)
    print("Extracting ERM checkpoint feature moments...", flush=True)
    print("=" * 60, flush=True)
    erm_moments = {name: [] for name in layers}
    for epoch, sd in erm_ckpts:
        erm_model.load_state_dict(sd)
        moments = extract_instance_moments(
            erm_model, val_dataset, cfg, device, layers=layers)
        for name in layers:
            erm_moments[name].append(moments[name])
    erm_moments = {name: torch.stack(parts, dim=0) for name, parts in erm_moments.items()}
    for name in layers:
        print(f"  {name}: {erm_moments[name].shape}", flush=True)

    # 6. Compute feature alignment metrics
    print("\n" + "=" * 60, flush=True)
    print("Computing feature alignment metrics...", flush=True)
    print("=" * 60, flush=True)

    results = {
        "dataset": args.dataset,
        "target_domain": args.target_domain,
        "seed": args.seed,
        "n_swad_ckpts": len(cifa_ckpts),
        "n_erm_ckpts": len(erm_ckpts),
        "n_aug": args.n_aug,
        "swad_epochs": [e for e, _ in cifa_ckpts],
        "erm_epochs": [e for e, _ in erm_ckpts],
    }

    for layer in layers:
        n_ch = RESNET_CHANNELS[cfg.get("backbone", "resnet50")][layer]
        print(f"\n--- {layer} (channels={n_ch}) ---", flush=True)

        # NN ratio
        nn_cross_sa, nn_w_s, nn_w_a, ratio_sa = feature_nn_ratio(
            swad_moments[layer], aug_moments[layer])
        nn_cross_ea, nn_w_e, nn_w_a2, ratio_ea = feature_nn_ratio(
            erm_moments[layer], aug_moments[layer])
        print(f"  NN ratio SWAD-aug: {ratio_sa:.4f}  (cross={nn_cross_sa:.6f}, "
              f"within-S={nn_w_s:.6f}, within-A={nn_w_a:.6f})", flush=True)
        print(f"  NN ratio ERM-aug:  {ratio_ea:.4f}  (cross={nn_cross_ea:.6f}, "
              f"within-E={nn_w_e:.6f}, within-A={nn_w_a2:.6f})", flush=True)
        results[f"{layer}_nn_ratio_swad"] = ratio_sa
        results[f"{layer}_nn_ratio_erm"] = ratio_ea

        # Gaussian moment KL
        kl_sa = feature_moment_kl(swad_moments[layer], aug_moments[layer], n_ch)
        kl_ea = feature_moment_kl(erm_moments[layer], aug_moments[layer], n_ch)
        print(f"  Moment KL SWAD-aug: {kl_sa:.6f}", flush=True)
        print(f"  Moment KL ERM-aug:  {kl_ea:.6f}", flush=True)
        results[f"{layer}_kl_swad"] = kl_sa
        results[f"{layer}_kl_erm"] = kl_ea

        # CKA
        cross_cka, w_cka_s, w_cka_a = feature_cka_comparison(
            swad_moments[layer], aug_moments[layer])
        cross_cka_e, w_cka_e, w_cka_a2 = feature_cka_comparison(
            erm_moments[layer], aug_moments[layer])
        print(f"  CKA cross SWAD-aug: {cross_cka:.4f}  (within-S={w_cka_s:.4f}, "
              f"within-A={w_cka_a:.4f})", flush=True)
        print(f"  CKA cross ERM-aug:  {cross_cka_e:.4f}  (within-E={w_cka_e:.4f}, "
              f"within-A={w_cka_a2:.4f})", flush=True)
        results[f"{layer}_cka_swad"] = cross_cka
        results[f"{layer}_cka_erm"] = cross_cka_e
        results[f"{layer}_cka_within_swad"] = w_cka_s
        results[f"{layer}_cka_within_aug"] = w_cka_a
        results[f"{layer}_cka_within_erm"] = w_cka_e

    # 7. Save
    out_path = os.path.join(
        args.output_dir,
        f"feature_alignment_{args.dataset}_{args.target_domain}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # 8. Summary
    print("\n" + "=" * 60, flush=True)
    print("FEATURE ALIGNMENT SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for layer in layers:
        print(f"\n{layer}:", flush=True)
        print(f"  NN ratio:  SWAD {results[f'{layer}_nn_ratio_swad']:.4f}  vs  "
              f"ERM {results[f'{layer}_nn_ratio_erm']:.4f}", flush=True)
        print(f"  Moment KL: SWAD {results[f'{layer}_kl_swad']:.6f}  vs  "
              f"ERM {results[f'{layer}_kl_erm']:.6f}", flush=True)
        print(f"  CKA:       SWAD {results[f'{layer}_cka_swad']:.4f}  vs  "
              f"ERM {results[f'{layer}_cka_erm']:.4f}", flush=True)


if __name__ == "__main__":
    main()
