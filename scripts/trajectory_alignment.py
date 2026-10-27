"""
R3-(4) Trajectory-Augmentation Alignment Verification.

Hypothesis (Assumption 3): SWAD-window checkpoints' prediction distribution
is aligned with CIFA augmentation-induced prediction distribution.

Experiment:
1. Train CIFA on PACS (target=sketch, seed=0), save every SWAD-window checkpoint.
2. Train ERM (no CIFA, no SWAD) as control, save late-training checkpoints.
3. Compute prediction distributions on a fixed validation batch:
   - Set A: each SWAD checkpoint -> clean forward (N_ckpt predictions)
   - Set B: final CIFA model -> multiple CIFA augmentations (M_aug predictions)
   - Control: ERM checkpoints -> clean forward
4. Alignment metrics:
   - Covariance matrix similarity (correlation between off-diagonal entries)
   - Mean prediction KL divergence
   - Prediction entropy distribution similarity
   - Cross-set nearest-neighbor distance ratio
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
from cifa_aug import BaselineTracker
from swad import AveragedModel, LossValley


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def get_predictions(model, dataset, cfg, device, batch_size=128, augment=False, tracker=None, n_aug=1):
    """Get prediction logits for all samples in dataset.
    If augment=True and tracker is provided, run n_aug augmentations per sample.
    Returns:
      if augment: tensor [n_aug, N, num_classes]
      else: tensor [N, num_classes]
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    model.eval()
    all_logits = []

    if augment:
        # Augmentation only runs in training mode (see networks.py: self.training check)
        # BN is frozen by freeze_bn, so train() only affects dropout (none in ResNet)
        model.train()
        import random as pyrandom
        aug_logits = [[] for _ in range(n_aug)]
        for i in range(n_aug):
            # Set both torch and python random seeds for deterministic augmentation
            torch.manual_seed(20000 + i)
            pyrandom.seed(30000 + i)
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits_c, logits_a = model(x, y, tracker, cfg)
                logits = logits_a if logits_a is not None else logits_c
                aug_logits[i].append(logits.cpu())
        model.eval()
        result = []
        for i in range(n_aug):
            result.append(torch.cat(aug_logits[i], dim=0))
        return torch.stack(result, dim=0)  # [n_aug, N, C]
    else:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _, _ = model.forward_path(x, y, None, augment=False, cfg=cfg)
            all_logits.append(logits.cpu())
        return torch.cat(all_logits, dim=0)  # [N, C]


def samplewise_variance_correlation(logits_set1, logits_set2):
    """Correlation of per-sample prediction variance between two sets.
    For each sample, compute variance of predicted probability across distributions in the set.
    Then correlate the two per-sample variance vectors.
    This measures whether the two sets 'disagree on the same samples' — a key alignment signal.
    """
    probs1 = F.softmax(logits_set1, dim=-1)  # [K1, N, C]
    probs2 = F.softmax(logits_set2, dim=-1)  # [K2, N, C]
    # Variance across distributions for each sample, averaged over classes -> [N]
    var1 = probs1.var(dim=0).mean(dim=-1)  # [N]
    var2 = probs2.var(dim=0).mean(dim=-1)  # [N]
    # Pearson correlation
    v1_c = var1 - var1.mean()
    v2_c = var2 - var2.mean()
    pearson = (v1_c * v2_c).sum() / (v1_c.norm() * v2_c.norm() + 1e-10)
    # Also compute ratio of mean variance (discreteness comparison)
    mean_var_ratio = var1.mean().item() / (var2.mean().item() + 1e-10)
    return pearson.item(), var1.mean().item(), var2.mean().item(), mean_var_ratio


def within_set_dispersion(logits_set):
    """Mean pairwise KL within a set — measures how spread out the predictions are."""
    probs = F.softmax(logits_set, dim=-1)  # [K, N, C]
    K = probs.shape[0]
    if K < 2:
        return 0.0
    # Pairwise KL
    p_exp = probs.unsqueeze(1)  # [K, 1, N, C]
    q_exp = probs.unsqueeze(0)  # [1, K, N, C]
    kl = (p_exp * (torch.log(p_exp + 1e-10) - torch.log(q_exp + 1e-10))).sum(dim=-1).mean(dim=-1)
    # Upper triangle mean
    idx = torch.triu_indices(K, K, offset=1)
    return kl[idx[0], idx[1]].mean().item()


def mean_prediction_kl(logits_set1, logits_set2):
    """KL divergence between mean predictions of two sets."""
    probs1 = F.softmax(logits_set1, dim=-1).mean(dim=0)  # [N, C]
    probs2 = F.softmax(logits_set2, dim=-1).mean(dim=0)  # [N, C]
    kl = (probs1 * (torch.log(probs1 + 1e-10) - torch.log(probs2 + 1e-10))).sum(dim=-1).mean()
    return kl.item()


def entropy_distribution_similarity(logits_set1, logits_set2):
    """Compare entropy distributions of two sets."""
    probs1 = F.softmax(logits_set1, dim=-1)  # [K1, N, C]
    probs2 = F.softmax(logits_set2, dim=-1)  # [K2, N, C]
    ent1 = -(probs1 * torch.log(probs1 + 1e-10)).sum(dim=-1)  # [K1, N]
    ent2 = -(probs2 * torch.log(probs2 + 1e-10)).sum(dim=-1)  # [K2, N]
    # Mean and std of entropy across samples
    m1, s1 = ent1.mean().item(), ent1.std().item()
    m2, s2 = ent2.mean().item(), ent2.std().item()
    # KS-like: difference in mean entropy normalized by pooled std
    pooled_std = np.sqrt((s1**2 + s2**2) / 2) + 1e-10
    diff = abs(m1 - m2) / pooled_std
    return m1, s1, m2, s2, diff


def cross_set_nearest_neighbor(logits_set1, logits_set2):
    """For each distribution in set1, find nearest in set2 (by KL).
    Returns mean nearest-neighbor distance and ratio to within-set distance.
    """
    probs1 = F.softmax(logits_set1, dim=-1)  # [K1, N, C]
    probs2 = F.softmax(logits_set2, dim=-1)  # [K2, N, C]
    K1, K2 = probs1.shape[0], probs2.shape[0]

    # Compute pairwise KL between all distributions
    def kl_matrix(p, q):
        # p: [K1, N, C], q: [K2, N, C] -> [K1, K2]
        p_exp = p.unsqueeze(1)  # [K1, 1, N, C]
        q_exp = q.unsqueeze(0)  # [1, K2, N, C]
        kl = (p_exp * (torch.log(p_exp + 1e-10) - torch.log(q_exp + 1e-10))).sum(dim=-1).mean(dim=-1)
        return kl  # [K1, K2]

    cross_kl = kl_matrix(probs1, probs2)  # [K1, K2]
    nn_cross = cross_kl.min(dim=1).values.mean().item()

    # Within-set distances
    within1 = kl_matrix(probs1, probs1)
    within1.fill_diagonal_(float('inf'))
    nn_within1 = within1.min(dim=1).values.mean().item()

    within2 = kl_matrix(probs2, probs2)
    within2.fill_diagonal_(float('inf'))
    nn_within2 = within2.min(dim=1).values.mean().item()

    ratio = nn_cross / ((nn_within1 + nn_within2) / 2 + 1e-10)
    return nn_cross, nn_within1, nn_within2, ratio


def train_and_save_checkpoints(args, method="cifa"):
    """Train model and save SWAD-window (or late-training) checkpoints.
    Returns list of checkpoint state_dicts and final model + tracker.
    """
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
    tracker = BaselineTracker(num_classes, channels, momentum=cfg["cifa_baseline_momentum"], device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    valley = LossValley(
        n_converge=cfg["swad_n_converge"],
        n_tolerance=cfg["swad_n_tolerance"],
        tolerance_ratio=cfg["swad_tolerance_ratio"],
        tolerance_epoch=cfg["swad_tolerance_epoch"]) if cfg["swad"] else None
    averaged = None

    merged_holdout = ConcatDataset(list(data["holdout_by_domain"].values()))

    checkpoints = []  # list of (epoch, state_dict)
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
                    # Save this checkpoint
                    checkpoints.append((epoch, copy.deepcopy(model.state_dict())))
                model.train()
            elif method == "erm":
                # For ERM control: save last N epochs as checkpoints
                if epoch > cfg["steps"] // steps_per_epoch - 15:
                    checkpoints.append((epoch, copy.deepcopy(model.state_dict())))

    # Final model
    if cfg["swad"] and averaged is not None and averaged.n > 0:
        model.load_state_dict(averaged.state_dict())

    print(f"[{method}] Saved {len(checkpoints)} checkpoints, "
          f"SWAD averaged {averaged.n if averaged else 0} checkpoints", flush=True)

    return model, tracker, checkpoints, data, cfg, device


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="pacs")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--target_domain", default="sketch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n_aug", type=int, default=20, help="number of CIFA augmentation views")
    p.add_argument("--output_dir", default="trajectory_alignment_results")
    args = p.parse_args()

    spec = DATASETS[args.dataset]
    if args.data_dir is None:
        args.data_dir = spec["default_dir"]

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 1. Train CIFA and save SWAD checkpoints ==========
    print("=" * 60, flush=True)
    print("Training CIFA (saving SWAD window checkpoints)...", flush=True)
    print("=" * 60, flush=True)
    cifa_model, cifa_tracker, cifa_ckpts, data, cfg, device = train_and_save_checkpoints(args, method="cifa")

    # ========== 2. Train ERM control and save late checkpoints ==========
    print("\n" + "=" * 60, flush=True)
    print("Training ERM control (saving late checkpoints)...", flush=True)
    print("=" * 60, flush=True)
    erm_model, _, erm_ckpts, _, _, _ = train_and_save_checkpoints(args, method="erm")

    # ========== 3. Collect predictions ==========
    print("\n" + "=" * 60, flush=True)
    print("Collecting predictions...", flush=True)
    print("=" * 60, flush=True)

    # Use merged holdout as the fixed validation set
    val_dataset = ConcatDataset(list(data["holdout_by_domain"].values()))
    print(f"Validation set size: {len(val_dataset)}", flush=True)

    # Set A: SWAD checkpoint predictions (clean forward)
    print(f"Computing {len(cifa_ckpts)} SWAD checkpoint predictions...", flush=True)
    swad_preds = []
    for epoch, sd in cifa_ckpts:
        cifa_model.load_state_dict(sd)
        preds = get_predictions(cifa_model, val_dataset, cfg, device, augment=False)
        swad_preds.append(preds)
    swad_preds = torch.stack(swad_preds, dim=0)  # [N_ckpt, N, C]
    print(f"  SWAD preds shape: {swad_preds.shape}", flush=True)

    # Restore final averaged model
    # Re-train average from checkpoints
    avg_sd = copy.deepcopy(cifa_ckpts[0][1])
    for name in avg_sd:
        avg_sd[name] = sum(sd[name] for _, sd in cifa_ckpts) / len(cifa_ckpts)
    cifa_model.load_state_dict(avg_sd)

    # Set B: CIFA augmentation predictions (final model, multiple augmentations)
    print(f"Computing {args.n_aug} CIFA augmentation predictions...", flush=True)
    cifa_aug_preds = get_predictions(cifa_model, val_dataset, cfg, device,
                                      augment=True, tracker=cifa_tracker, n_aug=args.n_aug)
    print(f"  CIFA aug preds shape: {cifa_aug_preds.shape}", flush=True)

    # Set C: ERM checkpoint predictions (control)
    print(f"Computing {len(erm_ckpts)} ERM checkpoint predictions...", flush=True)
    erm_preds = []
    for epoch, sd in erm_ckpts:
        erm_model.load_state_dict(sd)
        preds = get_predictions(erm_model, val_dataset, cfg, device, augment=False)
        erm_preds.append(preds)
    erm_preds = torch.stack(erm_preds, dim=0)  # [N_ckpt, N, C]
    print(f"  ERM preds shape: {erm_preds.shape}", flush=True)

    # ========== 4. Alignment metrics ==========
    print("\n" + "=" * 60, flush=True)
    print("Computing alignment metrics...", flush=True)
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

    # Metric 1: Sample-wise prediction variance correlation
    print("\n--- Sample-wise Variance Correlation ---", flush=True)
    pearson_sa, var_s, var_a, ratio_sa = samplewise_variance_correlation(swad_preds, cifa_aug_preds)
    pearson_ea, var_e, var_a2, ratio_ea = samplewise_variance_correlation(erm_preds, cifa_aug_preds)
    print(f"  SWAD vs CIFA-aug: Pearson={pearson_sa:.4f}, mean_var_SWAD={var_s:.6f}, mean_var_aug={var_a:.6f}, ratio={ratio_sa:.4f}", flush=True)
    print(f"  ERM  vs CIFA-aug: Pearson={pearson_ea:.4f}, mean_var_ERM={var_e:.6f}, mean_var_aug={var_a2:.6f}, ratio={ratio_ea:.4f}", flush=True)
    results["var_corr_swad_aug"] = pearson_sa
    results["var_corr_erm_aug"] = pearson_ea
    results["mean_var_swad"] = var_s
    results["mean_var_aug"] = var_a
    results["mean_var_erm"] = var_e

    # Metric 1b: Within-set dispersion (mean pairwise KL)
    print("\n--- Within-Set Dispersion (mean pairwise KL) ---", flush=True)
    disp_swad = within_set_dispersion(swad_preds)
    disp_aug = within_set_dispersion(cifa_aug_preds)
    disp_erm = within_set_dispersion(erm_preds)
    print(f"  SWAD: {disp_swad:.6f}, CIFA-aug: {disp_aug:.6f}, ERM: {disp_erm:.6f}", flush=True)
    print(f"  SWAD/aug ratio: {disp_swad/(disp_aug+1e-10):.4f}, ERM/aug ratio: {disp_erm/(disp_aug+1e-10):.4f}", flush=True)
    results["dispersion_swad"] = disp_swad
    results["dispersion_aug"] = disp_aug
    results["dispersion_erm"] = disp_erm

    # Metric 2: Mean prediction KL
    print("\n--- Mean Prediction KL ---", flush=True)
    kl_swad_aug = mean_prediction_kl(swad_preds, cifa_aug_preds)
    kl_erm_aug = mean_prediction_kl(erm_preds, cifa_aug_preds)
    kl_swad_erm = mean_prediction_kl(swad_preds, erm_preds)
    print(f"  SWAD vs CIFA-aug: {kl_swad_aug:.6f} nats", flush=True)
    print(f"  ERM  vs CIFA-aug: {kl_erm_aug:.6f} nats", flush=True)
    print(f"  SWAD vs ERM:      {kl_swad_erm:.6f} nats", flush=True)
    results["kl_swad_aug"] = kl_swad_aug
    results["kl_erm_aug"] = kl_erm_aug
    results["kl_swad_erm"] = kl_swad_erm

    # Metric 3: Entropy distribution
    print("\n--- Entropy Distribution ---", flush=True)
    m_s, s_s, m_a, s_a, diff_sa = entropy_distribution_similarity(swad_preds, cifa_aug_preds)
    m_e, s_e, m_a2, s_a2, diff_ea = entropy_distribution_similarity(erm_preds, cifa_aug_preds)
    print(f"  SWAD:  mean={m_s:.4f}, std={s_s:.4f}", flush=True)
    print(f"  CIFA-aug: mean={m_a:.4f}, std={s_a:.4f}", flush=True)
    print(f"  ERM:   mean={m_e:.4f}, std={s_e:.4f}", flush=True)
    print(f"  SWAD vs aug normalized diff: {diff_sa:.4f}", flush=True)
    print(f"  ERM  vs aug normalized diff: {diff_ea:.4f}", flush=True)
    results["entropy_swad_mean"] = m_s
    results["entropy_swad_std"] = s_s
    results["entropy_aug_mean"] = m_a
    results["entropy_aug_std"] = s_a
    results["entropy_erm_mean"] = m_e
    results["entropy_erm_std"] = s_e
    results["entropy_diff_swad_aug"] = diff_sa
    results["entropy_diff_erm_aug"] = diff_ea

    # Metric 4: Cross-set nearest neighbor
    print("\n--- Cross-Set Nearest Neighbor ---", flush=True)
    nn_cross_sa, nn_within_s, nn_within_a, ratio_sa = cross_set_nearest_neighbor(swad_preds, cifa_aug_preds)
    nn_cross_ea, nn_within_e, nn_within_a2, ratio_ea = cross_set_nearest_neighbor(erm_preds, cifa_aug_preds)
    print(f"  SWAD vs CIFA-aug: cross-NN={nn_cross_sa:.6f}, within-SWAD={nn_within_s:.6f}, within-aug={nn_within_a:.6f}, ratio={ratio_sa:.4f}", flush=True)
    print(f"  ERM  vs CIFA-aug: cross-NN={nn_cross_ea:.6f}, within-ERM={nn_within_e:.6f}, within-aug={nn_within_a2:.6f}, ratio={ratio_ea:.4f}", flush=True)
    results["nn_cross_swad_aug"] = nn_cross_sa
    results["nn_within_swad"] = nn_within_s
    results["nn_within_aug"] = nn_within_a
    results["nn_ratio_swad_aug"] = ratio_sa
    results["nn_cross_erm_aug"] = nn_cross_ea
    results["nn_within_erm"] = nn_within_e
    results["nn_ratio_erm_aug"] = ratio_ea

    # ========== 5. Save results ==========
    out_path = os.path.join(args.output_dir, f"alignment_{args.dataset}_{args.target_domain}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # ========== 6. Summary ==========
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY: Trajectory-Augmentation Alignment", flush=True)
    print("=" * 60, flush=True)
    print(f"  Variance correlation (higher=more aligned):", flush=True)
    print(f"    SWAD vs aug: {pearson_sa:.4f}  vs  ERM vs aug: {pearson_ea:.4f}", flush=True)
    print(f"  Mean prediction KL (lower=more aligned):", flush=True)
    print(f"    SWAD vs aug: {kl_swad_aug:.6f}  vs  ERM vs aug: {kl_erm_aug:.6f}", flush=True)
    print(f"  Cross-set NN ratio (lower=more aligned):", flush=True)
    print(f"    SWAD vs aug: {ratio_sa:.4f}  vs  ERM vs aug: {ratio_ea:.4f}", flush=True)
    print(f"  Within-set dispersion (closer to aug=more aligned):", flush=True)
    print(f"    SWAD: {disp_swad:.6f}  ERM: {disp_erm:.6f}  aug: {disp_aug:.6f}", flush=True)
    print(f"\n  Interpretation: If SWAD-aug metrics are substantially more aligned", flush=True)
    print(f"  than ERM-aug, Assumption 3 is empirically supported.", flush=True)


if __name__ == "__main__":
    main()
