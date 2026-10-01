"""Train CIFA under the DomainBed leave-one-domain-out protocol.

Examples (on your GPU box):
    # PACS (default dataset)
    python train.py --target_domain art_painting --seed 0
    # VLCS
    python train.py --dataset vlcs \
        --data_dir /mnt/sdd1/datasets_qsong/VLCS/VLCS \
        --target_domain CALTECH --seed 0
Ablations:
    --no_cifa            # SWAD only
    --no_cifa --no_swad  # ERM
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from config import RESNET_CHANNELS, hparams_registry, merge_overrides
from datasets import DATASETS, get_dataset, infinite_loader
from networks import CIFAResNet
from cifa_aug import BaselineTracker
from swad import AveragedModel, LossValley

# Variant name -> the single design choice it changes (everything else frozen).
VARIANT_OVERRIDES = {
    "full": {},
    "fullmoment": {"cifa_decompose": False},   # mix full moments (MixStyle-style)
    "noguard": {"cifa_use_guard": False},      # gate=1, no consistency penalty
    "uniform": {"cifa_aggregation": "uniform"},  # uniform instead of soft worst-case
    "noswad": {"swad": False},                 # disable weight averaging
}
VARIANTS = ["full", "fullmoment", "noguard", "uniform", "noswad"]


def combine_views(loss_c, logits_c, logits_a, y, cfg):
    """Combine the clean and augmented views; returns (loss, gate, cons).

    Honours the ablation switches: cifa_aggregation ('uniform') and
    cifa_use_guard (False -> gate fixed at 1, no KL penalty).
    """
    loss_a = F.cross_entropy(logits_a, y)
    with torch.no_grad():
        p_c = F.softmax(logits_c, 1)
        logp_c = F.log_softmax(logits_c, 1)
        logp_a = F.log_softmax(logits_a, 1)
        cons = (p_c * (logp_c - logp_a)).sum(1).mean()

    if cfg["cifa_aggregation"] == "uniform":
        q = torch.tensor([0.5, 0.5], device=logits_c.device)
    else:
        risks = torch.stack([loss_c.detach(), loss_a.detach()])
        q = F.softmax(cfg["cifa_soft_eta"] * risks, dim=0)

    if cfg["cifa_use_guard"]:
        gate = (cfg["cifa_consistency_delta"] /
                (cons.detach() + 1e-8)).clamp(max=1.0)
        loss = q[0] * loss_c + gate * (
            q[1] * loss_a + cfg["cifa_consistency_coef"] * cons)
    else:
        gate = torch.ones((), device=logits_c.device)
        loss = q[0] * loss_c + q[1] * loss_a
    return loss, gate, cons


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, dataset, cfg, device, batch_size=64):
    """Return (mean loss, accuracy) using the clean forward path."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _, _ = model.forward_path(x, y, None, augment=False, cfg=cfg)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n


def parse_hparam_strings(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = v
    return out


def load_config(path):
    """Load a YAML config file; returns an empty dict if it is empty."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def apply_typed(hparams, overrides, source="config"):
    """Apply already-typed overrides (as parsed from YAML).

    No string coercion is performed, unlike merge_overrides. Unknown keys are
    rejected so a typo in a config fails loudly instead of being ignored.
    """
    for k, v in (overrides or {}).items():
        if k not in hparams:
            raise KeyError(f"unknown hparam '{k}' in {source}")
        hparams[k] = v
    return hparams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="pacs", choices=sorted(DATASETS))
    p.add_argument("--data_dir", default=None,
                   help="defaults to the registry path of --dataset")
    p.add_argument("--target_domain", default=None,
                   help="held-out domain folder; defaults to the first domain")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no_cifa", action="store_true")
    p.add_argument("--no_swad", action="store_true")
    p.add_argument("--variant", default="full", choices=VARIANTS,
                   help="component-ablation variant; sets the corresponding switch")
    p.add_argument("--hparam", action="append", default=[],
                   help="override, e.g. --hparam cifa_prob=0.8")
    p.add_argument("--config", default=None,
                   help="YAML config (configs/*.yaml); CLI --hparam overrides it")
    args = p.parse_args()

    spec = DATASETS[args.dataset]
    file_conf = load_config(args.config) if args.config else {}
    if args.data_dir is None:
        args.data_dir = file_conf.get("data_dir") or spec["default_dir"]
    if args.target_domain is None:
        args.target_domain = spec["domains"][0]
    if args.target_domain not in spec["domains"]:
        raise SystemExit(
            f"target_domain must be one of {spec['domains']} for dataset "
            f"{args.dataset} (got {args.target_domain})")

    # Layered hparams: registry defaults < YAML config < command-line --hparam.
    cfg = apply_typed(hparams_registry(), file_conf.get("hparams", {}),
                      source=args.config or "config")
    cfg = merge_overrides(cfg, parse_hparam_strings(args.hparam))
    if args.no_cifa:
        cfg["cifa_enabled"] = False
    if args.no_swad:
        cfg["swad"] = False
    # Variant is authoritative: it pins the single ablated design choice.
    cfg.update(VARIANT_OVERRIDES[args.variant])

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = get_dataset(args.dataset, args.data_dir, args.target_domain,
                       holdout_fraction=cfg["holdout_fraction"], seed=args.seed)
    num_classes = len(data["classes"])

    train_loader = DataLoader(
        data["train_dataset"], batch_size=cfg["batch_size"], shuffle=True,
        drop_last=True, num_workers=args.workers,
        pin_memory=torch.cuda.is_available())
    steps_per_epoch = max(1, len(data["train_dataset"]) // cfg["batch_size"])

    model = CIFAResNet(
        backbone=cfg["backbone"], pretrained=cfg["pretrained"],
        num_classes=num_classes, layers=cfg["cifa_layers"],
        freeze_bn=cfg["freeze_bn"]).to(device)

    channels = {n: RESNET_CHANNELS[cfg["backbone"]][n]
                for n in cfg["cifa_layers"]}
    tracker = BaselineTracker(
        num_classes, channels,
        momentum=cfg["cifa_baseline_momentum"], device=device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    valley = LossValley(
        n_converge=cfg["swad_n_converge"],
        n_tolerance=cfg["swad_n_tolerance"],
        tolerance_ratio=cfg["swad_tolerance_ratio"],
        tolerance_epoch=cfg["swad_tolerance_epoch"]) if cfg["swad"] else None
    averaged = None

    merged_holdout = ConcatDataset(list(data["holdout_by_domain"].values()))

    # diagnostics: verify the augmentation is not silently gated away
    diag = {"n_aug": 0, "cons_sum": 0.0, "gate_sum": 0.0}

    model.train()
    loader_it = infinite_loader(train_loader)
    epoch = 0
    for step in range(1, cfg["steps"] + 1):
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
            print(f"[step {step}/{cfg['steps']}] loss={loss.item():.4f}",
                  flush=True)

        # ---- end of epoch: hold-out validation + SWAD ----
        if step % steps_per_epoch == 0:
            epoch += 1
            if cfg["swad"]:
                val_loss, _ = evaluate(model, merged_holdout, cfg, device)
                start, end = valley.update(epoch, val_loss)
                if valley.need_average(epoch):
                    if averaged is None:
                        averaged = AveragedModel(model)
                    averaged.update(model)
                model.train()

    # ---- final model selection: SWAD average if available ----
    if cfg["swad"] and averaged is not None and averaged.n > 0:
        model.load_state_dict(averaged.state_dict())
        print(f"SWAD averaged {averaged.n} epoch checkpoints", flush=True)

    # worst-domain hold-out accuracy (no target involved)
    holdout_acc = {}
    for dom, ds in data["holdout_by_domain"].items():
        _, acc = evaluate(model, ds, cfg, device)
        holdout_acc[dom] = acc
    worst_holdout = min(holdout_acc.values())

    target_loss, target_acc = evaluate(model, data["target"], cfg, device)
    print(f"target={args.target_domain} seed={args.seed} "
          f"acc={target_acc*100:.2f} worst_holdout={worst_holdout*100:.2f}",
          flush=True)

    method = "cifa" if cfg["cifa_enabled"] else (
        "swad" if cfg["swad"] else "erm")
    if diag["n_aug"] > 0:
        mean_cons = diag["cons_sum"] / diag["n_aug"]
        mean_gate = diag["gate_sum"] / diag["n_aug"]
        print(f"[diag] aug triggered {diag['n_aug']}/{cfg['steps']} steps "
              f"({diag['n_aug']/cfg['steps']:.0%}), "
              f"mean KL(cons)={mean_cons:.3f} nats, "
              f"mean gate={mean_gate:.3f}", flush=True)
    else:
        mean_cons = mean_gate = None

    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        "dataset": args.dataset,
        "method": method,
        "variant": args.variant,
        "target_domain": args.target_domain,
        "seed": args.seed,
        "target_acc": target_acc,
        "target_loss": target_loss,
        "holdout_acc": holdout_acc,
        "worst_holdout_acc": worst_holdout,
        "swad_n_checkpoints": averaged.n if averaged is not None else 0,
        "aug_trigger_fraction": diag["n_aug"] / cfg["steps"],
        "aug_mean_consistency_kl": mean_cons,
        "aug_mean_gate": mean_gate,
        "hparams": cfg,
    }
    fn = os.path.join(args.output_dir,
                      f"{args.variant}_{args.target_domain}_seed{args.seed}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", fn)


if __name__ == "__main__":
    main()
