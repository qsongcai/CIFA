"""A3 probe: collect the predictions needed to test Assumption 3.

Assumption 3 (trajectory--augmentation alignment) states that, within the
flat parameter neighbourhood selected by the dense-averaging window, the
empirical distribution of trajectory checkpoints approximates, for risk
comparisons, the distribution of predictors induced by the augmentation
family. This script trains one CIFA run and records, on the held-out target
domain, three prediction banks:

  TE  (trajectory ensemble)        : clean predictions of every checkpoint
                                     that enters the SWAD averaging window;
  hbar (dense weight average)      : clean predictions of the final model;
  AE  (augmentation-family ens.)   : M independent random-augmentation
                                     predictions of the final hbar model.

Penultimate features are stored alongside the probabilities for linear CKA.

Protocol note
-------------
The averaging window is selected from source-domain hold-out loss only
(LossValley); the target domain is never used to select checkpoints or
hyperparameters. The target is evaluated multiple times purely as a
post-hoc mechanism analysis (same status as the Table X diagnostics), not
for model selection, so this does not leak target information. The probe
writes to a separate directory (results/a3) and does not alter the main
training pipeline.

Example (GPU box):
    python a3_probe.py --dataset pacs --target_domain sketch --seed 0 --a3_m 32
CPU self-check (no data, no pretrained):
    python a3_probe.py --smoke
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import RESNET_CHANNELS, hparams_registry, merge_overrides
from datasets import DATASETS, get_dataset, infinite_loader
from networks import CIFAResNet
from cifa_aug import BaselineTracker
from swad import AveragedModel, LossValley
from train import (apply_typed, combine_views, evaluate, load_config,
                   parse_hparam_strings, set_seed, VARIANT_OVERRIDES)


@torch.no_grad()
def predict_bank(model, dataset, cfg, device, augment, tracker,
                 batch_size=64):
    """Return (probs[N,C], feat[N,D]) over `dataset`.

    augment=False -> clean path; augment=True -> one random CIFA view per
    image (requires `tracker` for the content baseline). The tracker is not
    updated here (forward_path is called directly).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    ps, fs = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, feat, _ = model.forward_path(
            x, y, tracker, augment=augment, cfg=cfg)
        ps.append(F.softmax(logits, 1).cpu())
        fs.append(feat.cpu())
    return torch.cat(ps), torch.cat(fs)


def build_real(args, cfg):
    """Build device/data/model/tracker/optimizer for a real benchmark."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = DATASETS[args.dataset]
    if args.data_dir is None:
        args.data_dir = spec["default_dir"]
    data = get_dataset(args.dataset, args.data_dir, args.target_domain,
                       holdout_fraction=cfg["holdout_fraction"],
                       seed=args.seed)
    num_classes = len(data["classes"])
    model = CIFAResNet(
        backbone=cfg["backbone"], pretrained=cfg["pretrained"],
        num_classes=num_classes, layers=cfg["cifa_layers"],
        freeze_bn=cfg["freeze_bn"]).to(device)
    channels = {n: RESNET_CHANNELS[cfg["backbone"]][n]
                for n in cfg["cifa_layers"]}
    tracker = BaselineTracker(
        num_classes, channels,
        momentum=cfg["cifa_baseline_momentum"], device=device)
    loader = DataLoader(
        data["train_dataset"], batch_size=cfg["batch_size"], shuffle=True,
        drop_last=True, num_workers=args.workers,
        pin_memory=torch.cuda.is_available())
    spe = max(1, len(data["train_dataset"]) // cfg["batch_size"])
    return device, data, model, tracker, loader, spe


def build_smoke(cfg):
    """Tiny in-memory pipeline for the CPU self-check."""
    device = torch.device("cpu")
    K, Dimg, Ntr, Nt = 4, 40, 48, 16
    num_classes = K

    class _DS(TensorDataset):
        pass

    xtr = torch.randn(Ntr, 3, Dimg, Dimg)
    ytr = torch.randint(0, K, (Ntr,))
    train_ds = TensorDataset(xtr, ytr)
    target_ds = TensorDataset(torch.randn(Nt, 3, Dimg, Dimg),
                              torch.randint(0, K, (Nt,)))
    # hold-out per domain (two pseudo source domains)
    holdout_by_domain = {
        "h0": TensorDataset(torch.randn(8, 3, Dimg, Dimg),
                            torch.randint(0, K, (8,))),
        "h1": TensorDataset(torch.randn(8, 3, Dimg, Dimg),
                            torch.randint(0, K, (8,)))}
    data = {"train_dataset": train_ds, "target": target_ds,
            "holdout_by_domain": holdout_by_domain, "classes": list(range(K))}
    model = CIFAResNet("resnet18", pretrained=False, num_classes=K,
                       layers=("layer1", "layer2"),
                       freeze_bn=True).to(device)
    channels = {n: RESNET_CHANNELS["resnet18"][n] for n in ("layer1", "layer2")}
    tracker = BaselineTracker(K, channels, momentum=0.9, device=device)
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                        drop_last=True)
    return device, data, model, tracker, loader


def run(args):
    # ---- configuration (same layering as train.py) ----
    file_conf = load_config(args.config) if args.config else {}
    cfg = apply_typed(hparams_registry(), file_conf.get("hparams", {}),
                      source=args.config or "config")
    cfg = merge_overrides(cfg, parse_hparam_strings(args.hparam))
    if args.no_cifa:
        cfg["cifa_enabled"] = False
    if args.no_swad:
        cfg["swad"] = False
    cfg.update(VARIANT_OVERRIDES[args.variant])

    if args.smoke:
        cfg["backbone"] = "resnet18"
        cfg["pretrained"] = False
        cfg["steps"] = 24
        cfg["batch_size"] = 12
        cfg["swad_tolerance_epoch"] = 2
        cfg["swad_n_converge"] = 2
        cfg["swad_n_tolerance"] = 2
        cfg["cifa_prob"] = 1.0
        set_seed(0)
        device, data, model, tracker, loader = build_smoke(cfg)
        steps_per_epoch = 6
    else:
        set_seed(args.seed)
        device, data, model, tracker, loader, steps_per_epoch = build_real(
            args, cfg)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    valley = LossValley(
        n_converge=cfg["swad_n_converge"],
        n_tolerance=cfg["swad_n_tolerance"],
        tolerance_ratio=cfg["swad_tolerance_ratio"],
        tolerance_epoch=cfg["swad_tolerance_epoch"]) if cfg["swad"] else None
    averaged = None

    from torch.utils.data import ConcatDataset
    merged_holdout = ConcatDataset(list(data["holdout_by_domain"].values()))

    # TE banks: one (probs, feat) per checkpoint that enters the window
    te_probs, te_feat, te_epochs = [], [], []

    model.train()
    loader_it = infinite_loader(loader)
    epoch = 0
    for step in range(1, cfg["steps"] + 1):
        x, y = next(loader_it)
        x, y = x.to(device), y.to(device)

        logits_c, logits_a = model(x, y, tracker, cfg)
        loss_c = F.cross_entropy(logits_c, y)
        if logits_a is not None:
            loss, gate, cons = combine_views(loss_c, logits_c, logits_a,
                                             y, cfg)
        else:
            loss = loss_c

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- end of epoch: hold-out validation + SWAD + TE capture ----
        if step % steps_per_epoch == 0:
            epoch += 1
            if cfg["swad"]:
                val_loss, _ = evaluate(model, merged_holdout, cfg, device)
                start, end = valley.update(epoch, val_loss)
                if valley.need_average(epoch):
                    if averaged is None:
                        averaged = AveragedModel(model)
                    averaged.update(model)
                    # TE: this epoch's checkpoint (still in eval from evaluate)
                    p, ft = predict_bank(model, data["target"], cfg, device,
                                         augment=False, tracker=tracker)
                    te_probs.append(p)
                    te_feat.append(ft)
                    te_epochs.append(epoch)
                model.train()

    # ---- final hbar ----
    if cfg["swad"] and averaged is not None and averaged.n > 0:
        model.load_state_dict(averaged.state_dict())
        print(f"SWAD averaged {averaged.n} checkpoints; captured "
              f"{len(te_probs)} TE banks", flush=True)

    hbar_probs, hbar_feat = predict_bank(model, data["target"], cfg, device,
                                         augment=False, tracker=tracker)

    # ---- AE: M independent random augmentation views of hbar ----
    ae_probs, ae_feat = [], []
    for m in range(args.a3_m):
        p, ft = predict_bank(model, data["target"], cfg, device,
                             augment=True, tracker=tracker)
        ae_probs.append(p)
        ae_feat.append(ft)

    # labels (fixed order, shuffle=False)
    ybank = torch.cat([yy for _, yy in
                       DataLoader(data["target"], batch_size=64,
                                  shuffle=False)])

    out = {
        "te_probs": torch.stack(te_probs).numpy() if te_probs else
            np.zeros((0,)),
        "te_feat": torch.stack(te_feat).numpy() if te_feat else
            np.zeros((0,)),
        "te_epochs": np.array(te_epochs),
        "hbar_probs": hbar_probs.numpy(),
        "hbar_feat": hbar_feat.numpy(),
        "ae_probs": torch.stack(ae_probs).numpy(),
        "ae_feat": torch.stack(ae_feat).numpy(),
        "y": ybank.numpy(),
    }
    meta = {"dataset": args.dataset, "target_domain": args.target_domain,
            "seed": args.seed, "variant": args.variant,
            "a3_m": args.a3_m, "num_classes": hbar_probs.shape[1]}

    if args.smoke:
        out_dir = "/tmp/a3_smoke"
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(
        out_dir, f"{args.dataset}_{args.target_domain}_seed{args.seed}.npz")
    np.savez_compressed(fn, **out)
    with open(os.path.join(out_dir,
              f"{args.dataset}_{args.target_domain}_seed{args.seed}.json"),
              "w") as f:
        json.dump(meta, f, indent=2)
    print("saved", fn)
    print(f"shapes: TE{out['te_probs'].shape} hbar{out['hbar_probs'].shape} "
          f"AE{out['ae_probs'].shape}")
    return fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="pacs", choices=sorted(DATASETS))
    p.add_argument("--data_dir", default=None)
    p.add_argument("--target_domain", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no_cifa", action="store_true")
    p.add_argument("--no_swad", action="store_true")
    p.add_argument("--variant", default="full",
                   choices=list(VARIANT_OVERRIDES))
    p.add_argument("--hparam", action="append", default=[])
    p.add_argument("--config", default=None)
    p.add_argument("--a3_m", type=int, default=32,
                   help="number of random augmentation views for AE")
    p.add_argument("--smoke", action="store_true",
                   help="CPU synthetic self-check")
    args = p.parse_args()

    if not args.smoke:
        spec = DATASETS[args.dataset]
        if args.target_domain is None:
            args.target_domain = spec["domains"][0]
        if args.target_domain not in spec["domains"]:
            raise SystemExit(f"target_domain must be one of {spec['domains']}")
    else:
        args.dataset, args.target_domain = "smoke", "smoke"
    run(args)


if __name__ == "__main__":
    main()
