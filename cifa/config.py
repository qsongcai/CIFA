"""Hyperparameter registry for CIFA.

The DomainBed / backbone / SWAD defaults reproduce the standard
DomainBed SWAD protocol (ResNet-50, Adam, 5000 steps, hold-out 0.2).
CIFA-specific fields implement Algorithm 1 of the paper.
"""
from copy import deepcopy


def hparams_registry():
    return {
        # ---- Backbone / optimization (DomainBed defaults) ----
        "backbone": "resnet50",
        "pretrained": True,
        "steps": 5000,
        "batch_size": 32,
        "lr": 5e-5,
        "weight_decay": 0.0,
        "holdout_fraction": 0.2,
        "freeze_bn": True,          # DomainBed keeps pretrained BN in eval mode
        "dropout": 0.0,

        # ---- SWAD (DomainBed hparams_registry SWAD values) ----
        "swad": True,
        "swad_n_converge": 3,
        "swad_n_tolerance": 6,
        "swad_tolerance_ratio": 0.3,
        "swad_tolerance_epoch": 5,

        # ---- CIFA feature-moment augmentation ----
        "cifa_enabled": True,
        # style-dominant layers (ResNet bottleneck block outputs)
        "cifa_layers": ["layer1", "layer2"],
        "cifa_prob": 0.5,                       # trigger probability p
        "cifa_mix_mode": "mix",                 # 'mix' | 'gaussian'
        "cifa_lambda_min": 0.1,
        "cifa_lambda_max": 0.9,
        "cifa_baseline_momentum": 0.9,          # EMA momentum of class prototypes
        "cifa_moments_eps": 1e-5,
        "cifa_sigma_min": 1e-3,

        # ---- content-consistency guard ----
        "cifa_consistency_coef": 1.0,           # lambda_c
        "cifa_consistency_delta": 2.0,          # tolerance delta (nats); above -> down-weight
        "cifa_use_guard": True,                 # False (ablation): gate=1, no KL penalty

        # ---- soft worst-case view aggregation ----
        "cifa_soft_eta": 0.5,
        "cifa_aggregation": "softworst",        # 'softworst' | 'uniform' (ablation)

        # ---- content-baseline / style-deviation decomposition ----
        "cifa_decompose": True,                 # False (ablation): mix full moments (MixStyle-style)
    }


# channel widths of ResNet block outputs used to size the baseline tracker
RESNET_CHANNELS = {
    "resnet18": {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512},
    "resnet50": {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
}


def merge_overrides(hparams, overrides):
    """Apply a dict of overrides parsed from the CLI, typed by the default."""
    hparams = deepcopy(hparams)
    if overrides:
        for k, v in overrides.items():
            if k not in hparams:
                raise KeyError(f"unknown hparam: {k}")
            cur = hparams[k]
            if isinstance(cur, bool):
                hparams[k] = v.strip().lower() in ("1", "true", "yes", "y")
            elif isinstance(cur, list):
                hparams[k] = [s for s in v.split(",") if s != ""]
            elif cur is None:
                hparams[k] = v
            else:
                hparams[k] = type(cur)(v)
    return hparams
