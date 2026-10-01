"""Generate configs/*.yaml from the in-code hparams registry.

Run from the repository root:

    python scripts/gen_configs.py

It regenerates the four dataset-level configs (configs/{pacs,vlcs,officehome,
terra}.yaml) and the five ablation-variant configs (configs/ablation/*.yaml)
directly from ``cifa/config.py``, so the committed YAML can never silently
drift from the defaults the code actually uses.
"""
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "cifa"))

from config import hparams_registry  # noqa: E402
from datasets import DATASETS  # noqa: E402

DEFAULTS = hparams_registry()

DATASET_HEADER = {
    "pacs": (
        "# CIFA configuration for PACS (DomainBed leave-one-domain-out, ResNet-50).\n"
        "# Set `data_dir` to your local PACS root (the folder containing\n"
        "# art_painting/cartoon/photo/sketch), or leave it null to use the\n"
        "# registry default.\n"
        "# Example:\n"
        "#   cd cifa\n"
        "#   python train.py --dataset pacs --target_domain art_painting \\\n"
        "#       --config ../configs/pacs.yaml --seed 0\n"
    ),
    "vlcs": (
        "# CIFA configuration for VLCS (leave-one-domain-out, ResNet-50).\n"
        "# `data_dir` must point at the VLCS root whose per-domain folders\n"
        "# (CALTECH/LABELME/PASCAL/SUN) each contain train/test/crossval.\n"
        "# See docs/DATASETS.md for the required layout.\n"
    ),
    "officehome": (
        "# CIFA configuration for OfficeHome (leave-one-domain-out, ResNet-50).\n"
        "# `data_dir` points at the root containing Art/Clipart/Product/Real World.\n"
        "# The on-disk 4th folder may be spelled 'Realworld'; datasets.py accepts either.\n"
    ),
    "terra": (
        "# CIFA configuration for TerraIncognita (leave-one-domain-out, ResNet-50).\n"
        "# `data_dir` points at the ImageFolder layout produced by prepare_terra.py\n"
        "# (folders location_38/location_43/location_46/location_100).\n"
    ),
}

# variant -> (single changed hparam, one-line purpose)
ABLATION = {
    "full": ({}, "Full CIFA (the baseline configuration)."),
    "fullmoment": (
        {"cifa_decompose": False},
        "Mix the FULL moments (MixStyle/DSU-style) instead of only the style "
        "deviation. Tests whether the baseline/deviation decomposition is needed.",
    ),
    "noguard": (
        {"cifa_use_guard": False},
        "Remove the KL guard and consistency penalty (gate fixed at 1). Tests "
        "whether unguarded mixing destabilises training.",
    ),
    "uniform": (
        {"cifa_aggregation": "uniform"},
        "Average the clean/augmented views uniformly instead of soft "
        "worst-case aggregation.",
    ),
    "noswad": (
        {"swad": False},
        "Disable SWAD weight averaging. Isolates the contribution of weight "
        "averaging from feature-moment augmentation.",
    ),
}


def dump(path, obj, header):
    with open(path, "w") as f:
        f.write(header)
        f.write("\n")
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True,
                       default_flow_style=False)


def main():
    for key in ("pacs", "vlcs", "officehome", "terra"):
        spec = DATASETS[key]
        obj = {
            "dataset": key,
            "data_dir": None,
            "domains": list(spec["domains"]),
            "hparams": dict(DEFAULTS),
        }
        dump(os.path.join(ROOT, "configs", f"{key}.yaml"),
             obj, DATASET_HEADER[key])

    for variant, (change, purpose) in ABLATION.items():
        hp = dict(DEFAULTS)
        hp.update(change)
        obj = {
            "dataset": "pacs",
            "data_dir": None,
            "variant": variant,
            "changed_hparams": change,
            "hparams": hp,
        }
        header = (
            f"# Ablation variant '{variant}' on PACS: {purpose}\n"
            "# Reproduce with either --variant {0} (recommended) or by using the\n"
            "# explicit hparams below:\n"
            "#   python train.py --dataset pacs --target_domain art_painting \\\n"
            "#       --config ../configs/ablation/{0}.yaml --seed 0\n"
        ).format(variant)
        dump(os.path.join(ROOT, "configs", "ablation", f"{variant}.yaml"),
             obj, header)

    print("Generated configs for 4 datasets and 5 ablation variants.")


if __name__ == "__main__":
    main()
