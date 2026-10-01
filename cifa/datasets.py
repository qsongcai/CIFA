"""Domain data loading under the DomainBed protocol.

Expected on-disk layout (ImageFolder style), one folder per domain and
one sub-folder per class.

PACS:
    <data_dir>/
        art_painting/{dog,elephant,giraffe,guitar,horse,house,person}/*.jpg
        cartoon/...
        photo/...
        sketch/...

VLCS (this release ships fixed sub-splits; train+test+crossval are merged,
the extra "full" subset is ignored):
    <data_dir>/
        CALTECH/{train,test,crossval}/{0,1,2,3,4}/*.jpg
        LABELME/...
        PASCAL/...
        SUN/...

OfficeHome (the 4th folder is officially "Real World" with a space, but some
releases spell it "Realworld"; the loader auto-detects either spelling):
    <data_dir>/
        Art/<65 class folders>/*
        Clipart/...
        Product/...
        Real World/  (or Realworld/)

For each source domain a fixed-seed split produces an in-domain training
subset (train transform) and a hold-out subset (test transform). The
hold-out fraction and SWAD window are the only uses of validation data;
the target domain is never touched until final evaluation.
"""
import os
import random
import re
from types import SimpleNamespace

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

PACS_DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
VLCS_DOMAINS = ["CALTECH", "LABELME", "PASCAL", "SUN"]
# Canonical OfficeHome names used in code/reports; the on-disk 4th folder may
# be spelled "Realworld" (no space) and is resolved via domain_aliases below.
OFFICEHOME_DOMAINS = ["Art", "Clipart", "Product", "Real World"]
# TerraIncognita (converted from the raw ECCV release into an ImageFolder tree;
# domains are camera locations, 10 wildlife classes).
TERRA_DOMAINS = ["location_38", "location_43", "location_46", "location_100"]

# dataset name -> domain folder order and default on-disk root
DATASETS = {
    "pacs": {
        "domains": PACS_DOMAINS,
        "default_dir":
            "/mnt/sdd1/datasets_qsong/PACS/Homework3-PACS-master/PACS",
    },
    "vlcs": {
        "domains": VLCS_DOMAINS,
        "default_dir": "/mnt/sdd1/datasets_qsong/VLCS/VLCS",
        # this release nests each domain under fixed split folders;
        # merging the three splits recovers the full DomainBed VLCS.
        "subroots": ["train", "test", "crossval"],
    },
    "officehome": {
        "domains": OFFICEHOME_DOMAINS,
        "default_dir": "/mnt/sdd1/datasets_qsong/OfficeHome",
        # canonical name -> candidate on-disk folder names (first existing wins)
        "domain_aliases": {
            "Real World": ["Real World", "Realworld", "Real_World"],
        },
    },
    "terra": {
        "domains": TERRA_DOMAINS,
        # ImageFolder tree produced by prepare_terra.py from the raw release
        "default_dir":
            "/mnt/sdd1/datasets_qsong/TerraIncognita/terra_imagefolder",
    },
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform():
    # DomainBed default image augmentation
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def test_transform():
    # DomainBed test transform: direct resize to 224x224
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImageList(Dataset):
    """Dataset from an explicit list of (path, label) pairs."""

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


def _norm_name(s):
    """Normalize a folder name: lowercase and drop spaces/_/- separators.

    "Real World", "Realworld", "RealWorld", "real_world" all -> "realworld".
    """
    return re.sub(r"[\s_\-]+", "", s.lower())


def _domain_dir(data_dir, canonical, aliases=None):
    """Resolve a canonical domain name to the on-disk folder that exists.

    First probes the canonical name and explicit aliases exactly; if none
    match, falls back to a case/separator-insensitive match against the actual
    sub-folders of data_dir (so "RealWorld" is found for canonical
    "Real World"). Returns (folder_path, actual_folder_name); raises if no
    folder normalizes to the canonical name.
    """
    candidates = [canonical] + [a for a in (aliases or []) if a != canonical]
    # 1) exact match
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.isdir(path):
            return path, name
    # 2) case/separator-insensitive match over real sub-folders
    want = _norm_name(canonical)
    actual = (sorted(os.listdir(data_dir))
              if os.path.isdir(data_dir) else None)
    if actual is not None:
        for name in actual:
            if (os.path.isdir(os.path.join(data_dir, name))
                    and _norm_name(name) == want):
                return os.path.join(data_dir, name), name
    raise FileNotFoundError(
        f"no folder for domain '{canonical}' under {data_dir}; "
        f"tried exact {candidates}, and no sub-folder normalizes to "
        f"'{want}'. Actual entries in {data_dir}: {actual}")


def _load_domain(domain_dir, subroots=None):
    """ImageFolder-like (.classes/.samples) for one domain.

    With subroots, merge several fixed split folders that sit one level
    below the domain folder (used for the nested VLCS release).
    """
    if subroots is None:
        return datasets.ImageFolder(domain_dir)
    samples, classes = [], None
    for sr in subroots:
        folder = datasets.ImageFolder(os.path.join(domain_dir, sr))
        if classes is None:
            classes = folder.classes
        elif folder.classes != classes:
            raise RuntimeError(f"class ordering mismatch under {domain_dir}/{sr}")
        samples += folder.samples
    return SimpleNamespace(classes=classes, samples=samples)


def get_loo_dataset(data_dir, domains, target_domain,
                    holdout_fraction=0.2, seed=0, subroots=None,
                    domain_aliases=None):
    """Build train / hold-out / target datasets for a leave-one-domain-out run.

    Returns dict with:
      train_samples   : merged list of (path,label) over source train splits
      holdout_by_domain : {domain: ImageList} (test transform)
      target          : ImageList over the held-out domain
      classes         : ordered class names

    domain_aliases maps a canonical domain name to alternate on-disk folder
    spellings; target_domain and the returned keys always use canonical names.
    """
    if target_domain not in domains:
        raise ValueError(f"target_domain must be one of {domains}")
    source_domains = [d for d in domains if d != target_domain]

    train_samples = []
    holdout_by_domain = {}
    classes = None

    for di, domain in enumerate(source_domains):
        domain_dir, actual = _domain_dir(
            data_dir, domain, (domain_aliases or {}).get(domain))
        folder = _load_domain(domain_dir, subroots)
        if classes is None:
            classes = folder.classes
        elif folder.classes != classes:
            raise RuntimeError(f"class ordering mismatch in {domain}")

        # deterministic split; offset seed per domain so splits differ
        tr_idx, va_idx = _split_indices(
            len(folder.samples), holdout_fraction, seed + 1000 * di)
        train_samples += [folder.samples[i] for i in tr_idx]
        holdout_by_domain[domain] = ImageList(
            [folder.samples[i] for i in va_idx], test_transform())

    target_dir, _ = _domain_dir(
        data_dir, target_domain, (domain_aliases or {}).get(target_domain))
    target_folder = _load_domain(target_dir, subroots)
    if target_folder.classes != classes:
        raise RuntimeError("target class ordering mismatch")
    target = ImageList(target_folder.samples, test_transform())

    # shuffle merged training list once (loader also shuffles)
    rng = random.Random(seed + 777)
    rng.shuffle(train_samples)

    return {
        "train_dataset": ImageList(train_samples, train_transform()),
        "train_samples": train_samples,
        "holdout_by_domain": holdout_by_domain,
        "target": target,
        "classes": classes,
    }


def get_dataset(name, data_dir, target_domain, holdout_fraction=0.2, seed=0):
    """Dataset-registry entry point used by train.py."""
    if name not in DATASETS:
        raise ValueError(f"dataset must be one of {sorted(DATASETS)}")
    reg = DATASETS[name]
    return get_loo_dataset(
        data_dir, reg["domains"], target_domain,
        holdout_fraction=holdout_fraction, seed=seed,
        subroots=reg.get("subroots"),
        domain_aliases=reg.get("domain_aliases"))


def get_pacs(data_dir, target_domain, holdout_fraction=0.2, seed=0):
    """PACS leave-one-domain-out datasets (kept for compatibility)."""
    return get_loo_dataset(
        data_dir, PACS_DOMAINS, target_domain,
        holdout_fraction=holdout_fraction, seed=seed)


def get_vlcs(data_dir, target_domain, holdout_fraction=0.2, seed=0):
    """VLCS leave-one-domain-out datasets (merges train/test/crossval)."""
    return get_loo_dataset(
        data_dir, VLCS_DOMAINS, target_domain,
        holdout_fraction=holdout_fraction, seed=seed,
        subroots=["train", "test", "crossval"])


def get_officehome(data_dir, target_domain, holdout_fraction=0.2, seed=0):
    """OfficeHome leave-one-domain-out datasets (4th folder auto-detected)."""
    return get_loo_dataset(
        data_dir, OFFICEHOME_DOMAINS, target_domain,
        holdout_fraction=holdout_fraction, seed=seed,
        domain_aliases={"Real World": ["Real World", "Realworld", "Real_World"]})


def get_terra(data_dir, target_domain, holdout_fraction=0.2, seed=0):
    """TerraIncognita leave-one-domain-out datasets (camera-location domains)."""
    return get_loo_dataset(
        data_dir, TERRA_DOMAINS, target_domain,
        holdout_fraction=holdout_fraction, seed=seed)


def infinite_loader(loader):
    """Cycle a DataLoader indefinitely (DomainBed trains a fixed #steps)."""
    while True:
        for batch in loader:
            yield batch
