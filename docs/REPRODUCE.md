# Reproduction guide

## Protocol

- **Benchmarks:** PACS, VLCS, OfficeHome, TerraIncognita (see
  [`DATASETS.md`](DATASETS.md)).
- **Leave-one-domain-out (LODO):** for each target domain, train on the other
  three; repeat for every domain.
- **Seeds:** 0–4 (5 seeds) for most benchmarks; the PACS CIFA/SWAD arms use
  0–9 (10 seeds). Main comparison = (3 datasets × 4 domains × 5 seeds + PACS
  4 domains × 10 seeds) × 2 methods = 200 runs; ERM controls add 80; the PACS
  ablation adds 100.
- **Backbone:** ResNet-50, ImageNet-pretrained; pretrained BatchNorm kept in
  eval mode (`freeze_bn`), following DomainBed.
- **Model selection:** source-domain 20% hold-out only (SWAD's LossValley). The
  target domain is never used for selection or tuning (`target-free`).
- **Hyperparameters:** frozen from the registry / YAML; no per-dataset tuning
  on the target. γ-style selection, where used, is inner leave-one-environment
  CV by worst validation accuracy.

## Reproduction paths

### A. Audit without training (seconds, CPU)

```bash
pip install -r requirements.txt
python scripts/collect_tables.py --root results     # regenerate both tables
cd cifa && python smoke_test.py                     # implementation correctness
```

### B. One dataset / method

```bash
DATA_DIR=/data/PACS bash scripts/run_sweep.sh pacs cifa "0 1 2 3 4"
# modes: cifa | swad | erm
```

### C. All main results

```bash
DATA_DIR_pacs=/data/PACS DATA_DIR_vlcs=/data/VLCS \
DATA_DIR_officehome=/data/OfficeHome DATA_DIR_terra=/data/terra_imagefolder \
bash scripts/reproduce_main.sh
```

### D. PACS ablation (100 runs)

```bash
DATA_DIR=/data/PACS bash scripts/run_ablation.sh
```

## Outputs

Each run writes one JSON, named `{variant}_{target_domain}_seed{seed}.json`,
with:

| Field | Meaning |
|---|---|
| `dataset`, `method`, `variant` | identifiers (present on newer runs) |
| `target_domain`, `seed` | the held-out domain and trial |
| `target_acc`, `target_loss` | held-out-domain accuracy / loss |
| `holdout_acc` | per-source-domain hold-out accuracy |
| `worst_holdout_acc` | worst source hold-out (selection signal, not OOD) |
| `swad_n_checkpoints` | number of averaged checkpoints |
| `aug_trigger_fraction` | how often the augmentation actually fired |
| `aug_mean_consistency_kl` | mean clean/augmented KL |
| `aug_mean_gate` | mean guard gate value |
| `hparams` | the exact configuration used |

The earliest PACS paired runs predate some diagnostic fields; freshly
reproduced runs contain all of them. `aggregate.py` prints a quick mean±std
table for one output directory.

## Checking a re-run against the paper

```bash
# tables for your own runs
python scripts/collect_tables.py --root reproduce_output
# side-by-side overall comparison (reported -> re-run)
python scripts/collect_tables.py --root results \
    --compare-root reproduce_output
```

Expect the re-run overall means within roughly one std of the reported values;
the signs of the worst-domain gaps and the OfficeHome null result should be
stable, while individual cells vary with scheduling and library versions.

## Compute and runtime

- All runs are single-process per job; the sweep scripts round-robin jobs across
  `NGPU` GPUs (one job per GPU at a time).
- Each run is 5000 optimizer steps at batch size 32 on a ResNet-50.
- **Measured end-to-end wall-clock per run** (one job on a single NVIDIA
  RTX 4090, including startup and data loading, `--workers 6`; CIFA/SWAD figures
  are from the seed-3/4 sweep, n=7–8 per dataset, ERM from the 5-seed sweep,
  n=19–20; mm:ss):

| Dataset | CIFA | SWAD | ERM |
|---|---|---|---|
| PACS | ~7:00 | ~7:30 | ~4:31 |
| VLCS | ~6:47 | ~7:19 | ~4:27 |
| OfficeHome | ~16:26 | ~15:59 | ~8:02 |
| TerraIncognita | ~14:44 | ~14:19 | ~8:07 |

  CIFA and SWAD timings are within run-to-run noise of each other. The PACS
  ablation variants run ~7:52 each, except `noswad` (~6:02).
- Total work is 380 runs (200 main + 80 ERM + 100 ablation), about 56 GPU-hours
  (~28 h on two RTX 4090 GPUs). A reviewer mainly interested in the claims can
  reproduce a single dataset × method (20 runs; 40 for PACS) or reduce seeds
  (`"0 1"`) for a fast check.

## Pre-submission / reproduction checklist

- [ ] `smoke_test.py` ends with `ALL SMOKE TESTS PASSED`.
- [ ] `collect_tables.py --root results` reproduces the paper's overall and
      per-domain numbers.
- [ ] Dataset folder layouts match [`DATASETS.md`](DATASETS.md); class order is
      identical across domains.
- [ ] Re-runs write to `reproduce_output/`, not over `results/`.
- [ ] Worst-domain gaps are positive on PACS/VLCS/Terra and ≈0 on OfficeHome.
- [ ] `aug_trigger_fraction` is non-trivial (the operator is not silently gated
      away) and `aug_mean_gate` behaves as described.
