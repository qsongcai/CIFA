# Reproduction guide

## Protocol

- **Benchmarks:** PACS, VLCS, OfficeHome, TerraIncognita (see
  [`DATASETS.md`](DATASETS.md)).
- **Leave-one-domain-out (LODO):** for each target domain, train on the other
  three; repeat for every domain.
- **Seeds:** 0–4 (5 seeds) for most benchmarks; the PACS comparison uses
  0–9 (10 seeds). The PACS CIFA/SWAD/classic-baseline numbers in the revision
  come from the authoritative unified-code batch (one codebase, identical
  SWAD schedule). The earlier reference-implementation batch is retained only
  for the batch-effect analysis.
- **Backbone:** ResNet-50 (and ResNet-18 for the budget checks),
  ImageNet-pretrained; pretrained BatchNorm kept in eval mode (`freeze_bn`),
  following DomainBed. ViT-B/16 cross-architecture checks are included.
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

### E. Revision experiments (ViT, alignment, curvature, TOST, gate pressure, cost)

The added experiments and their analysis scripts are:

```bash
# ViT-B/16 cross-architecture (PACS end-to-end, layer ablation, CLIP probe; VLCS end-to-end)
bash scripts/run_vit_e2e_all.sh
python scripts/vit_e2e_analysis.py            # aggregate -> means, paired deltas

# trajectory-augmentation alignment (prediction and feature space)
python scripts/trajectory_alignment.py
python scripts/feature_alignment.py

# curvature ratio kappa r^2 (float64 / float32 / quantization)
bash scripts/kappa/smoke_float64.py
python scripts/kappa/float64_hvp.py
python scripts/kappa/kappa_analyze.py

# three-margin TOST ladder (results/tables/tost_multi_margin.csv)
python scripts/tost_multi_margin.py

# gate pressure under strong augmentation
python scripts/gate_pressure_analysis.py

# batch effect and mixed-effects re-analysis
python scripts/batch_mixed_effects.py

# zero-checkpoint fallback analysis
python scripts/zero_checkpoint_analysis.py

# wall-clock / loader cost
bash scripts/compute_cost/run_loader_test.sh
python scripts/compute_cost/inspect_results.py
```

The raw outputs of these experiments live under `results/vit`,
`results/alignment`, `results/kappa`, `results/gate_pressure` and
`results/compute_cost`.

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

Expect the re-run overall means within roughly one std of the reported values.
The central finding is a statistical tie between CIFA and SWAD on PACS together
with null results on the other benchmarks, so the sign of a small delta is not
itself a stable quantity.

## Compute and runtime

- All runs are single-process per job; the sweep scripts round-robin jobs across
  `NGPU` GPUs (one job per GPU at a time).
- Each run is 5000 optimizer steps at batch size 32 on a ResNet-50.
- **Measured end-to-end wall-clock per run** (one job on a single NVIDIA
  RTX 4090, including startup and data loading, `--workers 6`; mm:ss):

| Dataset | CIFA | SWAD | ERM |
|---|---|---|---|
| PACS | ~7:00 | ~7:30 | ~4:31 |
| VLCS | ~6:47 | ~7:19 | ~4:27 |
| OfficeHome | ~16:26 | ~15:59 | ~8:02 |
| TerraIncognita | ~14:44 | ~14:19 | ~8:07 |

  CIFA and SWAD timings are within run-to-run noise of each other (SWAD is
  about 1.6–2.0× ERM, dominated by the per-epoch full hold-out validation used
  to locate the SWAD window, not by checkpoint averaging). Enabling
  multi-worker/persistent/pinned loading reduces the hold-out evaluation from
  111.2 s to 34.3 s and the net SWAD overhead from 33.5% to 9.6%.

## Pre-submission / reproduction checklist

- [ ] `smoke_test.py` ends with `ALL SMOKE TESTS PASSED`.
- [ ] `collect_tables.py --root results` reproduces the paper's overall and
      per-domain numbers.
- [ ] Dataset folder layouts match [`DATASETS.md`](DATASETS.md); class order is
      identical across domains.
- [ ] Re-runs write to `reproduce_output/`, not over `results/`.
- [ ] The PACS CIFA–SWAD contrast is a statistical tie (small delta, equivalence
      established by TOST), and the other benchmarks read as null.
- [ ] `aug_trigger_fraction` is non-trivial (the operator is not silently gated
      away) and `aug_mean_gate` behaves as described (dormant on PACS,
      activatable under strong augmentation).
