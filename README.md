# CIFA — Content-conditioned Invariant Feature-statistic Augmentation

**Reproducibility package** for

> *Content-conditioned Invariant Feature-statistic Augmentation: Unifying
> Robustness and Weight Averaging for Domain Generalization*
>
> Qingsong Cai (corresponding author), Hao Wang, Tianze Hao — School of
> Computer and Artificial Intelligence, Beijing Technology and Business
> University, Beijing 100048, China · under revision at **Electronics** (MDPI),
> Manuscript ID: electronics-4654208.

CIFA shows that the feature-statistic mixing used by MixStyle/DSU and the
weight averaging used by SWAD/DiWA are not two unrelated tricks. Under a
content–style decomposition, mixing only the **style deviation** (while
holding the content baseline and the normalized residual fixed) admits a
**content-preservation bound**, and the dense weight average can be read as a
second-order approximation to the trajectory ensemble. The operator is
target-free, pairs with SWAD, and is supported by a reproducible experimental
platform.

This repository lets a reviewer (a) re-derive every number in the paper from
the raw per-run files, on a CPU, in seconds; and (b) re-run the full
experiment sweep with one command.

---

## 1. Quick start (CPU, no dataset, no GPU)

```bash
# 1. install
pip install -r requirements.txt

# 2. recompute the reported tables from the raw run files -> results/tables/*.md
python scripts/collect_tables.py --root results

# 3. correctness test of the augmentation, EMA baseline, gradients and SWAD
cd cifa && python smoke_test.py && cd ..
```

The tables are computed, not copied, so they match the paper only if the run
files genuinely produce them.

## 2. Reproduce the experiments (GPU)

Point `DATA_DIR` at each benchmark (see [`docs/DATASETS.md`](docs/DATASETS.md)
for the required ImageFolder layout), then either run everything or one piece:

```bash
# all four benchmarks, SWAD control + CIFA
DATA_DIR_pacs=/data/PACS DATA_DIR_vlcs=/data/VLCS \
DATA_DIR_officehome=/data/OfficeHome DATA_DIR_terra=/data/terra_imagefolder \
bash scripts/reproduce_main.sh

# PACS component ablation
DATA_DIR=/data/PACS bash scripts/run_ablation.sh

# or a single dataset / method / seed list
DATA_DIR=/data/PACS bash scripts/run_sweep.sh pacs cifa "0 1 2 3 4"
```

Re-runs are written to `reproduce_output/` (git-ignored) and never overwrite
the reported results in `results/`. The added revision experiments (ViT,
alignment, curvature, TOST, gate pressure, cost) are listed in
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## 3. Main results

Overall OOD accuracy (%), DomainBed leave-one-domain-out, ResNet-50, ten seeds
for PACS (macro mean±std over seeds). The PACS row uses the authoritative
unified-code batch; the reference-batch gain did not replicate:

| Dataset | ERM | SWAD | CIFA | Δ (CIFA−SWAD) | Pos. cells | Worst-domain Δ |
|---|---|---|---|---|---|---|
| PACS | 84.27±1.93 | 88.16±0.62 | 87.60±0.92 | **−0.57** | 14/40 | −0.93 (5/10) |
| VLCS | 75.77±1.20 | 79.09±0.85 | 79.15±0.79 | +0.06 | 18/40 | −0.18 (5/10) |
| OfficeHome | 67.24±0.57 | 71.45±0.42 | 71.34±0.61 | −0.11 | 18/40 | −0.26 (4/10) |
| TerraIncognita | 50.09±2.24 | 51.17±2.02 | 51.80±2.12 | +0.63 | 24/40 | +2.23 (7/10) |

**Read these numbers honestly.** CIFA is statistically tied with SWAD on PACS
and the other benchmarks show null or near-null differences. The pooled
two-batch PACS estimate is +0.21 (CI [−0.30, +0.72], p = 0.39), and the
exploratory worst-domain direction is not robust. See
[`docs/CLAIMS.md`](docs/CLAIMS.md).

## 4. PACS classic-baseline comparison and ablations

**Unified-code classic comparison (Table 5).** All methods share one codebase,
seeds, deterministic splits and the identical SWAD schedule:

| Method | Overall |
|---|---|
| EFDM | 88.93 |
| CORAL | 88.04 |
| CIFA | 87.60 |
| Fish | 87.54 |
| RSC | 87.47 |
| SWAD | 88.16 |

CIFA is statistically equivalent to CORAL, RSC and Fish at a one-point margin;
EFDM is the nominal leader, but the contrast meets neither the superiority nor
the strictest-adjusted equivalence criterion.

**Content baseline / style deviation (fullmoment).** The decomposition is an
enabling condition for the content-preservation bound, not a measured
performance source: fullmoment is statistically indistinguishable from full
CIFA (paired +0.23 ± 0.75, p = 0.53; 11/20 cells; worst +0.09, p = 0.90).

**SWAD / noswad.** Removing SWAD is the largest drop; its main value is
variance reduction. Under the optimized loader full CIFA (w4) exceeds noswad by
7.50 points on average, while noswad has a very large spread (65.54–89.40).

## 5. Added revision experiments

- **Cross-architecture (ViT-B/16).** PACS end-to-end: ERM 81.06, SWAD 87.86,
  CIFA 88.41 (CIFA−SWAD +0.55, CIFA−ERM +7.35, 20/20); CLIP linear probe tie
  (−0.006 ± 0.077). VLCS end-to-end: CIFA−ERM +2.89 overall and +3.39 on the
  worst source hold-out (19/20; the hold-out is a source-domain diagnostic, not
  a target-domain number).
- **Trajectory–augmentation alignment.** Prediction-space nearest-neighbor
  ratio 1.23 (SWAD) vs 2.14 (ERM); prediction KL 0.036 vs 0.072; feature-space
  distance 39%/47% of the ERM distance. This is coverage rather than
  distributional equality.
- **Curvature ratio κr² (float64).** Window radius 1.705 ± 0.111, true cell
  radius 1.47×10⁻⁶, within-cell second-order term ~1.16×10⁻¹³, window gap
  0.0828 ± 0.0115; the bounded kink sequence dominates the smooth curvature.
- **Gate pressure.** Under strong augmentation on VLCS the gate activates in
  all four runs (mean gate 0.132), reduces post-augmentation KL from 0.196 to
  0.137, and leaves accuracy unchanged (mean +0.25, worst −0.10).
- **Equivalence testing.** TOST on a three-margin ladder: at ±2 equivalence is
  widespread, at ±1 it holds for pooled PACS, VLCS and OfficeHome, at ±0.5 only
  for VLCS and OfficeHome.

## 6. Method in one paragraph

At a style-dominant block, split the instance moments $m(Z)=(\mu,\sigma)$ into a
class-prototype **content baseline** $b$ and a **style deviation** $d=m-b$ with
$d\perp C$. Mix only the deviation,
$d_{\text{aug}}=\lambda d+(1-\lambda)d'$, and recompose:

$$T(Z)=\widetilde{\sigma}\,\frac{Z-\mu}{\sigma}+\widetilde{\mu},
\qquad (\widetilde{\mu},\widetilde{\sigma})=b+d_{\text{aug}}.$$

The baseline and the normalized residual are held fixed, so the operator
approximately preserves content under a stated bounded-likelihood-ratio
hypothesis. A convex-hull minimax analysis gives a conditional risk bound (with
an explicit, non-identifiable coverage gap), and SWAD's dense average is read as
a second-order approximation to the trajectory ensemble under an alignment
assumption. See `cifa/cifa_aug.py` (Algorithm 1) and the paper for the
assumptions.

## 7. Repository layout

```
CIFA-Repro/
├── cifa/                 # core package (run inside this dir)
│   ├── config.py         # hparams registry, overrides
│   ├── datasets.py       # 4 benchmarks, DomainBed transforms/splits
│   ├── networks.py       # segmented ResNet, clean + augmented forward
│   ├── cifa_aug.py       # baseline EMA + style-deviation mixing (Alg. 1)
│   ├── swad.py           # LossValley + dense weight averaging
│   ├── train.py          # entry point; --config, --variant, --hparam
│   ├── a3_probe.py       # trajectory / augmentation prediction probe
│   ├── aggregate.py      # JSON -> mean±std table
│   └── smoke_test.py     # CPU correctness test
├── configs/              # 4 dataset YAMLs + ablation YAMLs
├── scripts/
│   ├── run_sweep.sh / reproduce_main.sh / run_ablation.sh
│   ├── collect_tables.py          # raw JSON -> Markdown tables
│   ├── vit_e2e_analysis.py        # ViT aggregation
│   ├── trajectory_alignment.py / feature_alignment.py
│   ├── tost_multi_margin.py       # three-margin TOST
│   ├── gate_pressure_analysis.py
│   ├── batch_mixed_effects.py     # batch effect / mixed-effects
│   ├── zero_checkpoint_analysis.py
│   ├── kappa/                     # float64 / float32 / kink analysis
│   └── compute_cost/              # wall-clock / loader cost
├── results/              # raw run JSON + tables/
│   ├── pacs/ vlcs/ officehome/ terra/ erm/
│   ├── vit/ alignment/ kappa/ gate_pressure/ compute_cost/
│   └── tables/
├── figures/              # pairwise, ablation, gate, sensitivity figures
├── docs/                 # SETUP, DATASETS, REPRODUCE, CLAIMS, A3_PROBE
├── requirements.txt / environment.yml
├── CITATION.cff
└── LICENSE
```

## 8. Documentation

- [`docs/SETUP.md`](docs/SETUP.md) — environment and install troubleshooting.
- [`docs/DATASETS.md`](docs/DATASETS.md) — download links and exact folder layouts.
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md) — full protocol, revision experiments, compute.
- [`docs/CLAIMS.md`](docs/CLAIMS.md) — what the paper does and does not claim.
- [`docs/A3_PROBE.md`](docs/A3_PROBE.md) — trajectory–augmentation alignment probe.

## 9. Citation

Please cite the paper (bibliography in [`CITATION.cff`](CITATION.cff)); the
volume, article number and DOI will be added upon publication.

## License

Code is released under the [MIT License](LICENSE). The datasets retain their
original licenses; this package does not redistribute them.
