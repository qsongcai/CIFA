# CIFA — Content-conditioned Invariant Feature-statistic Augmentation

**Reproducibility package** for

> *Content-conditioned Invariant Feature-statistic Augmentation: Unifying
> Robustness and Weight Averaging for Domain Generalization*
>
> Qingsong Cai (corresponding author), Hao Wang, Tianze Hao — School of
> Computer and Artificial Intelligence, Beijing Technology and Business
> University, Beijing 100048, China · submitted to **IEEE Access**.

CIFA shows that the feature-statistic mixing used by MixStyle/DSU and the weight
averaging used by SWAD/DiWA are not two unrelated tricks: under a
content–style decomposition, mixing only the **style deviation** (while holding
the content baseline and the normalized residual fixed) is itself a
feature-level **Causal Invariant Transform (CIT)**. The resulting operator is
plug-and-play, uses no target-domain data, and pairs with SWAD.

This repository lets a reviewer (a) re-derive every number in the paper from the
raw per-run files, on a CPU, in seconds; and (b) re-run the full experiment
sweep with one command.

---

## 1. Quick start (CPU, no dataset, no GPU)

```bash
# 1. install
pip install -r requirements.txt

# 2. recompute ALL reported tables from the 380 raw run files -> results/tables/*.md
python scripts/collect_tables.py --root results

# 3. correctness test of the augmentation, EMA baseline, gradients and SWAD
cd cifa && python smoke_test.py && cd ..
```

Step 2 regenerates `results/tables/main_results.md` and
`results/tables/ablation_results.md`; the numbers are computed, not copied, so
they match the paper only if the run files genuinely produce them.

## 2. Reproduce the experiments (GPU)

Point `DATA_DIR` at each benchmark (see [`docs/DATASETS.md`](docs/DATASETS.md)
for the required ImageFolder layout), then either run everything or one piece:

```bash
# all four benchmarks, SWAD control + CIFA; 5 seeds for most benchmarks, 10 for PACS
DATA_DIR_pacs=/data/PACS DATA_DIR_vlcs=/data/VLCS \
DATA_DIR_officehome=/data/OfficeHome DATA_DIR_terra=/data/terra_imagefolder \
bash scripts/reproduce_main.sh

# PACS component ablation: 5 variants x 4 domains x 5 seeds = 100 runs
DATA_DIR=/data/PACS bash scripts/run_ablation.sh

# or a single dataset / method / seed list
DATA_DIR=/data/PACS bash scripts/run_sweep.sh pacs cifa "0 1 2 3 4"
```

**Runtime.** On a single RTX 4090 one run (5000 steps) takes roughly 7 min
(PACS/VLCS), 14–15 min (Terra), or 16 min (OfficeHome); the full 380-run sweep
is about 56 GPU-hours (~28 h on two GPUs). See
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

Re-runs are written to `reproduce_output/` (git-ignored) and never overwrite
the paper's reported results in `results/`. Build the re-run tables with

```bash
python scripts/collect_tables.py --root reproduce_output
# compare reported vs re-run overall accuracy
python scripts/collect_tables.py --root results --compare-root reproduce_output
```

## 3. Main results

Overall accuracy (%), DomainBed leave-one-domain-out, ResNet-50, 5 seeds
(10 for the PACS CIFA/SWAD arms; mean±std, sample std ddof=1):

| Dataset | ERM | SWAD | CIFA | Δ (CIFA−SWAD) | positive cells | worst-domain Δ |
|---|---|---|---|---|---|---|
| PACS | 84.38±1.27 | 87.30±0.51 | **88.29±0.58** | +0.99 | 27/40 | **+3.14 (9/10)** |
| VLCS | 76.04±0.91 | 79.03±0.58 | **79.29±0.73** | +0.26 | 11/20 | **+0.59 (4/5)** |
| OfficeHome | 67.30±0.13 | 71.41±0.56 | 71.45±0.79 | +0.04 | 11/20 | +0.08 (3/5) |
| TerraIncognita | 49.10±1.82 | 49.96±2.08 | **50.96±1.57** | +1.00 | 13/20 | **+1.85 (4/5)** |

`worst-domain Δ` compares, per seed, CIFA against SWAD on the domain where SWAD
is weakest that seed. Per-domain tables:
[`results/tables/main_results.md`](results/tables/main_results.md).

**Read these numbers honestly.** CIFA on average *matches* SWAD; its consistent
advantage is a small improvement on the **worst domain** of PACS, VLCS and
Terra. On OfficeHome there is **no measurable gain** (Δ≈0.04). See
[`docs/CLAIMS.md`](docs/CLAIMS.md).

## 4. Ablations on PACS

**Remaining component ablation (4 variants $\times$ 4 targets $\times$ 5 seeds):**

| variant | overall mean±std | Δ vs full (overall) | Δ vs full (worst domain) |
|---|---|---|---|
| full | 88.36±0.48 | — | — |
| fullmoment (mix full moments) | 88.59±0.67 | +0.23 | +0.09 |
| uniform aggregation | 87.88±0.32 | −0.47 | −1.28 |
| noswad | 85.93±1.70 | −2.42 | −1.36 |

**Gate $\times$ penalty factorial (4 cells $\times$ 4 targets $\times$ 10 seeds):**
removing both safeguards (`noguard`) changes overall accuracy by −0.36 (paired
−0.36±0.79, 95% CI [−0.92, +0.20]; a TOST procedure at ±1 point establishes
equivalence) and the own-worst target by −1.55 (p = 0.079, not significant).
Neither safeguard gives a detectable gain on PACS: the gate is never invoked
(mean gate identically 1.00 across the tolerance sweep δ ∈ {0.5, 1, 2, 8}), so
both act as dormant safeguards that fire only under stronger, deliberately
stressed augmentations. Overall / own-worst cells: full 88.29/81.06, gateonly
88.60/80.54, penaltyonly 88.15/80.65, noguard 87.93/79.52.

`fullmoment` is within run-to-run noise: the baseline/deviation split is what
makes the content-preservation bound possible, not a source of accuracy. Removing SWAD is
the largest single drop, i.e. weight averaging sets the overall level; the
feature-moment term targets the worst domain.

## 5. Method in one paragraph

At a style-dominant block, split the instance moments $m(Z)=(\mu,\sigma)$ into a
class-prototype **content baseline** $b$ (an EMA of moments) and a **style
deviation** $d=m-b$ with $d\perp C$. Mix only the deviation,
$d_{\text{aug}}=\lambda d+(1-\lambda)d'$, and recompose:

$$T(Z)=\widetilde{\sigma}\,\frac{Z-\mu}{\sigma}+\widetilde{\mu},
\qquad (\widetilde{\mu},\widetilde{\sigma})=b+d_{\text{aug}}.$$

The baseline and the normalized residual $(Z-\mu)/\sigma$ are held fixed, so
$T$ approximately preserves causal content and is a feature-level CIT; the
approximation error is bounded by testable conditions and an explicit
recomposition assumption. A convex-hull minimax analysis gives a risk bound
(with an explicit coverage gap), and SWAD's dense average is read as a
second-order approximation to the trajectory / augmentation-family ensemble,
which needs an explicit alignment assumption. See `cifa/cifa_aug.py`
(Algorithm 1) and the paper for the assumptions.

## 6. Repository layout

```
CIFA-Repro/
├── cifa/                 # core package (run inside this dir)
│   ├── config.py         # hparams registry, overrides
│   ├── datasets.py       # 4 benchmarks, DomainBed transforms/splits
│   ├── networks.py       # segmented ResNet, clean + augmented forward
│   ├── cifa_aug.py       # baseline EMA + style-deviation mixing (Alg. 1)
│   ├── swad.py           # LossValley + dense weight averaging
│   ├── train.py          # entry point; --config, --variant, --hparam
│   ├── aggregate.py      # JSON -> mean±std table
│   └── smoke_test.py     # CPU correctness test
├── configs/              # 4 dataset YAMLs + 5 ablation YAMLs
├── scripts/
│   ├── run_sweep.sh          # one dataset/method sweep
│   ├── reproduce_main.sh    # all four benchmarks
│   ├── run_ablation.sh      # PACS 100-run ablation
│   ├── collect_tables.py    # raw JSON -> Markdown tables (audit tool)
│   ├── gen_configs.py       # regenerate configs from code defaults
│   └── prepare_terra.py     # raw Terra release -> ImageFolder
├── results/              # paper's reported runs (380 JSON) + tables/
├── docs/                 # SETUP, DATASETS, REPRODUCE, CLAIMS
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## 7. Documentation

- [`docs/SETUP.md`](docs/SETUP.md) — environment and install troubleshooting.
- [`docs/DATASETS.md`](docs/DATASETS.md) — download links and exact folder layouts.
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md) — full protocol, compute, expected outputs.
- [`docs/CLAIMS.md`](docs/CLAIMS.md) — what the paper does and does not claim.

## 8. Citation

Please cite the paper (bibliography in [`CITATION.cff`](CITATION.cff), to be
finalized with the publication details).

## License

Code is released under the [MIT License](LICENSE). The datasets retain their
original licenses; this package does not redistribute them.
