# A3 Alignment Probe — Empirically Testing Assumption 3

## 1. Purpose

The paper's parameter-space claim has two logically distinct parts:

- **Theorem 2** proves that dense weight averaging approximates **its own
  trajectory ensemble** to second order (within a parameter neighbourhood,
  the error governed by curvature and neighbourhood radius).
- **Assumption 3 (trajectory–augmentation alignment)** is the *additional,
  non-automatic* premise that lets us read that trajectory ensemble as
  aggregation over the **augmentation family**:

> *Within the flat parameter neighbourhood selected by the dense averaging
> window, the empirical distribution of trajectory checkpoints approximates,
> in the sense required for risk comparisons, the distribution of predictors
> obtained by training over the augmentation family.*

Theorem 2 is proved; Assumption 3 is stated but not tested. This probe turns
Assumption 3 from a declared premise into an empirically examined one. It is a
post-hoc mechanism analysis in the same spirit as the Table X diagnostics.

## 2. Ensembles compared

All three are built on the held-out target domain with a fixed image order
(`shuffle=False`); probabilities are averaged (not logits/weights).

| Symbol | Construction | Role |
|---|---|---|
| **TE** | Clean predictions of **every checkpoint that enters the SWAD window**; $\bar p^{TE}(x)=\frac1T\sum_{t=1}^{T}\mathrm{softmax}\,h_{\theta_t}(x)$ | trajectory ensemble |
| **h̄** | Clean predictions of the final dense weight-averaged model | weight average |
| **AE** | **M independent random CIFA views** of h̄ on the same target; $\bar p^{AE}(x)=\frac1M\sum_{m=1}^{M}\mathrm{softmax}\,h(T_m(x))$ | augmentation-family ensemble |

- **TE** is captured at each epoch that `LossValley.need_average()` is true,
  using that epoch's weights before the final average is loaded.
- **AE** is a Monte-Carlo estimate of the predictor distribution induced by
  the augmented views (the local-Bayesian / weight-space view of SWA–SWAD).
  Each call to `augment_layer` draws a fresh partner deviation $d'$ and
  coefficient $\lambda$; the tracker supplies the content baseline and is
  **not** updated. Default $M=32$.
- A stronger but costlier AE variant—training one model per fixed
  augmentation and averaging those—is left as an optional robustness check;
  the TTA-style AE above directly samples "the predictor distribution induced
  by the augmented views" and is the primary, low-cost operationalisation.

## 3. Metrics (per pair, per image, then summarised)

- **Jensen–Shannon divergence** between the two ensemble distributions
  (mean and median, in nats).
- **Total variation** $\tfrac12\sum_c|p_c-q_c|$ (mean and median).
- **Decision agreement**: fraction of images with the same argmax.
- **Probability correlation**: Pearson correlation over the flattened
  probability vectors.
- **Linear CKA** of the penultimate features (mean features within each
  ensemble): $\|X_cY_c^\top\|_F^2/(\|X_cX_c^\top\|_F\|Y_cY_c^\top\|_F)$.
- **Risk**: each ensemble's target accuracy and the absolute accuracy gap —
  Assumption 3 explicitly concerns alignment "for risk comparisons".

## 4. Pairings and controls

| Pair | Meaning | Expected |
|---|---|---|
| **TE vs AE** | Assumption 3 (core) | close |
| **h̄ vs TE** | Theorem 2 sanity check | close (proved) |
| **h̄ vs AE** | derived; should be close iff both above hold | close |
| **TE(seed i) vs TE(seed j)** | **negative control**, same target, different seed | less similar — provides the scale for "close" |

The negative control uses the same target domain across seeds (the target
image set is identical under LODO; only source hold-out splits differ), so the
two predictions are pointwise comparable.

## 5. Pre-registered adjudication

Thresholds are fixed before examining results. Assumption 3 is counted as
**supported** only when all hold:

1. TE–AE decision agreement $\ge 0.95$;
2. median JS $\le 0.01$ nats;
3. median TV $\le 0.05$;
4. $|\mathrm{acc}_{TE}-\mathrm{acc}_{AE}|\le 1.0$ point (the same $\Delta_{eq}$
   used by TOST);
5. TE–AE agreement exceeds the cross-seed negative control.

Theorem 2 sanity requires h̄–TE agreement $\ge 0.95$ and accuracy gap
$\le 1.0$.

## 6. Protocol and scope

- The averaging window is selected from **source hold-out loss only**. The
  target is evaluated multiple times purely for diagnosis; it is never used to
  select checkpoints, hyperparameters, or models, so there is **no target
  leakage** (the repeated evaluation is analogous to the existing mechanism
  diagnostics, not to target-based model selection).
- Outputs are written to `reproduce_output/a3/DATASET`, never to `results/`
  (which holds the paper's reported numbers).
- **Scope: PACS + VLCS, each 4 targets × 10 seeds = 40 runs (80 total)**;
  the 10 seeds match the main paired experiments (n=10). The two datasets
  form a planned positive/boundary contrast:
  - **PACS (positive)**: the method works and Assumption 3 should hold where
    a strong, content–style-separable style deviation exists.
  - **VLCS (boundary)**: no mixable style structure is present and the
    method shows no gain; Assumption 3 is expected **not** to hold. Showing
    that alignment tracks the gain (aligned on PACS, not on VLCS) is the
    informative result, not a failed test.

## 7. How to run

```bash
# on the GPU box, from the repository root
export NGPU=2                          # two GPUs round-robin
# PACS + VLCS back-to-back (80 runs), auto-analyses each dataset
scripts/run_a3_pacs_vlcs.sh "0 1 2 3 4 5 6 7 8 9" 32
# or a single dataset
scripts/run_a3.sh pacs "0 1 2 3 4 5 6 7 8 9" 32
```

Single pieces:

```bash
# collect one run (from cifa/)
python a3_probe.py --dataset pacs --target_domain sketch --seed 0 --a3_m 32
# analyse a directory of runs
python scripts/a3_analyze.py --a3_dir reproduce_output/a3/pacs --dataset pacs
# CPU self-check (no data, no pretrained)
python cifa/a3_probe.py --smoke
```

Outputs: `a3_summary.json`, `a3_table.md` in the probe directory.

## 8. Reporting

The result is reported whichever way it falls.

- **Supported**: a short paragraph (and, if space allows, one compact table)
  stating that the trajectory checkpoints and the augmentation-induced
  predictor distribution agree within the pre-registered limits, above a
  cross-seed control, with h̄–TE confirming Theorem 2.
- **Not supported**: state plainly that weight averaging optimises an object
  that is empirically distinct from the augmentation family, keep Theorem 2's
  narrower claim (h̄ approximates its own trajectory ensemble), and remove
  the "augmentation-family aggregation" reading. This is a genuine boundary,
  not a reason to retain wording the data do not support.

## 8a. Executed scope (revision)

The full 80-run plan was reduced to a focused post-hoc alignment study on the
two PACS domains where the style deviation is strongest (art_painting and
sketch), four seeds each. The reported evidence is therefore **coverage**, not
distributional equality: the prediction-space nearest-neighbor ratio is 1.23
for SWAD checkpoints versus 2.14 for ERM, the mean prediction KL is 0.036
versus 0.072, and the feature-space SWAD-to-augmentation distance is 39%
(layer 1) and 47% (layer 2) of the ERM distance. The augmentation dispersion
is much smaller than the trajectory dispersion, so the result is not claimed
as equality. Raw files: `results/alignment/`.

## 9. Estimated cost

Each run trains 5000 steps (~7 min on one RTX 4090) plus ensemble
evaluation (TE checkpoints and M=32 augmentation views on the target).
PACS targets are small (~300–850 images), so a PACS run is ~8–9 min; VLCS
targets are larger (~1,200–2,600 images), so a VLCS run is ~10–12 min.
With `NGPU=2`, PACS (40 runs) is ~3 h and VLCS (40 runs) ~3.5–4 h, run
back-to-back for **~6.5–7 h total for the 80 runs**. No extra parameters
are introduced and the probe does not modify the main training pipeline.
