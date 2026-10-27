# Claims and scope

This document states, plainly, what the paper does and does not claim. The
theory is conditional and the empirical results are reported honestly,
including the null results; please read everything in that light.

## What we claim

1. **A unified reading of two heuristic families.** Under an explicit
   content–style model, mixing only the *style deviation* of feature moments
   admits a **content-preservation bound**. This gives one framework that
   covers feature-statistic mixing (MixStyle/DSU) and connects it to weight
   averaging (SWAD/DiWA).
2. **Approximate content preservation.** Holding the content baseline and the
   normalized residual fixed while resampling the deviation keeps the content
   approximately invariant. The bound is stated under an explicit
   bounded-likelihood-ratio hypothesis; the total-variation term is the main
   bound and the KL term is a valid corollary. The bound is not zero.
3. **A conditional minimax risk bound with an explicit coverage gap.** The
   convex-hull minimax argument bounds risk over the augmentation family and
   states the coverage gap when that hull does not contain the target shift.
   The coverage gap is not identifiable without target access, so this is a
   conditional bound, not a guarantee. The convex-hull radius is an empirical
   estimate, not a universal Carathéodory d+1 claim.
4. **Weight averaging as an approximate trajectory ensemble.** SWAD's dense
   average is a second-order approximation to the trajectory ensemble under a
   C² regularity hypothesis, with an explicit kink term for ReLU/BatchNorm
   networks. Reading it as an augmentation-family ensemble needs an explicit
   alignment assumption that is supported empirically but not proven.
5. **A target-free operator with a reproducible platform.** CIFA never
   accesses the target domain; model selection uses only source-domain hold-out
   splits. This repository lets every reported number be re-derived from the
   raw run files.

## What we do NOT claim

- **Not a mean-accuracy gain over SWAD.** The authoritative unified-code PACS
  batch gives a statistical tie: CIFA−SWAD = −0.57 over ten seeds (14/40
  positive cells), and the pooled two-batch estimate is +0.21 (CI
  [−0.30, +0.72], p = 0.39). The early reference-batch gain (+0.99) did not
  replicate.
- **No robust worst-domain gain.** The unified batch reads −0.93 on the
  own-worst domain (5/10 positive); the pooled twenty-seed estimate is +0.89
  (p = 0.28). The exploratory worst-domain direction is therefore not robust.
- **VLCS, OfficeHome and Terra are null or near-null overall.** Δ = +0.06,
  −0.11 and +0.63 respectively; none establishes a mean-accuracy improvement.
- **The b/d decomposition is not a performance source.** The "fullmoment"
  variant without the decomposition is statistically indistinguishable from
  full CIFA (paired +0.23 ± 0.75, p = 0.53; 11/20 cells; worst +0.09,
  p = 0.90). The decomposition is an enabling condition for the
  content-preservation bound, not a measured accuracy gain.
- **Not causal effect recovery.** The goal is prediction robustness under
  shift, not recovering a structural causal coefficient. The consistency
  diagnostic measures prediction stability, not latent causal-content
  preservation.
- **The guarantees are conditional.** They hold only when the content–style
  decomposition, independence, recomposition, coverage and alignment
  assumptions hold.
- **Separability is an enabling condition, not a proven necessity.** The
  class-conditional style-moment separability statement is a hypothesis
  suggested by the model and the observed experiments, not an independently
  verified causal fact.

## Assumptions and how they are checked

| Assumption | Role | How it is checked |
|---|---|---|
| Content–style generation (C determines the label, S indexes the domain) | foundation of the reading | content–style model; channel-moment decomposition |
| Channel moments split into a content baseline b and style deviation d=m−b | defines what may be mixed | class prototypes / EMA in `cifa_aug.py`; ablation `fullmoment` |
| Recomposition keeps content approximately fixed | bounds transform error | bounded-likelihood-ratio bound; prediction KL; gate |
| Convex-hull coverage of the target augmentation | risk bound applies | coverage gap stated; radius estimated empirically |
| Assumption 3: trajectory–augmentation alignment | weight-averaging-as-ensemble reading | 16-run alignment study (`results/alignment`, Appendix Table A3) |

## Additional evidence in the revision

- **Cross-architecture (ViT-B/16).** PACS end-to-end: ERM 81.06, SWAD 87.86,
  CIFA 88.41 (CIFA−SWAD +0.55, CIFA−ERM +7.35, 20/20). A CLIP linear probe
  gives a tie (CIFA−SWAD −0.006 ± 0.077). VLCS end-to-end: CIFA−ERM +2.89
  overall and +3.39 on the worst domain (19/20).
- **Trajectory–augmentation alignment.** In the prediction space the
  nearest-neighbor ratio is 1.23 for SWAD checkpoints versus 2.14 for ERM, and
  the mean prediction KL is 0.036 versus 0.072. In the feature space the
  SWAD-to-augmentation distance is 39% (layer 1) and 47% (layer 2) of the ERM
  distance. This is coverage rather than distributional equality (the
  augmentation dispersion is much smaller than the trajectory dispersion).
- **Curvature ratio κr² (float64).** Window radius 1.705 ± 0.111, true cell
  radius 1.47×10⁻⁶, within-cell second-order term ~1.16×10⁻¹³, window gap
  0.0828 ± 0.0115. The window gap is roughly 8.8×10¹¹ times the within-cell
  term, so the ensemble deviation is dominated by the bounded kink sequence.
- **Gate pressure.** Under strong augmentation on VLCS the gate activates in
  all four runs (mean gate 0.132), reduces the post-augmentation KL from 0.196
  to 0.137, and leaves accuracy unchanged (mean +0.25, worst −0.10). On PACS
  the gate is dormant.
- **Equivalence testing.** TOST is reported on a three-margin ladder: at ±2
  equivalence is widespread, at ±1 it holds for pooled PACS, VLCS and
  OfficeHome, and at ±0.5 only for VLCS and OfficeHome.

## Known limitations

- Evidence is from four DomainBed benchmarks with ResNet-50/18, plus the ViT
  cross-architecture checks; larger shifts and more heterogeneous settings are
  not covered.
- The alignment study covers only two PACS domains over a short SWAD window.
- Hyperparameters and the layer choice follow DomainBed/SWAD conventions; the
  conclusions may depend on these.
- Theoretical bounds explain structure rather than predict exact accuracy.

## How to verify the claims yourself

```bash
# recompute every cell, delta and worst-domain gap from the raw runs
python scripts/collect_tables.py --root results
# inspect the per-run diagnostics (aug trigger fraction, gate, consistency KL)
python - <<'PY'
import json, glob
for fn in sorted(glob.glob('results/pacs/cifa/*.json'))[:5]:
    r = json.load(open(fn))
    print(fn.split('/')[-1],
          'trigger=%.2f gate=%.2f kl=%.3f' % (
          r.get('aug_trigger_fraction', float('nan')),
          r.get('aug_mean_gate', float('nan')),
          r.get('aug_mean_consistency_kl', float('nan'))))
PY
```
