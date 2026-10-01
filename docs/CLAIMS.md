# Claims and scope

This document states, plainly, what the paper does and does not claim. The
theory is conditional and the empirical gains are modest on purpose; please read
the results in that light.

## What we claim

1. **A unified reading of two heuristic families.** Under an explicit
   content–style model, mixing only the *style deviation* of feature moments is
   a feature-level **Causal Invariant Transform (CIT)**. This gives one
   framework that covers feature-statistic mixing (MixStyle/DSU) and connects
   it to weight averaging (SWAD/DiWA).
2. **Approximate causal-content preservation.** Holding the content baseline
   and the normalized residual fixed while resampling the deviation keeps the
   causal content approximately invariant. The approximation error is bounded
   by (i) testable deviation conditions and (ii) an explicit **recomposition
   assumption** — it is not zero.
3. **A minimax risk bound with an explicit coverage gap.** The convex-hull
   minimax argument bounds risk over the augmentation family and states the
   coverage gap when that hull does not contain the target shift.
4. **Weight averaging as an approximate ensemble.** SWAD's dense average is a
   second-order approximation to the trajectory / augmentation-family
   aggregate; reading it this way needs an explicit alignment assumption.
5. **A plug-and-play, target-free operator.** CIFA never accesses the target
   domain; model selection uses only source-domain hold-out splits.

## What we do NOT claim

- **Not a uniform large win over SWAD.** On average CIFA *matches* SWAD. The
  consistent empirical effect is a **small gain on the worst domain** of PACS
  (+3.14 on Sketch), VLCS (+0.59) and Terra (+1.85). These are not large
  improvements.
- **OfficeHome is a null result.** Δ≈+0.04 overall with 11/20 positive cells;
  we report it as no measurable gain, not as a success.
- **Not causal effect recovery.** Unlike instrumental-variable / anchor
  approaches, the goal is prediction robustness under shift, not recovering a
  structural causal coefficient. When an anchor carries a direct effect on the
  label it is not a valid instrument; the relevant check here is worst-domain
  prediction, not parameter recovery.
- **The guarantees are conditional.** They hold only when the content–style
  decomposition, independence, recomposition and coverage assumptions hold.
- **The "style deviation is class-conditionally separable" statement is a
  post-hoc hypothesis**, offered as a mechanism, not an independently verified
  causal fact.

## Assumptions to check

| Assumption | Role | How it is checked |
|---|---|---|
| Content–style generation: C determines the label, S indexes the domain, C⊥S | foundation of the CIT reading | content–style model; channel-moment decomposition |
| Channel moments split into a content baseline b and style deviation d=m−b | defines what may be mixed | EMA class prototypes in `cifa_aug.py`; ablation `fullmoment` |
| Recomposition: replacing d leaves causal content approximately fixed | bounds transform error | consistency KL + guard; ablation `noguard` |
| Convex-hull coverage of the target augmentation | risk bound applies | coverage gap stated explicitly |
| Alignment for weight averaging | SWAD-as-ensemble reading | stated assumption; ablation `noswad` |

## Known limitations

- Evidence is from four DomainBed benchmarks (five seeds each, ten for PACS);
  larger shifts and
  more heterogeneous datasets are not covered (a DomainNet pilot was negative
  and is not included).
- Hyperparameters and the layer choice follow DomainBed/SWAD conventions; the
  conclusions may depend on these.
- Theoretical bounds are not numerically tight against the observed errors; they
  explain structure rather than predict exact accuracy.

## How to verify the claims yourself

```bash
# recompute every cell, delta and worst-domain gap from the raw runs
python scripts/collect_tables.py --root results
# inspect the per-run diagnostics (aug trigger fraction, mean gate, consistency KL)
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
