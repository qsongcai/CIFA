# Component ablation (results)

PACS, 5 variants × 4 target domains × 5 seeds = 100 runs.

| variant | overall mean±std | Δ vs full (overall) | Δ vs full (worst domain) |
|---|---|---|---|
| full | 88.36±0.48 | +0.00 | +0.00 |
| fullmoment | 88.59±0.67 | +0.23 | +0.09 |
| noguard | 87.34±1.27 | -1.01 | -3.36 |
| uniform | 87.88±0.32 | -0.47 | -1.28 |
| noswad | 85.93±1.70 | -2.42 | -1.36 |

> `fullmoment` (+0.23) is within run-to-run noise: the baseline/deviation split is what enables the content-preservation bound, not a source of accuracy. `noswad` is the largest drop (−2.42), i.e. weight averaging drives the overall level; the feature-moment term targets the worst domain.
