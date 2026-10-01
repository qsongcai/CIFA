# Main results (results)

## Overall accuracy (%, leave-one-domain-out, ResNet-50)

| Dataset | ERM | SWAD | CIFA | Δ (CIFA−SWAD) | positive cells | worst-domain Δ |
|---|---|---|---|---|---|---|
| pacs | 84.38±1.27 | 87.13±0.46 | 88.30±0.28 | +1.17 | 15/20 | +3.71 (5/5) |
| vlcs | 76.04±0.91 | 79.03±0.58 | 79.29±0.73 | +0.26 | 11/20 | +0.59 (4/5) |
| officehome | 67.30±0.13 | 71.41±0.56 | 71.45±0.79 | +0.04 | 11/20 | +0.08 (3/5) |
| terra | 49.10±1.82 | 49.96±2.08 | 50.96±1.57 | +1.00 | 13/20 | +1.85 (4/5) |

## pacs — per target domain

| target domain | SWAD (mean±std) | CIFA (mean±std) | Δ mean | positive seeds |
|---|---|---|---|---|
| art_painting | 89.74±1.27 | 89.80±0.92 | +0.07 | 3/5 |
| cartoon | 82.14±0.97 | 82.88±1.26 | +0.74 | 3/5 |
| photo | 98.28±0.44 | 98.46±0.29 | +0.18 | 4/5 |
| sketch | 78.35±2.36 | 82.06±1.57 | +3.71 | 5/5 |

## vlcs — per target domain

| target domain | SWAD (mean±std) | CIFA (mean±std) | Δ mean | positive seeds |
|---|---|---|---|---|
| CALTECH | 98.57±0.36 | 98.57±0.57 | +0.00 | 3/5 |
| LABELME | 63.39±0.82 | 63.98±1.44 | +0.59 | 4/5 |
| PASCAL | 78.82±1.64 | 79.62±0.77 | +0.81 | 3/5 |
| SUN | 75.34±1.90 | 74.98±1.72 | -0.37 | 1/5 |

## officehome — per target domain

| target domain | SWAD (mean±std) | CIFA (mean±std) | Δ mean | positive seeds |
|---|---|---|---|---|
| Art | 69.00±0.64 | 69.24±1.28 | +0.24 | 3/5 |
| Clipart | 54.96±1.33 | 55.04±1.04 | +0.08 | 3/5 |
| Product | 80.00±0.46 | 80.00±0.99 | -0.01 | 3/5 |
| Real World | 81.68±0.40 | 81.53±0.59 | -0.15 | 2/5 |

## terra — per target domain

| target domain | SWAD (mean±std) | CIFA (mean±std) | Δ mean | positive seeds |
|---|---|---|---|---|
| location_38 | 46.48±7.49 | 48.01±5.97 | +1.53 | 3/5 |
| location_43 | 55.95±3.43 | 55.72±5.06 | -0.23 | 3/5 |
| location_46 | 40.79±2.40 | 41.42±3.35 | +0.63 | 3/5 |
| location_100 | 56.62±2.23 | 58.69±3.17 | +2.07 | 4/5 |

> `worst-domain Δ` compares, for each seed, CIFA against SWAD on the domain where SWAD is weakest that seed. OfficeHome shows no measurable gain (Δ≈0.04 overall); see docs/CLAIMS.md.
