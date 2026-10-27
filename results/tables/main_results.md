# Main results (results)

## Overall accuracy (%, leave-one-domain-out, ResNet-50)

| Dataset | ERM | SWAD | CIFA | Δ (CIFA−SWAD) | positive cells | worst-domain Δ |
|---|---|---|---|---|---|---|
| pacs | 84.27±1.93 | 88.16±0.62 | 87.60±0.92 | -0.57 | 14/40 | -0.93 (5/10) |
| vlcs | 75.77±1.20 | 79.09±0.85 | 79.15±0.79 | +0.06 | 18/40 | -0.18 (5/10) |
| officehome | 67.24±0.57 | 71.45±0.42 | 71.34±0.61 | -0.11 | 18/40 | -0.26 (4/10) |
| terra | 50.09±2.24 | 51.17±2.02 | 51.80±2.12 | +0.63 | 24/40 | +2.23 (7/10) |

## pacs — per target domain

| target domain | SWAD (mean±std) | CIFA (mean±std) | Δ mean | positive seeds |
|---|---|---|---|---|
| art_painting | 91.13±1.44 | 89.91±0.87 | -1.22 | 1/10 |
| cartoon | 83.22±1.28 | 82.70±1.61 | -0.52 | 4/10 |
| photo | 98.29±0.32 | 98.40±0.24 | +0.10 | 4/10 |
| sketch | 80.02±1.48 | 79.39±3.57 | -0.62 | 5/10 |

## vlcs — per target domain

| target domain | SWAD (mean±std) | CIFA (mean±std) | Δ mean | positive seeds |
|---|---|---|---|---|
| CALTECH | 98.47±0.59 | 98.59±0.55 | +0.12 | 5/10 |
| LABELME | 63.84±1.64 | 63.66±1.29 | -0.18 | 5/10 |
| PASCAL | 78.36±2.30 | 79.04±1.33 | +0.68 | 5/10 |
| SUN | 75.66±1.50 | 75.29±1.31 | -0.37 | 3/10 |

## officehome — per target domain

| target domain | SWAD (mean±std) | CIFA (mean±std) | Δ mean | positive seeds |
|---|---|---|---|---|
| Art | 68.61±0.67 | 68.40±1.29 | -0.21 | 3/10 |
| Clipart | 55.59±1.23 | 55.33±0.89 | -0.26 | 4/10 |
| Product | 79.94±0.42 | 80.03±0.81 | +0.09 | 6/10 |
| Real World | 81.66±0.52 | 81.60±0.44 | -0.06 | 5/10 |

## terra — per target domain

| target domain | SWAD (mean±std) | CIFA (mean±std) | Δ mean | positive seeds |
|---|---|---|---|---|
| location_38 | 46.74±5.78 | 48.50±4.90 | +1.76 | 6/10 |
| location_43 | 58.16±3.74 | 57.73±4.50 | -0.43 | 6/10 |
| location_46 | 40.65±3.72 | 42.12±3.01 | +1.47 | 6/10 |
| location_100 | 59.14±3.70 | 58.86±5.75 | -0.27 | 6/10 |

> `worst-domain Δ` compares, for each seed, each arm's accuracy on its own weakest target domain, then differenced. The PACS row uses the authoritative unified-code batch; the contrast is a statistical tie, and the other benchmarks read as null or near-null. See docs/CLAIMS.md.
