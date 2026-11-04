# Data audit — Mendeley Indian dataset

Written by `src/data/audit.py`. Source: DOI 10.17632/dzz48mvjht.1.

## Shape and types

- Raw shape: 1000 rows × 14 columns
- Missing values in the raw file: 0
- Exact duplicate rows (ignoring `patientid`): 0 (dropped)
- Duplicate `patientid` values: 0
- Rows after cleaning: 1000

| column | dtype |
|---|---|
| patientid | int64 |
| age | int64 |
| gender | int64 |
| chestpain | int64 |
| restingBP | int64 |
| serumcholestrol | int64 |
| fastingbloodsugar | int64 |
| restingrelectro | int64 |
| maxheartrate | int64 |
| exerciseangia | int64 |
| oldpeak | float64 |
| slope | int64 |
| noofmajorvessels | int64 |
| target | int64 |

## Class balance (`target`)

| target | n | percent |
|---|---|---|
| 0.0 | 420.0 | 42.0 |
| 1.0 | 580.0 | 58.0 |

Minority class = 42.0%. Rule (PLAN.md Section 3): class weights only if below 40% → **no re-weighting**.

## Numeric columns (raw)

| column | min | median | max | mean | sd |
|---|---|---|---|---|---|
| age | 20.0 | 49.0 | 80.0 | 49.24 | 17.86 |
| restingBP | 94.0 | 147.0 | 200.0 | 151.75 | 29.97 |
| serumcholestrol | 0.0 | 318.0 | 602.0 | 311.45 | 132.44 |
| maxheartrate | 71.0 | 146.0 | 202.0 | 145.48 | 34.19 |
| oldpeak | 0.0 | 2.4 | 6.2 | 2.71 | 1.72 |

## Physiologically impossible values

- `serumcholestrol == 0`: 53 rows (5.3%) → set to missing; imputed inside the pipeline.
- `restingBP == 0`: 0 rows (0.0%) → set to missing; imputed inside the pipeline.
- Target rate among `serumcholestrol == 0` rows: 100.0% (overall 58.0%).

## Categorical columns: value counts and target rate

### `gender`

| gender | n | target_rate | documented |
|---|---|---|---|
| 0 | 235 | 0.566 | yes |
| 1 | 765 | 0.584 | yes |

### `fastingbloodsugar`

| fastingbloodsugar | n | target_rate | documented |
|---|---|---|---|
| 0 | 704 | 0.483 | yes |
| 1 | 296 | 0.811 | yes |

### `exerciseangia`

| exerciseangia | n | target_rate | documented |
|---|---|---|---|
| 0 | 502 | 0.6 | yes |
| 1 | 498 | 0.56 | yes |

### `chestpain`

| chestpain | n | target_rate | documented |
|---|---|---|---|
| 0 | 420 | 0.257 | yes |
| 1 | 224 | 0.688 | yes |
| 2 | 312 | 0.894 | yes |
| 3 | 44 | 0.886 | yes |

### `restingrelectro`

| restingrelectro | n | target_rate | documented |
|---|---|---|---|
| 0 | 454 | 0.392 | yes |
| 1 | 344 | 0.605 | yes |
| 2 | 202 | 0.96 | yes |

### `slope`

| slope | n | target_rate | documented |
|---|---|---|---|
| 0 | 180 | 0.0 | **NO** |
| 1 | 299 | 0.237 | yes |
| 2 | 322 | 0.963 | yes |
| 3 | 199 | 1.0 | yes |

### `noofmajorvessels`

| noofmajorvessels | n | target_rate | documented |
|---|---|---|---|
| 0 | 275 | 0.2 | yes |
| 1 | 344 | 0.587 | yes |
| 2 | 265 | 0.853 | yes |
| 3 | 116 | 0.836 | yes |

### `target`

| target | n | target_rate | documented |
|---|---|---|---|
| 0 | 420 | 0.0 | yes |
| 1 | 580 | 1.0 | yes |

## Codes outside the documented coding

- `slope == 0`: 180 rows. Kept as its own category for RC1 and RC2.
- Target rate when `slope == 0`: 0.0%. **Decision needed before RC3:** how to map Indian slope codes onto UCI slope 1–3 (set `e3.indian_slope_mapping` in configs/config.yaml).

## Figures

- `results/figures/eda_numeric.png`: distributions of numeric attributes by target
- `results/figures/eda_categorical.png`: target rate by category

