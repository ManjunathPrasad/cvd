# Harmonisation of UCI Heart Disease to the Indian schema

| UCI column | Becomes | Rule |
|---|---|---|
| `sex` | `gender` | same coding (1 male, 0 female) |
| `cp` | `chestpain` | 1–4 → 0–3 (subtract 1): typical, atypical, non-anginal, asymptomatic |
| `trestbps` | `restingBP` | 0 → missing |
| `chol` | `serumcholestrol` | 0 → missing (Switzerland codes all cholesterol as 0) |
| `fbs` | `fastingbloodsugar` | same coding (1 if > 120 mg/dL) |
| `restecg` | `restingrelectro` | same coding (0 normal, 1 ST-T abnormality, 2 LVH) |
| `thalach` | `maxheartrate` | unchanged |
| `exang` | `exerciseangia` | same coding (1 yes, 0 no) |
| `oldpeak` | `oldpeak` | unchanged |
| `slope` | `slope` | 1 up, 2 flat, 3 down (kept); Indian codes mapped via config e3.indian_slope_mapping |
| `num` | `target` | num > 0 → 1 |
| `ca` | `noofmajorvessels` | kept only for the India–Cleveland 12-attribute sensitivity analysis |
| `thal` | `—` | dropped (not in the Indian data) |

## Cohort sizes and missingness after harmonisation

| Cohort | Rows | Disease rate | missing restingBP | missing serumcholestrol | missing slope | missing noofmajorvessels |
|---|---|---|---|---|---|---|
| Cleveland | 303 | 45.9% | 0% | 0% | 0% | 1% |
| Hungary | 294 | 36.1% | 0% | 8% | 65% | 99% |
| Switzerland | 123 | 93.5% | 2% | 100% | 14% | 96% |
| VA | 200 | 74.5% | 28% | 28% | 51% | 99% |
