# Sanity checks (PLAN.md Section 4, step 7)

Highest test ROC-AUC: 1.000 (SoftVoting). Trigger: 0.98. **Checks required.**

Checks use `CatBoost` (best single model by nested-CV ROC-AUC) with its tuned hyperparameters.

## 1. Near-duplicate rows shared between dev and test

- Test rows identical to a dev row (all 12 attributes): 0
- Test rows within 0.05 × the typical nearest-neighbour distance of a dev row: 0
- Median distance test→nearest dev row: 1.770; within dev: 1.759 (standardised units)

## 2. Label-permutation test

- Test ROC-AUC after training on shuffled dev labels (20 repeats): mean 0.497, range 0.302–0.686 (expected ≈ 0.5)

## 3. Removing one attribute at a time

| removed | test_auc | delta_vs_full |
|---|---|---|
| slope | 0.995 | -0.004 |
| chestpain | 0.998 | -0.001 |
| restingBP | 0.998 | -0.001 |
| noofmajorvessels | 0.999 | -0.001 |
| oldpeak | 0.999 | -0.001 |
| gender | 0.999 | -0.0 |
| serumcholestrol | 0.999 | -0.0 |
| maxheartrate | 0.999 | 0.0 |
| restingrelectro | 0.999 | 0.0 |
| fastingbloodsugar | 1.0 | 0.0 |
| exerciseangia | 1.0 | 0.0 |
| age | 1.0 | 0.0 |

## 4. Attributes that nearly determine the label (dev set)

`slope`:

| slope | n | disease_rate |
|---|---|---|
| 0 | 144 | 0.0 |
| 1 | 243 | 0.251 |
| 2 | 250 | 0.96 |
| 3 | 163 | 1.0 |

`chestpain`:

| chestpain | n | disease_rate |
|---|---|---|
| 0 | 340 | 0.247 |
| 1 | 177 | 0.706 |
| 2 | 250 | 0.908 |
| 3 | 33 | 0.848 |

`noofmajorvessels`:

| noofmajorvessels | n | disease_rate |
|---|---|---|
| 0 | 222 | 0.203 |
| 1 | 264 | 0.587 |
| 2 | 224 | 0.848 |
| 3 | 90 | 0.822 |

`restingrelectro`:

| restingrelectro | n | disease_rate |
|---|---|---|
| 0 | 361 | 0.393 |
| 1 | 279 | 0.606 |
| 2 | 160 | 0.956 |

Rows with cholesterol recorded as 0 (set to missing before modelling): disease rate 100.0% (n = 43).

## Interpretation (generated from the numbers above)

- The largest single-attribute drop comes from removing `slope` (ΔAUC -0.004).
- Permutation AUC ≈ 0.50 shows the pipeline does not leak labels; 0 exact dev–test duplicates.

