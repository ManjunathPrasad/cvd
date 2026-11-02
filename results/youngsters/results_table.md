# Prediction results — heart_attack_youngsters_india.csv

Research prototype, not for clinical use. Every number below is written by `src/run_youngsters.py`; nothing is typed by hand.

- Rows: 10000; positive class (heart-attack likelihood = Yes): 20.38%
- Split: stratified 80/20 (seed 42). Test set used once.
- Nested CV on dev: outer 2-fold, inner 2-fold, Optuna TPE; trials per model: LogisticRegression 2, RandomForest 2, XGBoost 2, LightGBM 2, CatBoost 2, EBM 0, Dummy(prior) 0
- Threshold: Youden's J on inner out-of-fold predictions; 95% CIs from 1,000 bootstrap resamples.
- TabPFN not run (needs PyTorch and gated model weights in this environment).

## Held-out test set (n = 2000)

| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Sensitivity | Specificity | F1 | MCC | Brier | Cal. intercept | Cal. slope |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 0.522 (0.490–0.555) | 0.230 (0.203–0.266) | 0.782 (0.742–0.820) | 0.242 (0.222–0.262) | 0.330 | 0.022 | 0.186 | -0.15 | 0.06 |
| SoftVoting(top3) | 0.521 (0.488–0.553) | 0.232 (0.203–0.266) | 0.804 (0.766–0.842) | 0.205 (0.187–0.225) | 0.328 | 0.009 | 0.183 | -0.60 | 0.17 |
| RandomForest | 0.518 (0.484–0.549) | 0.216 (0.193–0.246) | 0.603 (0.555–0.650) | 0.430 (0.405–0.454) | 0.315 | 0.027 | 0.223 | -1.16 | 0.26 |
| LightGBM | 0.516 (0.481–0.548) | 0.233 (0.203–0.268) | 0.292 (0.249–0.332) | 0.735 (0.713–0.756) | 0.251 | 0.024 | 0.186 | -0.39 | 0.04 |
| EBM | 0.514 (0.484–0.545) | 0.220 (0.195–0.251) | 0.301 (0.260–0.345) | 0.726 (0.704–0.747) | 0.254 | 0.024 | 0.162 | -0.00 | 0.74 |
| Stacking(top3) | 0.511 (0.479–0.544) | 0.214 (0.191–0.244) | 0.904 (0.874–0.933) | 0.091 (0.078–0.106) | 0.332 | -0.006 | 0.162 | -0.00 | 0.78 |
| CatBoost | 0.509 (0.479–0.542) | 0.203 (0.181–0.229) | 0.039 (0.021–0.059) | 0.958 (0.947–0.967) | 0.065 | -0.006 | 0.202 | -0.33 | 0.03 |
| LogisticRegression | 0.503 (0.475–0.534) | 0.216 (0.194–0.249) | 0.002 (0.000–0.008) | 0.999 (0.997–1.000) | 0.005 | 0.012 | 0.251 | -1.37 | 0.17 |
| Dummy(prior) | 0.500 (0.500–0.500) | 0.204 (0.185–0.221) | 0.000 (0.000–0.000) | 1.000 (1.000–1.000) | 0.000 | 0.000 | 0.162 | 0.00 | 0.66 |

## Nested cross-validation on dev (mean ± SD over outer folds)

| Model | ROC-AUC | PR-AUC | Sensitivity | Specificity | MCC | Brier |
|---|---|---|---|---|---|---|
| Stacking(top3) | 0.509 ± 0.002 | 0.208 ± 0.003 | 0.560 ± 0.161 | 0.459 ± 0.165 | 0.016 ± 0.003 | 0.162 ± 0.000 |
| XGBoost | 0.506 ± 0.011 | 0.209 ± 0.008 | 0.929 ± 0.053 | 0.072 ± 0.061 | -0.000 ± 0.014 | 0.192 ± 0.000 |
| SoftVoting(top3) | 0.502 ± 0.014 | 0.208 ± 0.006 | 0.789 ± 0.087 | 0.223 ± 0.093 | 0.011 ± 0.005 | 0.178 ± 0.001 |
| LightGBM | 0.501 ± 0.015 | 0.207 ± 0.003 | 0.875 ± 0.125 | 0.131 ± 0.136 | 0.004 ± 0.012 | 0.187 ± 0.002 |
| Dummy(prior) | 0.500 ± 0.000 | 0.204 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.162 ± 0.000 |
| RandomForest | 0.499 ± 0.014 | 0.203 ± 0.004 | 0.253 ± 0.190 | 0.754 ± 0.170 | 0.005 ± 0.019 | 0.233 ± 0.013 |
| CatBoost | 0.497 ± 0.011 | 0.205 ± 0.006 | 0.750 ± 0.224 | 0.244 ± 0.249 | -0.014 ± 0.033 | 0.222 ± 0.035 |
| EBM | 0.491 ± 0.002 | 0.199 ± 0.000 | 0.221 ± 0.313 | 0.770 ± 0.325 | -0.007 ± 0.010 | 0.164 ± 0.002 |
| LogisticRegression | 0.491 ± 0.001 | 0.201 ± 0.001 | 0.357 ± 0.066 | 0.627 ± 0.059 | -0.014 ± 0.006 | 0.252 ± 0.000 |

## Paired bootstrap of test ROC-AUC

| Comparison | ΔAUC | 95% CI | p |
|---|---|---|---|
| XGBoost - LogisticRegression | +0.020 | -0.022 to +0.057 | 0.376 |
| XGBoost - RandomForest | +0.005 | -0.025 to +0.036 | 0.732 |
| XGBoost - Dummy(prior) | +0.022 | -0.010 to +0.055 | 0.174 |

## Sanity checks

- Label-permutation test (LightGBM trained on shuffled dev labels): test ROC-AUC = 0.477.
- Features with univariate association to the label at Bonferroni p < 0.05: 0 of 26. Smallest raw p-value: 0.00469 (Stress Level).

Run time: 7.2 min.
