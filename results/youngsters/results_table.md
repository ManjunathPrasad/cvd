# Prediction results — heart_attack_youngsters_india.csv

Research prototype, not for clinical use. Every number below is written by `src/run_youngsters.py`; nothing is typed by hand.

- Rows: 10000; positive class (heart-attack likelihood = Yes): 20.38%
- Split: stratified 80/20 (seed 42). Test set used once.
- Nested CV on dev: outer 5-fold, inner 3-fold, Optuna TPE; trials per model: LogisticRegression 20, RandomForest 10, XGBoost 20, LightGBM 20, CatBoost 8, EBM 0, Dummy(prior) 0
- Threshold: Youden's J on inner out-of-fold predictions; 95% CIs from 1,000 bootstrap resamples.
- TabPFN not run (needs PyTorch and gated model weights in this environment).

## Held-out test set (n = 2000)

| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Sensitivity | Specificity | F1 | MCC | Brier | Cal. intercept | Cal. slope |
|---|---|---|---|---|---|---|---|---|---|
| CatBoost | 0.518 (0.485–0.550) | 0.219 (0.194–0.249) | 0.525 (0.478–0.575) | 0.501 (0.478–0.526) | 0.302 | 0.021 | 0.229 | -1.18 | 0.09 |
| Stacking(top3) | 0.517 (0.484–0.548) | 0.222 (0.196–0.253) | 0.958 (0.938–0.976) | 0.029 (0.021–0.037) | 0.333 | -0.029 | 0.162 | 0.00 | 1.42 |
| SoftVoting(top3) | 0.513 (0.481–0.544) | 0.216 (0.191–0.246) | 0.743 (0.700–0.784) | 0.268 (0.246–0.289) | 0.323 | 0.010 | 0.219 | -1.10 | 0.11 |
| XGBoost | 0.511 (0.481–0.541) | 0.209 (0.187–0.237) | 0.728 (0.684–0.771) | 0.270 (0.248–0.293) | 0.318 | -0.002 | 0.219 | -1.08 | 0.05 |
| EBM | 0.510 (0.480–0.541) | 0.219 (0.195–0.252) | 1.000 (1.000–1.000) | 0.000 (0.000–0.000) | 0.339 | 0.000 | 0.162 | 0.00 | 1.35 |
| RandomForest | 0.506 (0.471–0.538) | 0.224 (0.196–0.255) | 0.647 (0.602–0.692) | 0.352 (0.327–0.376) | 0.310 | -0.001 | 0.214 | -1.08 | 0.17 |
| LogisticRegression | 0.503 (0.473–0.534) | 0.211 (0.189–0.242) | 0.865 (0.830–0.897) | 0.149 (0.132–0.165) | 0.334 | 0.016 | 0.250 | -1.36 | 0.27 |
| Dummy(prior) | 0.500 (0.500–0.500) | 0.204 (0.185–0.221) | 0.000 (0.000–0.000) | 1.000 (1.000–1.000) | 0.000 | 0.000 | 0.162 | 0.00 | 0.66 |
| LightGBM | 0.488 (0.457–0.517) | 0.195 (0.174–0.218) | 0.250 (0.209–0.291) | 0.712 (0.689–0.735) | 0.211 | -0.034 | 0.230 | -0.20 | -0.03 |

## Nested cross-validation on dev (mean ± SD over outer folds)

| Model | ROC-AUC | PR-AUC | Sensitivity | Specificity | MCC | Brier |
|---|---|---|---|---|---|---|
| CatBoost | 0.514 ± 0.020 | 0.214 ± 0.010 | 0.788 ± 0.252 | 0.215 ± 0.230 | 0.009 ± 0.027 | 0.233 ± 0.007 |
| SoftVoting(top3) | 0.514 ± 0.021 | 0.213 ± 0.008 | 0.548 ± 0.266 | 0.451 ± 0.274 | -0.001 ± 0.026 | 0.223 ± 0.015 |
| RandomForest | 0.513 ± 0.017 | 0.211 ± 0.010 | 0.607 ± 0.160 | 0.407 ± 0.165 | 0.013 ± 0.012 | 0.232 ± 0.019 |
| XGBoost | 0.509 ± 0.017 | 0.213 ± 0.009 | 0.433 ± 0.282 | 0.575 ± 0.288 | 0.006 ± 0.024 | 0.231 ± 0.019 |
| Stacking(top3) | 0.506 ± 0.013 | 0.212 ± 0.009 | 0.426 ± 0.425 | 0.576 ± 0.430 | 0.012 ± 0.027 | 0.162 ± 0.000 |
| EBM | 0.502 ± 0.021 | 0.205 ± 0.009 | 0.446 ± 0.240 | 0.553 ± 0.238 | -0.000 ± 0.015 | 0.163 ± 0.001 |
| Dummy(prior) | 0.500 ± 0.000 | 0.204 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.162 ± 0.000 |
| LightGBM | 0.499 ± 0.027 | 0.208 ± 0.016 | 0.633 ± 0.103 | 0.348 ± 0.092 | -0.016 ± 0.027 | 0.218 ± 0.019 |
| LogisticRegression | 0.496 ± 0.015 | 0.206 ± 0.008 | 0.563 ± 0.400 | 0.430 ± 0.399 | -0.005 ± 0.013 | 0.251 ± 0.001 |

## Paired bootstrap of test ROC-AUC

| Comparison | ΔAUC | 95% CI | p |
|---|---|---|---|
| CatBoost - LogisticRegression | +0.015 | -0.020 to +0.049 | 0.428 |
| CatBoost - RandomForest | +0.012 | -0.015 to +0.041 | 0.346 |
| CatBoost - Dummy(prior) | +0.018 | -0.015 to +0.050 | 0.256 |

## Sanity checks

- Label-permutation test (LightGBM trained on shuffled dev labels): test ROC-AUC = 0.506.
- Features with univariate association to the label at Bonferroni p < 0.05: 0 of 26. Smallest raw p-value: 0.00469 (Stress Level).

Run time: 21.6 min.
