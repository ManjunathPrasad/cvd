# CVD prediction — interpretable ML benchmark

> **Research prototype, not for clinical use.**

Final-year project, Dept. of AIML, MITE. See `PLAN.md` for the full research plan (RC1–RC3).

## Current experiment: `heart_attack_youngsters_india.csv`

A leakage-free benchmark of tabular ML models on the 10,000-row "heart attack in youngsters (India)"
CSV (ages 18–35, 25 attributes, label `Heart Attack Likelihood`).

**Note:** this is *not* the 1,000-patient Mendeley dataset (DOI 10.17632/dzz48mvjht.1) that PLAN.md
and the paper describe. The pipeline here is separate (`src/run_youngsters.py`) so the planned
RC1–RC3 work on the Mendeley data is unaffected.

### Protocol
- Stratified 80/20 dev/test split, seed 42; the test set is used once.
- Nested CV on dev (outer 5-fold, inner 3-fold) with Optuna TPE tuning on ROC-AUC.
- Encoding/scaling inside scikit-learn Pipelines fitted on training folds only.
- Class imbalance (20% positive) handled with class weights; no resampling.
- Threshold = Youden's J on inner out-of-fold predictions; a sensitivity ≥ 0.90 threshold is also reported.
- Test metrics with 1,000-resample bootstrap 95% CIs; paired bootstrap for AUC differences.
- Models: logistic regression, random forest, XGBoost, LightGBM, CatBoost, explainable boosting
  machine, soft-voting and stacking ensembles of the top 3, plus a prior-only dummy baseline.

### Run
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp /path/to/heart_attack_youngsters_india.csv data/raw/
.venv/bin/python -m src.run_youngsters        # full run (add --quick for a smoke test)
.venv/bin/python -m src.plots                 # re-draw figures from results/
.venv/bin/python -m pytest -q
```

### Outputs (`results/youngsters/`)
| File | Content |
|---|---|
| `results_table.md` | Main results table (test metrics with CIs, nested-CV metrics, comparisons, sanity checks) |
| `test_metrics.csv` / `cv_metrics.csv` | Machine-readable metrics |
| `test_predictions.csv` | Per-patient test-set probabilities for every model |
| `model_comparison.csv` | Paired bootstrap ΔAUC |
| `feature_association.csv` | Univariate χ² / Mann–Whitney tests of each attribute vs the label |
| `figures/` | ROC, PR, calibration, AUC forest plot, threshold metrics, confusion matrix, feature association |
| `best_params.json`, `environment.txt` | Tuned hyperparameters and package versions |
