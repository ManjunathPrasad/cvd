# Interpretable ML for cardiovascular disease prediction in Indian patients

> **Research prototype, not for clinical use.**

Final-year project, Dept. of AIML, MITE (target: ICIEURS 2026). See `PLAN.md` for the full research plan.

| RC | Question | Run | Outputs |
|---|---|---|---|
| RC1 | Which interpretable model predicts CVD best, under a leakage-free protocol? | `python -m src.run_e1` | Table 4, Figures 1–4, sanity checks, SHAP/EBM explanations |
| RC2 | How much performance is kept with only the tests available at each level of care? | `python -m src.run_e2` | Table 5, Figure 5, counterfactual examples |
| RC3 | How well do models transfer between Indian and Western cohorts? | `python -m src.run_e3` | Table 6, adaptation and fairness results |

## Data (manual download)
- **Indian dataset** (RC1–RC3): Mendeley Data, DOI [10.17632/dzz48mvjht.1](https://data.mendeley.com/datasets/dzz48mvjht). Save as `data/raw/mendeley_india.csv`.
- **UCI Heart Disease** (RC3 only): DOI [10.24432/C52P4X](https://archive.ics.uci.edu/dataset/45/heart-disease). Copy `processed.cleveland.data`, `processed.hungarian.data`, `processed.switzerland.data` and `processed.va.data` into `data/external/`.

## Quick start
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m src.run_all --stages audit e1 e2          # RC1 + RC2 from the raw CSV
.venv/bin/python -m src.report_presentation                  # reports/presentation_results.md
.venv/bin/streamlit run app/streamlit_app.py                  # demo
```
Add `--quick` to `run_all` for a 2-minute smoke run (2 tuning trials, 3-fold CV; writes to `results/_quick/`).

RC3 additionally needs the UCI files and an agreed mapping for the Indian `slope` codes
(`e3.indian_slope_mapping` in `configs/config.yaml`; see `results/data_audit.md`). Then:
```bash
.venv/bin/python -m src.run_all --stages e3
```

## Protocol (PLAN.md Section 4)
- Audit: exact duplicates dropped; `serumcholestrol == 0` and `restingBP == 0` set to missing; undocumented codes reported.
- Stratified 80/20 dev/test split (seed 42); the test set is used once.
- Nested CV on dev: outer 5-fold, inner 5-fold, Optuna (40 trials) on ROC-AUC.
- Imputation, encoding and scaling are inside scikit-learn Pipelines fitted on training folds only.
- Threshold: Youden's J on inner out-of-fold predictions; a sensitivity ≥ 0.90 threshold is also reported.
- Test metrics with 1,000-resample bootstrap 95% CIs; paired bootstrap for AUC differences.
- Models: logistic regression, random forest, XGBoost, LightGBM, CatBoost, TabPFN, EBM, soft voting and stacking.
  TabPFN is skipped when its gated weights are unavailable (reason in `results/tabpfn_status.md`).

## Repository layout
```
configs/      config.yaml (paths, seed, tiers, CV, tuning), search_spaces.yaml
src/data/     load.py (validation, cleaning, split), audit.py, harmonise.py (UCI)
src/features/ preprocess.py — build_preprocessor(model_family, features)
src/models/   zoo.py (all models), tune.py (Optuna)
src/eval/     metrics.py, nested_cv.py, dca.py, sanity.py, plots.py, report.py
src/explain/  shap_analysis.py, ebm_analysis.py, counterfactuals.py
src/          run_e1.py, run_e2.py, run_e3.py, run_all.py, report_presentation.py
app/          streamlit_app.py
tests/        pytest suite (validation, tiers, leakage, harmonisation, metrics)
results/      every number and figure (written by scripts only)
models/       demo_T4.joblib, demo_T2.joblib (used by the demo)
reports/      presentation_results.md, tripod_ai_checklist.md
```

## Negative control: `heart_attack_youngsters_india.csv`
`src/run_youngsters.py` benchmarks the same models on a 10,000-row synthetic-looking CSV whose label is unrelated to
its features (all test ROC-AUC ≈ 0.50; results in `results/youngsters/`). It shows the protocol does not produce
good-looking numbers from noise.
