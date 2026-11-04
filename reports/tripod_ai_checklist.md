# TRIPOD+AI checklist — working draft

Reporting guideline: Collins GS et al., *TRIPOD+AI statement*, BMJ 2024;385:e078378.
**Before submission, copy the official checklist (item numbers and exact wording) from the paper's supplement
and fill the page numbers.** This draft groups the topics and records where this repository provides each one.

| Topic | Where it is addressed | Status |
|---|---|---|
| Title identifies a prediction-model study, population and outcome | Paper title | To check |
| Structured abstract | Paper | To write |
| Background, rationale, existing models | Paper Sections 1–2 (literature survey) | Drafted |
| Objectives (development and/or evaluation) | Paper Section 1.1 (RC1–RC3) | Drafted |
| Data sources, dates, setting | `results/data_audit.md`; Mendeley DOI 10.17632/dzz48mvjht.1; UCI DOI 10.24432/C52P4X | Partly (dates/setting from source) |
| Eligibility and participants | `results/data_audit.md` (1,000 rows, 0 duplicates) | Done |
| Data preparation and pre-processing | `src/data/load.py`, `src/features/preprocess.py`; zeros → missing; imputation inside pipelines | Done |
| Outcome definition | `target` (PLAN.md Section 2.1); UCI `num > 0` (`src/data/harmonise.py`) | Done |
| Predictors: definition, timing, measurement | PLAN.md Section 2.1; tiers in `configs/config.yaml` | Done |
| Sample size | 1,000 patients, 580 events; dev 800 / test 200 (`results/data_audit.md`, `results/e1_summary.json`) | To justify in paper |
| Missing data handling | Median/mode imputation fitted on training folds; counts in `results/data_audit.md` | Done |
| Analytical methods: models, tuning, internal validation | `src/models/`, `src/eval/nested_cv.py`, `configs/search_spaces.yaml` | Done |
| Class imbalance | Minority 42% ≥ 40% → no re-weighting (`results/data_audit.md`) | Done |
| Fairness | E3 fairness by sex and age band (`src/run_e3.py`, `results/e3_fairness.csv`) | Needs UCI data |
| Model output and thresholds | Probabilities; Youden and sensitivity ≥ 0.90 thresholds (`results/test_metrics.csv`) | Done |
| Development vs evaluation data | 80/20 split, test used once; external cohorts in E3 | Done (E3 pending) |
| Performance measures with uncertainty | `results/table4_models.csv`, `table5_tiers.csv` (bootstrap 95% CIs) | Done |
| Calibration | Intercept, slope, curves (`results/test_metrics.csv`, `figures/fig3_calibration.png`) | Done |
| Clinical utility | Decision curves (`results/decision_curves.csv`, `figures/fig5_decision_curves.png`) | Done |
| Model updating / recalibration | E3 Platt, isotonic, refit (`results/e3_adaptation_summary.csv`) | Needs UCI data |
| Explainability | SHAP, EBM, LR coefficients (`results/feature_importance_agreement.csv`) | Done |
| Ethics approval | Public de-identified datasets; state institutional position | To write |
| Funding, conflicts of interest | Paper | To write |
| Protocol / registration | PLAN.md (version 2, 2 Nov 2026) | State in paper |
| Data availability | Public DOIs above | Done |
| Code availability | Public repository URL | Add when published |
| Patient and public involvement | Not done; state explicitly | To write |
| Results: participant flow and characteristics | `results/data_audit.md`, EDA figures | Done |
| Results: full model specification | `results/e1_summary.json` (final hyperparameters), `models/demo_*.joblib` | Done |
| Discussion: limitations (single centre, slope-0 artefact, external validity) | `results/sanity.md`, `results/data_audit.md` | To write |
| Usability / intended use | Demo marked research prototype, not for clinical use | Done |
