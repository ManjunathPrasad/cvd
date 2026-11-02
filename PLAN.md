# CVD Prediction in Indian Patients — Implementation Plan

**Paper:** *A Comparative Study of Interpretable Machine Learning for Cardiovascular Disease Prediction in Indian Patients: Test Availability and Cross-Population Transfer* (final-year project, Dept. of AIML, MITE; target: ICIEURS 2026)

**Version 2 (2 Nov 2026).** This version replaces the first PLAN.md. It matches Section 3 of the paper draft. If you change anything here while running experiments, update Section 3 of the paper to match.

**Deadlines:**
- 9 Nov 2026: project presentation. Needs RC1, RC2 and the demo.
- Dec 2026 to Jan 2027: RC3, then finish the paper.

**How to use this file:** put it in the root of an empty repository as `PLAN.md`, and start with the kickoff prompt in Section 11. Work one stage at a time and review each stage before starting the next.

---

## 0. Ground rules

1. Work **stage by stage** (Section 8). At the end of each stage: run the scripts and tests, print the key numbers, commit with the message `stage-N: <summary>`, and **stop for review**.
2. **Never invent data, metrics or citations.** Every number in a table, figure, report or the app must be written by a script in this repo to `results/`.
3. If a data file is missing, or its columns or codes do not match Section 2, **stop and ask**. Do not download or create a substitute.
4. Every step that learns from data (imputation, encoding, scaling, any resampling, threshold choice) goes **inside a scikit-learn or imbalanced-learn `Pipeline`** and is fitted on training folds only.
5. Use `SEED = 42` everywhere. Save `pip freeze` output to `results/environment.txt`.
6. Keep the code simple enough for students to explain every file in a viva: plain functions, one YAML config, docstrings and type hints on public functions.
7. The README and the demo must state: *research prototype, not for clinical use.*

**Do not:**
- report accuracy alone;
- resample (SMOTE or anything else) before splitting;
- tune hyperparameters or choose thresholds on the test set;
- report training-set metrics as results;
- use CNN, LSTM or other sequence models (the data has no order).

---

## 1. Research questions

| RC | Question | Experiment | Paper output |
|---|---|---|---|
| RC1 | Which interpretable model predicts CVD best on Indian patients, under a leakage-free protocol? | E1 | Table 4, Figures 1–4 |
| RC2 | How much performance is kept when the model may use only the attributes available at each level of clinical testing? | E2 | Table 5, Figure 5 |
| RC3 | How well do models transfer between Indian and Western cohorts, and how much does recalibration recover? | E3 | Table 6 |

- **Task:** binary classification, heart disease present (1) or absent (0).
- **Primary metric:** ROC-AUC.
- **Secondary metrics:** PR-AUC, sensitivity, specificity, F1-score, MCC and Brier score, plus calibration intercept and slope.

---

## 2. Data

### 2.1 Indian dataset (RC1, RC2, RC3)

- **Source:** https://data.mendeley.com/datasets/dzz48mvjht (DOI 10.17632/dzz48mvjht.1). Check the licence on the page and cite the DOI.
- **Manual step:** a student downloads the CSV and saves it as `data/raw/mendeley_india.csv`. The code must not scrape the site.

Expected columns. Validate on load and stop if they differ.

| Column | Meaning | Documented coding | Tier |
|---|---|---|---|
| `patientid` | ID | drop before modelling | — |
| `age` | years | numeric | T1 |
| `gender` | sex | 1 male, 0 female | T1 |
| `chestpain` | chest-pain type | 0 typical angina, 1 atypical, 2 non-anginal, 3 asymptomatic | T1 |
| `restingBP` | resting blood pressure | mm Hg | T1 |
| `serumcholestrol` | serum cholesterol | mg/dL | T2 |
| `fastingbloodsugar` | fasting blood sugar | 1 if > 120 mg/dL | T2 |
| `restingrelectro` | resting ECG | 0 normal, 1 ST-T abnormality, 2 LVH | T2 |
| `maxheartrate` | maximum heart rate (exercise test) | numeric | T3 |
| `exerciseangia` | exercise-induced angina | 1 yes, 0 no | T3 |
| `oldpeak` | ST depression | numeric | T3 |
| `slope` | slope of peak-exercise ST segment | documented as 1 up, 2 flat, 3 down; **the data may also contain 0** | T3 |
| `noofmajorvessels` | vessels coloured by fluoroscopy | 0–3 | T4 |
| `target` | heart disease | 1 present, 0 absent | label |

**Data audit (Stage 1).** Write all findings to `results/data_audit.md`:
- Shape, data types, value counts of every categorical column, and min, median and max of every numeric column.
- Class balance of `target`.
- Exact duplicate rows, ignoring `patientid`: count them and drop them before splitting.
- Physiologically impossible values (`serumcholestrol == 0`, `restingBP == 0`): count them and set them to missing.
- Codes outside the documented coding, especially `slope == 0`: report them, keep them as their own category for RC1 and RC2, and **ask before choosing a mapping for RC3**.

### 2.2 UCI Heart Disease, four centres (RC3 only)

- **Source:** https://archive.ics.uci.edu/dataset/45/heart-disease (DOI 10.24432/C52P4X).
- **Manual step:** a student downloads the zip and copies these files into `data/external/`:
  - `processed.cleveland.data` (303 rows)
  - `processed.hungarian.data` (294 rows)
  - `processed.switzerland.data` (123 rows)
  - `processed.va.data` (200 rows)
- **Do not use** the Kaggle "combined" files or Statlog. They contain duplicated Cleveland records.

**Format:** no header row; `?` means missing. The 14 columns, in order, are: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num`.

**Harmonisation** (`src/data/harmonise.py`, unit-tested; write the full mapping to `results/harmonisation_table.md`):

| UCI column | Becomes | Rule |
|---|---|---|
| `sex` | `gender` | same coding |
| `cp` 1–4 | `chestpain` 0–3 | subtract 1, then check the meaning matches Section 2.1 |
| `trestbps` | `restingBP` | 0 → missing |
| `chol` | `serumcholestrol` | 0 → missing (Switzerland codes all cholesterol as 0) |
| `fbs`, `restecg`, `exang`, `oldpeak` | Indian names | same coding; verify the value ranges |
| `thalach` | `maxheartrate` | — |
| `slope` 1–3 | `slope` | align with the Indian coding agreed in Stage 1 |
| `num` 0–4 | `target` | `num > 0` → 1 |
| `ca`, `thal` | dropped | `ca` is missing outside Cleveland; `thal` is not in the Indian data |

RC3 uses the **11 attributes shared by all cohorts**, which excludes `noofmajorvessels`. A sensitivity analysis between India and Cleveland may use all 12.

---

## 3. Models (nine)

| Group | Models | Preprocessing |
|---|---|---|
| Reference | Logistic regression | median/mode imputation, one-hot encoding of `chestpain`, `restingrelectro` and `slope`, standard scaling |
| Tree ensembles | Random forest, XGBoost, LightGBM, CatBoost | imputation only; CatBoost receives `cat_features` |
| Foundation model | TabPFN (`pip install tabpfn`) | imputation only; default settings, no tuning. If installation or the weights licence blocks, skip it and log why. |
| Interpretable | Explainable boosting machine (`interpret`) | imputation only |
| Ensembles | Soft voting and stacking (logistic-regression meta-learner, `cv=5`) | built from the three best single models by CV ROC-AUC |

**Class imbalance:** if the minority class is at least 40% of rows, do nothing. Otherwise use `class_weight="balanced"` or `scale_pos_weight`.

**Tuning:** Optuna with 40 trials per model, or `RandomizedSearchCV`; objective ROC-AUC. Search spaces go in `configs/search_spaces.yaml`.

---

## 4. Evaluation protocol (paper Section 3.4)

1. **Split:** stratified 80/20 split of the Indian data (seed 42) into `dev` and `test`. The test set is used once, in Stage 4.
2. **Nested cross-validation on `dev`:** outer stratified 5-fold, inner stratified 5-fold for tuning. Report mean ± SD of all metrics in `results/cv_metrics.csv`.
3. **Threshold:** maximise Youden's J on the inner validation predictions. Also report the threshold that gives sensitivity ≥ 0.90.
4. **Final models:** tune on all of `dev`, refit, and evaluate once on `test`. Report every metric with a 95% bootstrap CI (1,000 resamples) in `results/test_metrics.csv`.
5. **Model comparison:** paired bootstrap of the test-AUC difference (best model vs logistic regression, best model vs random forest).
6. **Calibration:** Brier score, calibration intercept and slope, and calibration curves.
7. **Sanity checks** (`results/sanity.md`), required if any test AUC is ≥ 0.98:
   - near-duplicate rows shared between `dev` and `test`;
   - label-permutation test (AUC should fall to about 0.5);
   - removal of single attributes one at a time.

---

## 5. Experiments

### E1: Model comparison (RC1)

- Train all nine models on the 12 attributes and evaluate them as in Section 4.
- **Explanations:** SHAP for the best tree ensemble (beeswarm and bar plots, dependence plots for the top 3 attributes, waterfall plots for one true positive and one false negative). Compare the top attributes with the logistic-regression coefficients and the EBM shape functions in `results/feature_importance_agreement.csv`.
- **Outputs:**
  - Table 4: `results/table4_models.csv` and `.tex`.
  - Figures 1–3: ROC, PR and calibration curves.
  - Figure 4: SHAP summary.

### E2: Test availability (RC2)

- Feature tiers are cumulative:
  - T1 = `age, gender, chestpain, restingBP`
  - T2 = T1 + `serumcholestrol, fastingbloodsugar, restingrelectro`
  - T3 = T2 + `maxheartrate, exerciseangia, oldpeak, slope`
  - T4 = T3 + `noofmajorvessels`
- Retrain the best four E1 models on each tier with the same protocol. Report the change in ROC-AUC vs T4 with paired bootstrap 95% CIs.
- **Decision-curve analysis:** net benefit for thresholds from 0.05 to 0.50, compared with treating all and treating none. Report net benefit at a pre-specified threshold of 0.20.
- **Counterfactuals:** DiCE (`dice-ml`) on the best T2 model, with only `restingBP`, `serumcholestrol` and `fastingbloodsugar` allowed to change. Give 3 examples for patients predicted positive.
- **Outputs:**
  - Table 5: `results/table5_tiers.csv` and `.tex`.
  - Figure 5: decision curves.
  - `results/counterfactual_examples.md`.

### E3: Cross-population transfer (RC3, Dec 2026)

- Use the 11 shared attributes (Section 2.2).
- **Transfer directions:**
  - India → each UCI cohort;
  - pooled UCI → India;
  - leave-one-cohort-out over all five cohorts.
- **Measures:** internal vs external ROC-AUC, calibration slope and Brier score.
- **Adaptation:** Platt and isotonic recalibration, and refitting with 10% and 20% labelled target data. Each is repeated over 20 random draws; the remaining target data is used for testing.
- **Fairness:** sensitivity, specificity and calibration by sex and by age band (< 45, 45–60, > 60), plus the equalised-odds difference.
- **Output:** Table 6, `results/table6_transfer.csv` and `.tex`.

---

## 6. Repository structure

```
cvd-india/
├── PLAN.md  README.md  requirements.txt
├── configs/        config.yaml (paths, seed, tiers, split), search_spaces.yaml
├── data/raw/       mendeley_india.csv           (manual download)
├── data/external/  processed.*.data             (manual download, RC3)
├── src/
│   ├── data/       load.py, audit.py, harmonise.py
│   ├── features/   preprocess.py   # build_preprocessor(model_family, features)
│   ├── models/     zoo.py, tune.py
│   ├── eval/       nested_cv.py, metrics.py, bootstrap.py, calibration.py, dca.py, plots.py, sanity.py
│   ├── explain/    shap_analysis.py, ebm_analysis.py, counterfactuals.py
│   └── run_e1.py, run_e2.py, run_e3.py, run_all.py
├── app/streamlit_app.py
├── tests/
└── results/  figures/  models/  reports/
```

**Dependencies:** pandas, numpy, scikit-learn, imbalanced-learn, xgboost, lightgbm, catboost, optuna, tabpfn, interpret, shap, dice-ml, matplotlib, seaborn, streamlit, joblib, pyyaml, pytest. Record the installed versions in `results/environment.txt`.

---

## 7. Streamlit demo (for 9 Nov)

- **Model choice:** a toggle between the full T4 model and the screening T2 model.
- **Inputs:** plain-language labels with units and dropdowns for coded fields.
- **Output:**
  - the predicted probability;
  - an illustrative risk band (Low < 0.30, Moderate 0.30–0.60, High > 0.60), labelled as illustrative;
  - a SHAP waterfall plot;
  - for the T2 model, a DiCE counterfactual.
- **Sidebar:** data source, test metrics with CIs read from `results/` (never hard-coded), and limitations.
- **Banner:** research prototype, not for clinical use.

---

## 8. Stages and schedule

| Stage | Dates | Work | Done when |
|---|---|---|---|
| 0 | 2 Nov | Scaffold, virtual environment, config, README, smoke test | `pytest` passes |
| 1 | 2–3 Nov | Loader with column validation; data audit; EDA figures | `results/data_audit.md` reviewed by the students and the guide |
| 2 | 3–4 Nov | Preprocessing pipelines; all nine models; nested CV | `results/cv_metrics.csv` complete |
| 3 | 5 Nov | Final models on the test set; bootstrap CIs; calibration; sanity checks; SHAP and EBM (E1) | Table 4 and Figures 1–4 |
| 4 | 6 Nov | Feature tiers, decision curves, counterfactuals (E2) | Table 5, Figure 5, counterfactual examples |
| 5 | 7 Nov | Streamlit demo; `reports/presentation_results.md` | Demo runs locally |
| — | 8 Nov | Slides and rehearsal; no new code | — |
| — | 9 Nov | Presentation | — |
| 6 | Dec | Harmonise the UCI data; E3 transfer, recalibration and fairness | Table 6 |
| 7 | Dec–Jan | Fill paper Section 4; complete the TRIPOD+AI checklist in `reports/tripod_ai_checklist.md` | Draft ready for submission |

---

## 9. Definition of done

**For 9 Nov:**
- [ ] `python -m src.run_all --stages e1 e2` reproduces Tables 4 and 5 and Figures 1–5 from the raw CSV on a laptop CPU.
- [ ] No metric is typed by hand anywhere.
- [ ] `pytest` passes: column validation, tier definitions, and no leakage (preprocessors are fitted only on training indices).
- [ ] The demo runs locally.

**For the paper:**
- [ ] E3 complete.
- [ ] Every number in paper Section 4 traces to a file in `results/`.

---

## 10. Paper link

| Paper item | Produced by |
|---|---|
| Section 3.2 audit counts | `results/data_audit.md` |
| Table 3 | Section 2.1 of this plan (static) |
| Table 4, Figures 1–4 | E1 |
| Table 5, Figure 5 | E2 |
| Table 6 | E3 |
| Code-availability link (Section 3.6) | public repository URL, once published |

---

## 11. Stage prompts

**Kickoff:**
> Read PLAN.md fully. We are at Stage 0. Create the repository scaffold, requirements.txt, configs/config.yaml (including the four feature tiers from Section 5) and a README, plus a smoke test. Do not start Stage 1. Summarise what you created.

**Each later stage:**
> Continue with Stage N of PLAN.md. Follow the ground rules in Section 0. Run everything you write, show me the key numbers and any data problems, commit, and stop.

**If results look too good:**
> A test AUC is 0.98 or higher. Run the sanity checks in Section 4, step 7, and explain in results/sanity.md which attributes drive the result.
