"""Leakage-free benchmark of tabular ML models on the heart-attack-youngsters-India CSV.

Protocol (mirrors PLAN.md Section 4):
  1. Stratified 80/20 split (seed 42) into dev and test. Test is touched once, at the end.
  2. Nested CV on dev: outer stratified 5-fold, inner stratified K-fold with Optuna (TPE) tuning,
     objective ROC-AUC. All encoding/scaling is inside a Pipeline fitted on training folds only.
  3. Threshold = max Youden's J on inner out-of-fold predictions (never on the evaluation fold).
  4. Final models are tuned on all of dev, refitted, and evaluated once on test with
     1,000-resample bootstrap 95% CIs. Paired bootstrap for AUC differences.
  5. Soft-voting and stacking ensembles built from the three best single models by CV ROC-AUC.

Usage:  python -m src.run_youngsters [--quick]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from src.metrics import (
    all_metrics,
    bootstrap_ci,
    paired_bootstrap_auc,
    sensitivity_threshold,
    youden_threshold,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "heart_attack_youngsters_india.csv"
OUT = ROOT / "results" / "youngsters"
TARGET = "Heart Attack Likelihood"

EXPECTED_COLUMNS = [
    "Age", "Gender", "Region", "Urban/Rural", "SES", "Smoking Status", "Alcohol Consumption",
    "Diet Type", "Physical Activity Level", "Screen Time (hrs/day)", "Sleep Duration (hrs/day)",
    "Family History of Heart Disease", "Diabetes", "Hypertension", "Cholesterol Levels (mg/dL)",
    "BMI (kg/m²)", "Stress Level", "Blood Pressure (systolic/diastolic mmHg)",
    "Resting Heart Rate (bpm)", "ECG Results", "Chest Pain Type", "Maximum Heart Rate Achieved",
    "Exercise Induced Angina", "Blood Oxygen Levels (SpO2%)", "Triglyceride Levels (mg/dL)", TARGET,
]

N_JOBS = 4


# --------------------------------------------------------------------------- data
def load_data(path: Path = DATA) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Load the CSV, validate columns, split blood pressure, encode the label."""
    df = pd.read_csv(path)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {sorted(set(df.columns) ^ set(EXPECTED_COLUMNS))}")
    bp = df.pop("Blood Pressure (systolic/diastolic mmHg)").str.split("/", expand=True).astype(float)
    df["Systolic BP"] = bp[0]
    df["Diastolic BP"] = bp[1]
    y = (df.pop(TARGET) == "Yes").astype(int).to_numpy()
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    df[cat_cols] = df[cat_cols].astype(object)
    num_cols = [c for c in df.columns if c not in cat_cols]
    return df, y, cat_cols, num_cols


# --------------------------------------------------------------------------- models
class CatBoostWrapper(ClassifierMixin, BaseEstimator):
    """Thin scikit-learn wrapper so CatBoost can be cloned inside StackingClassifier."""

    def __init__(self, cat_features: tuple = (), params: dict | None = None):
        self.cat_features = cat_features
        self.params = params

    def fit(self, X, y):
        self.model_ = CatBoostClassifier(cat_features=list(self.cat_features), auto_class_weights="Balanced",
                                         random_seed=SEED, thread_count=N_JOBS, verbose=0,
                                         allow_writing_files=False, **(self.params or {}))
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def pos_weight(y: np.ndarray) -> float:
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def make_model(name: str, params: dict, y_train: np.ndarray, cat_cols: list[str], num_cols: list[str]):
    """Return an unfitted estimator (a Pipeline where preprocessing is needed)."""
    ordinal = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)],
        remainder="passthrough",
    )
    if name == "LogisticRegression":
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ])
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED, **params)
        return Pipeline([("pre", pre), ("clf", clf)])
    if name == "RandomForest":
        clf = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=N_JOBS, **params)
        return Pipeline([("pre", ordinal), ("clf", clf)])
    if name == "XGBoost":
        clf = XGBClassifier(scale_pos_weight=pos_weight(y_train), tree_method="hist", eval_metric="logloss",
                            random_state=SEED, n_jobs=N_JOBS, verbosity=0, **params)
        return Pipeline([("pre", ordinal), ("clf", clf)])
    if name == "LightGBM":
        clf = LGBMClassifier(class_weight="balanced", random_state=SEED, n_jobs=N_JOBS, verbose=-1, **params)
        return Pipeline([("pre", ordinal), ("clf", clf)])
    if name == "CatBoost":
        return CatBoostWrapper(cat_features=tuple(cat_cols), params=dict(params))
    if name == "EBM":
        return ExplainableBoostingClassifier(random_state=SEED, n_jobs=N_JOBS, outer_bags=4, **params)
    if name == "Dummy(prior)":
        return DummyClassifier(strategy="prior")
    raise KeyError(name)


def suggest(name: str, trial: optuna.Trial) -> dict:
    """Optuna search spaces."""
    if name == "LogisticRegression":
        return {"C": trial.suggest_float("C", 1e-3, 10, log=True)}
    if name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_float("max_features", 0.2, 1.0),
        }
    if name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    if name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    if name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 600, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        }
    return {}  # EBM (default settings, glass-box) and Dummy are not tuned


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def oof_predictions(name: str, params: dict, X: pd.DataFrame, y: np.ndarray, cv: StratifiedKFold,
                    cat_cols: list[str], num_cols: list[str]) -> np.ndarray:
    """Out-of-fold probabilities for one fixed configuration."""
    oof = np.zeros(len(y))
    for tr, va in cv.split(X, y):
        m = make_model(name, params, y[tr], cat_cols, num_cols)
        m.fit(X.iloc[tr], y[tr])
        oof[va] = predict_proba(m, X.iloc[va])
    return oof


def tune(name: str, X: pd.DataFrame, y: np.ndarray, n_trials: int, inner_k: int,
         cat_cols: list[str], num_cols: list[str]) -> tuple[dict, np.ndarray]:
    """Optuna/TPE tuning by inner stratified K-fold ROC-AUC.
    Returns best params and inner out-of-fold probabilities of the best configuration."""
    from sklearn.metrics import roc_auc_score

    cv = StratifiedKFold(inner_k, shuffle=True, random_state=SEED)
    best = {"auc": -1.0, "params": {}, "oof": None}

    def objective(trial: optuna.Trial) -> float:
        params = suggest(name, trial)
        oof = oof_predictions(name, params, X, y, cv, cat_cols, num_cols)
        auc = roc_auc_score(y, oof)
        if auc > best["auc"]:
            best.update(auc=auc, params=params, oof=oof)
        return auc

    tunable = name not in ("EBM", "Dummy(prior)")
    if tunable and n_trials > 0:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials)
    else:
        best["params"] = {}
        best["oof"] = oof_predictions(name, {}, X, y, cv, cat_cols, num_cols)
    return best["params"], best["oof"]


# --------------------------------------------------------------------------- ensembles
def stacking_model(base: list[tuple[str, dict]], y_train: np.ndarray, cat_cols, num_cols):
    ests = [(n, make_model(n, p, y_train, cat_cols, num_cols)) for n, p in base]
    return StackingClassifier(ests, final_estimator=LogisticRegression(max_iter=5000),
                              stack_method="predict_proba", cv=StratifiedKFold(5, shuffle=True, random_state=SEED))


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few trials, for a smoke run")
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    args = ap.parse_args()

    trials = {"LogisticRegression": 20, "RandomForest": 10, "XGBoost": 20, "LightGBM": 20, "CatBoost": 8,
              "EBM": 0, "Dummy(prior)": 0}
    if args.quick:
        trials = {k: min(v, 2) for k, v in trials.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    X, y, cat_cols, num_cols = load_data()
    print(f"Data: {X.shape}, positives {y.mean():.3%}, cat={len(cat_cols)} num={len(num_cols)}")
    idx = np.arange(len(y))
    dev_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    Xd, yd, Xt, yt = X.iloc[dev_idx].reset_index(drop=True), y[dev_idx], X.iloc[test_idx].reset_index(drop=True), y[test_idx]
    print(f"dev={len(yd)} test={len(yt)}")

    singles = ["Dummy(prior)", "LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost", "EBM"]
    outer = StratifiedKFold(args.outer, shuffle=True, random_state=SEED)
    folds = list(outer.split(Xd, yd))

    # ---------------- nested CV, single models
    cv_rows, fold_params, fold_preds = [], {}, {}
    for name in singles:
        fold_params[name], fold_preds[name] = [], []
        for k, (tr, va) in enumerate(folds):
            params, inner_oof = tune(name, Xd.iloc[tr], yd[tr], trials[name], args.inner, cat_cols, num_cols)
            thr = youden_threshold(yd[tr], inner_oof) if name != "Dummy(prior)" else 0.5
            m = make_model(name, params, yd[tr], cat_cols, num_cols).fit(Xd.iloc[tr], yd[tr])
            p = predict_proba(m, Xd.iloc[va])
            fold_params[name].append(params)
            fold_preds[name].append((p, thr))
            cv_rows.append({"model": name, "fold": k, **all_metrics(yd[va], p, thr)})
        auc = np.mean([r["roc_auc"] for r in cv_rows if r["model"] == name])
        print(f"[{time.time()-t0:6.0f}s] CV {name:20s} ROC-AUC {auc:.4f}")

    cv_df = pd.DataFrame(cv_rows)
    ranking = (cv_df[~cv_df.model.isin(["Dummy(prior)"])].groupby("model").roc_auc.mean()
               .sort_values(ascending=False))
    top3 = list(ranking.index[:3])
    print("Top-3 by CV ROC-AUC:", top3)

    # ---------------- nested CV, ensembles (use per-fold tuned params of the top-3)
    for k, (tr, va) in enumerate(folds):
        pv = np.mean([fold_preds[n][k][0] for n in top3], axis=0)
        # voting threshold: average of inner-OOF is not stored, so use mean of base thresholds
        thr_v = float(np.mean([fold_preds[n][k][1] for n in top3]))
        cv_rows.append({"model": "SoftVoting(top3)", "fold": k, **all_metrics(yd[va], pv, thr_v)})
        st = stacking_model([(n, fold_params[n][k]) for n in top3], yd[tr], cat_cols, num_cols)
        icv = StratifiedKFold(args.inner, shuffle=True, random_state=SEED)
        st_oof = cross_val_predict(clone(st), Xd.iloc[tr], yd[tr], cv=icv, method="predict_proba")[:, 1]
        st.fit(Xd.iloc[tr], yd[tr])
        ps = predict_proba(st, Xd.iloc[va])
        cv_rows.append({"model": "Stacking(top3)", "fold": k, **all_metrics(yd[va], ps, youden_threshold(yd[tr], st_oof))})
    print(f"[{time.time()-t0:6.0f}s] CV ensembles done")

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(OUT / "cv_metrics_per_fold.csv", index=False)
    metric_cols = [c for c in cv_df.columns if c not in ("model", "fold")]
    agg = cv_df.groupby("model")[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.sort_values("roc_auc_mean", ascending=False).to_csv(OUT / "cv_metrics.csv")

    # ---------------- final models: tune on all dev, refit, evaluate once on test
    final_params, test_probs, thresholds, sens90 = {}, {}, {}, {}
    for name in singles:
        params, dev_oof = tune(name, Xd, yd, trials[name], args.inner, cat_cols, num_cols)
        final_params[name] = params
        thresholds[name] = youden_threshold(yd, dev_oof) if name != "Dummy(prior)" else 0.5
        sens90[name] = sensitivity_threshold(yd, dev_oof) if name != "Dummy(prior)" else 0.5
        m = make_model(name, params, yd, cat_cols, num_cols).fit(Xd, yd)
        test_probs[name] = predict_proba(m, Xt)
        test_probs[name + "__devoof"] = dev_oof
        print(f"[{time.time()-t0:6.0f}s] final {name} fitted")

    test_probs["SoftVoting(top3)"] = np.mean([test_probs[n] for n in top3], axis=0)
    v_oof = np.mean([test_probs[n + "__devoof"] for n in top3], axis=0)
    thresholds["SoftVoting(top3)"] = youden_threshold(yd, v_oof)
    sens90["SoftVoting(top3)"] = sensitivity_threshold(yd, v_oof)
    st = stacking_model([(n, final_params[n]) for n in top3], yd, cat_cols, num_cols)
    st_oof = cross_val_predict(clone(st), Xd, yd, cv=StratifiedKFold(args.inner, shuffle=True, random_state=SEED),
                               method="predict_proba")[:, 1]
    st.fit(Xd, yd)
    test_probs["Stacking(top3)"] = predict_proba(st, Xt)
    thresholds["Stacking(top3)"] = youden_threshold(yd, st_oof)
    sens90["Stacking(top3)"] = sensitivity_threshold(yd, st_oof)

    models = singles + ["SoftVoting(top3)", "Stacking(top3)"]
    rows = []
    for name in models:
        p = test_probs[name]
        met = all_metrics(yt, p, thresholds[name])
        ci = bootstrap_ci(yt, p, thresholds[name])
        s90 = all_metrics(yt, p, sens90[name])
        row = {"model": name, **met}
        for kk, (lo, hi) in ci.items():
            row[f"{kk}_lo"], row[f"{kk}_hi"] = lo, hi
        row["sens90_threshold"], row["sens90_sensitivity"], row["sens90_specificity"] = (
            sens90[name], s90["sensitivity"], s90["specificity"])
        rows.append(row)
    test_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    test_df.to_csv(OUT / "test_metrics.csv", index=False)

    # ---------------- paired comparisons
    best = next(m for m in test_df.model if m != "Dummy(prior)")
    comp = []
    for ref in ["LogisticRegression", "RandomForest", "Dummy(prior)"]:
        if ref == best:
            continue
        d, lo, hi, pv = paired_bootstrap_auc(yt, test_probs[best], test_probs[ref])
        comp.append({"comparison": f"{best} - {ref}", "delta_auc": d, "ci_lo": lo, "ci_hi": hi, "p_value": pv})
    pd.DataFrame(comp).to_csv(OUT / "model_comparison.csv", index=False)

    # ---------------- per-patient test predictions
    pred = pd.DataFrame({"row_id_in_csv": test_idx, "true_label": yt})
    for name in models:
        pred[f"prob_{name}"] = np.round(test_probs[name], 4)
    pred[f"pred_{best}"] = (test_probs[best] >= thresholds[best]).astype(int)
    pred.to_csv(OUT / "test_predictions.csv", index=False)

    # ---------------- sanity: univariate association + label permutation
    assoc = []
    for c in cat_cols:
        tab = pd.crosstab(X[c], y)
        chi2, pv, _, _ = stats.chi2_contingency(tab)
        assoc.append({"feature": c, "test": "chi-square", "statistic": chi2, "p_value": pv})
    for c in num_cols:
        u, pv = stats.mannwhitneyu(X.loc[y == 1, c], X.loc[y == 0, c])
        assoc.append({"feature": c, "test": "Mann-Whitney U", "statistic": u, "p_value": pv})
    assoc_df = pd.DataFrame(assoc).sort_values("p_value")
    assoc_df["bonferroni_p"] = np.minimum(assoc_df.p_value * len(assoc_df), 1.0)
    assoc_df.to_csv(OUT / "feature_association.csv", index=False)

    rng = np.random.default_rng(SEED)
    yperm = rng.permutation(yd)
    mperm = make_model("LightGBM", final_params["LightGBM"], yperm, cat_cols, num_cols).fit(Xd, yperm)
    from sklearn.metrics import roc_auc_score
    perm_auc = roc_auc_score(yt, predict_proba(mperm, Xt))

    with open(OUT / "best_params.json", "w") as f:
        json.dump({"final": final_params, "top3": top3, "thresholds_youden": thresholds,
                   "thresholds_sens90": sens90, "trials": trials, "outer_k": args.outer,
                   "inner_k": args.inner}, f, indent=2, default=float)

    write_report(test_df, agg, comp, assoc_df, perm_auc, y, trials, args, time.time() - t0)
    (OUT / "environment.txt").write_text(
        subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    from src import plots
    plots.main()
    print(f"Done in {time.time()-t0:.0f}s. Results in {OUT}")


def _ci(r: pd.Series, k: str) -> str:
    return f"{r[k]:.3f} ({r[k+'_lo']:.3f}–{r[k+'_hi']:.3f})"


def write_report(test_df, agg, comp, assoc_df, perm_auc, y, trials, args, secs) -> None:
    lines = ["# Prediction results — heart_attack_youngsters_india.csv", "",
             "Research prototype, not for clinical use. Every number below is written by "
             "`src/run_youngsters.py`; nothing is typed by hand.", "",
             f"- Rows: {len(y)}; positive class (heart-attack likelihood = Yes): {y.mean():.2%}",
             "- Split: stratified 80/20 (seed 42). Test set used once.",
             f"- Nested CV on dev: outer {args.outer}-fold, inner {args.inner}-fold, Optuna TPE; trials per model: "
             + ", ".join(f"{k} {v}" for k, v in trials.items()),
             "- Threshold: Youden's J on inner out-of-fold predictions; 95% CIs from 1,000 bootstrap resamples.",
             "- TabPFN not run (needs PyTorch and gated model weights in this environment).", "",
             "## Held-out test set (n = %d)" % int(round(len(y) * 0.2)), "",
             "| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Sensitivity | Specificity | F1 | MCC | Brier | Cal. intercept | Cal. slope |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in test_df.iterrows():
        lines.append(f"| {r.model} | {_ci(r,'roc_auc')} | {_ci(r,'pr_auc')} | {_ci(r,'sensitivity')} | "
                     f"{_ci(r,'specificity')} | {r.f1:.3f} | {r.mcc:.3f} | {r.brier:.3f} | "
                     f"{r.cal_intercept:.2f} | {r.cal_slope:.2f} |")
    lines += ["", "## Nested cross-validation on dev (mean ± SD over outer folds)", "",
              "| Model | ROC-AUC | PR-AUC | Sensitivity | Specificity | MCC | Brier |", "|---|---|---|---|---|---|---|"]
    for m, r in agg.sort_values("roc_auc_mean", ascending=False).iterrows():
        lines.append(f"| {m} | " + " | ".join(f"{r[k+'_mean']:.3f} ± {r[k+'_std']:.3f}" for k in
                                             ["roc_auc", "pr_auc", "sensitivity", "specificity", "mcc", "brier"]) + " |")
    lines += ["", "## Paired bootstrap of test ROC-AUC", "", "| Comparison | ΔAUC | 95% CI | p |", "|---|---|---|---|"]
    for c in comp:
        lines.append(f"| {c['comparison']} | {c['delta_auc']:+.3f} | {c['ci_lo']:+.3f} to {c['ci_hi']:+.3f} | {c['p_value']:.3f} |")
    lines += ["", "## Sanity checks", "",
              f"- Label-permutation test (LightGBM trained on shuffled dev labels): test ROC-AUC = {perm_auc:.3f}.",
              f"- Features with univariate association to the label at Bonferroni p < 0.05: "
              f"{int((assoc_df.bonferroni_p < 0.05).sum())} of {len(assoc_df)}. "
              f"Smallest raw p-value: {assoc_df.p_value.min():.3g} ({assoc_df.iloc[0].feature}).",
              "", f"Run time: {secs/60:.1f} min."]
    (OUT / "results_table.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
