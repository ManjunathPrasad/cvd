"""Leakage-free evaluation protocol (PLAN.md Section 4), shared by E1 and E2.

1. Nested CV on dev: outer stratified K-fold; inside each outer-training fold, Optuna tuning with an
   inner stratified K-fold. Threshold = Youden's J on the inner out-of-fold predictions.
2. Final model: tune on all of dev, refit, predict the test set once.
3. Ensembles (soft voting, stacking) built from the three best single models by CV ROC-AUC.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import load_config
from src.eval.metrics import all_metrics, sensitivity_threshold, youden_threshold
from src.models.tune import tune
from src.models.zoo import make_model, make_stacking


@dataclass
class ProtocolResult:
    """Everything an experiment needs for tables, figures and explanations."""
    features: list[str]
    cv_rows: list[dict] = field(default_factory=list)          # per outer fold, per model
    fold_params: dict = field(default_factory=dict)            # model -> [params per fold]
    final_params: dict = field(default_factory=dict)           # model -> params tuned on all dev
    thresholds: dict = field(default_factory=dict)             # model -> Youden threshold (from dev OOF)
    sens90: dict = field(default_factory=dict)                 # model -> threshold giving sens >= target on dev
    dev_oof: dict = field(default_factory=dict)                # model -> dev out-of-fold probabilities
    test_probs: dict = field(default_factory=dict)             # model -> test probabilities
    fitted: dict = field(default_factory=dict)                 # model -> estimator fitted on all dev
    top3: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    def cv_table(self) -> pd.DataFrame:
        df = pd.DataFrame(self.cv_rows)
        cols = [c for c in df.columns if c not in ("model", "fold")]
        agg = df.groupby("model")[cols].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        return agg.sort_values("roc_auc_mean", ascending=False).reset_index()


def _outer_fold(name, X, y, tr, va, features, k):
    params, inner_oof = tune(name, X.iloc[tr], y[tr], features)
    thr = youden_threshold(y[tr], inner_oof)
    m = make_model(name, params, features, y[tr]).fit(X.iloc[tr], y[tr])
    p = m.predict_proba(X.iloc[va])[:, 1]
    return k, params, inner_oof, thr, p


def run_protocol(models: list[str], X_dev: pd.DataFrame, y_dev: np.ndarray, X_test: pd.DataFrame,
                 features: list[str], ensembles: bool = True, log=print) -> ProtocolResult:
    """Run nested CV and the final test evaluation for `models` on `features`."""
    cfg = load_config()
    seed, k_out, k_in = cfg["seed"], cfg["cv"]["outer_folds"], cfg["cv"]["inner_folds"]
    n_jobs = cfg["cv"]["n_jobs"]
    Xd, Xt = X_dev[features], X_test[features]
    folds = list(StratifiedKFold(k_out, shuffle=True, random_state=seed).split(Xd, y_dev))
    res = ProtocolResult(features=list(features))
    inner = {}  # (model, fold) -> (inner_oof, thr, p_outer)

    for name in models:
        t0 = time.time()
        jobs = 1 if name == "TabPFN" else n_jobs
        out = Parallel(n_jobs=jobs)(delayed(_outer_fold)(name, Xd, y_dev, tr, va, features, k)
                                    for k, (tr, va) in enumerate(folds))
        res.fold_params[name] = [None] * k_out
        for k, params, inner_oof, thr, p in sorted(out, key=lambda o: o[0]):
            va = folds[k][1]
            res.fold_params[name][k] = params
            inner[(name, k)] = (inner_oof, thr, p)
            res.cv_rows.append({"model": name, "fold": k, **all_metrics(y_dev[va], p, thr)})
        res.timings[name] = time.time() - t0
        auc = np.mean([r["roc_auc"] for r in res.cv_rows if r["model"] == name])
        log(f"  CV {name:20s} ROC-AUC {auc:.4f}  ({res.timings[name]:.0f}s)")

    cvt = res.cv_table()
    res.top3 = list(cvt[cvt.model.isin(models)].model[:3])

    if ensembles and len(res.top3) == 3:
        for k, (tr, va) in enumerate(folds):
            pv = np.mean([inner[(n, k)][2] for n in res.top3], axis=0)
            thr_v = youden_threshold(y_dev[tr], np.mean([inner[(n, k)][0] for n in res.top3], axis=0))
            res.cv_rows.append({"model": "SoftVoting", "fold": k, **all_metrics(y_dev[va], pv, thr_v)})
        def _stack_fold(k, tr, va):
            st = make_stacking([(n, res.fold_params[n][k]) for n in res.top3], features, y_dev[tr])
            oof = cross_val_predict(clone(st), Xd.iloc[tr], y_dev[tr], method="predict_proba",
                                    cv=StratifiedKFold(k_in, shuffle=True, random_state=seed))[:, 1]
            st.fit(Xd.iloc[tr], y_dev[tr])
            return k, st.predict_proba(Xd.iloc[va])[:, 1], youden_threshold(y_dev[tr], oof)
        jobs = 1 if "TabPFN" in res.top3 else n_jobs
        for k, p, thr in sorted(Parallel(n_jobs=jobs)(delayed(_stack_fold)(k, tr, va)
                                                     for k, (tr, va) in enumerate(folds))):
            res.cv_rows.append({"model": "Stacking", "fold": k, **all_metrics(y_dev[folds[k][1]], p, thr)})
        log(f"  CV ensembles of {res.top3} done")

    # ---- final models on all of dev, evaluated once on test
    sens_target = cfg["evaluation"]["sensitivity_target"]
    def _final(name):
        params, oof = tune(name, Xd, y_dev, features)
        m = make_model(name, params, features, y_dev).fit(Xd, y_dev)
        return name, params, oof, m
    jobs = n_jobs
    finals = Parallel(n_jobs=jobs)(delayed(_final)(n) for n in models if n != "TabPFN")
    if "TabPFN" in models:
        finals.append(_final("TabPFN"))
    for name, params, oof, m in finals:
        res.final_params[name], res.dev_oof[name], res.fitted[name] = params, oof, m
        res.thresholds[name] = youden_threshold(y_dev, oof)
        res.sens90[name] = sensitivity_threshold(y_dev, oof, sens_target)
        res.test_probs[name] = m.predict_proba(Xt)[:, 1]

    if ensembles and len(res.top3) == 3:
        res.test_probs["SoftVoting"] = np.mean([res.test_probs[n] for n in res.top3], axis=0)
        res.dev_oof["SoftVoting"] = np.mean([res.dev_oof[n] for n in res.top3], axis=0)
        st = make_stacking([(n, res.final_params[n]) for n in res.top3], features, y_dev)
        res.dev_oof["Stacking"] = cross_val_predict(clone(st), Xd, y_dev, method="predict_proba",
                                                    cv=StratifiedKFold(k_in, shuffle=True, random_state=seed))[:, 1]
        res.fitted["Stacking"] = st.fit(Xd, y_dev)
        res.test_probs["Stacking"] = st.predict_proba(Xt)[:, 1]
        for e in ("SoftVoting", "Stacking"):
            res.thresholds[e] = youden_threshold(y_dev, res.dev_oof[e])
            res.sens90[e] = sensitivity_threshold(y_dev, res.dev_oof[e], sens_target)
    log("  final models fitted on dev and evaluated on test")
    return res


def test_table(res: ProtocolResult, y_test: np.ndarray, n_boot: int | None = None) -> pd.DataFrame:
    """Test metrics with bootstrap 95% CIs, at the Youden threshold and the sensitivity-target threshold."""
    from src.eval.metrics import bootstrap_ci

    cfg = load_config()
    n_boot = n_boot or cfg["evaluation"]["bootstrap_resamples"]
    rows = []
    for name, p in res.test_probs.items():
        met = all_metrics(y_test, p, res.thresholds[name])
        ci = bootstrap_ci(y_test, p, res.thresholds[name], n=n_boot, seed=cfg["seed"])
        s = all_metrics(y_test, p, res.sens90[name])
        row = {"model": name, **met}
        for k, (lo, hi) in ci.items():
            row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
        row.update(sens90_threshold=res.sens90[name], sens90_sensitivity=s["sensitivity"],
                   sens90_specificity=s["specificity"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
