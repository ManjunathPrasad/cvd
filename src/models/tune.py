"""Hyperparameter tuning with Optuna (TPE), objective = inner-CV ROC-AUC."""
from __future__ import annotations

import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import load_config, load_search_spaces
from src.models.zoo import TUNED, make_model

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def n_trials_for(name: str) -> int:
    t = load_config()["tuning"]
    return int((t.get("n_trials_override") or {}).get(name, t["n_trials"]))


def suggest(name: str, trial: optuna.Trial) -> dict:
    """Sample one configuration from configs/search_spaces.yaml."""
    params = {}
    for p, spec in (load_search_spaces().get(name) or {}).items():
        if spec["type"] == "float":
            params[p] = trial.suggest_float(p, spec["low"], spec["high"], log=spec.get("log", False))
        elif spec["type"] == "int":
            if spec.get("log"):
                params[p] = trial.suggest_int(p, spec["low"], spec["high"], log=True)
            else:
                params[p] = trial.suggest_int(p, spec["low"], spec["high"], step=spec.get("step", 1))
        elif spec["type"] == "cat":
            params[p] = trial.suggest_categorical(p, spec["choices"])
    return params


def oof_predictions(name: str, params: dict, X: pd.DataFrame, y: np.ndarray, features: list[str],
                    k: int, seed: int) -> np.ndarray:
    """Out-of-fold probabilities for one fixed configuration (k-fold stratified)."""
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        m = make_model(name, params, features, y[tr]).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def tune(name: str, X: pd.DataFrame, y: np.ndarray, features: list[str], n_trials: int | None = None,
         inner_k: int | None = None) -> tuple[dict, np.ndarray]:
    """Tune `name` on (X, y). Returns best params and the inner out-of-fold probabilities of the best
    configuration (used for threshold choice). Untuned models return {} and their inner OOF."""
    cfg = load_config()
    seed = cfg["seed"]
    inner_k = inner_k or cfg["cv"]["inner_folds"]
    n_trials = n_trials_for(name) if n_trials is None else n_trials
    if name not in TUNED or n_trials == 0:
        return {}, oof_predictions(name, {}, X, y, features, inner_k, seed)

    best = {"auc": -1.0, "params": {}, "oof": None}

    def objective(trial: optuna.Trial) -> float:
        params = suggest(name, trial)
        oof = oof_predictions(name, params, X, y, features, inner_k, seed)
        auc = roc_auc_score(y, oof)
        if auc > best["auc"]:
            best.update(auc=auc, params=params, oof=oof)
        return auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return best["params"], best["oof"]
