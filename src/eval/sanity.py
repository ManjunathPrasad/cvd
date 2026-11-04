"""Sanity checks for suspiciously high performance (PLAN.md Section 4, step 7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.models.zoo import make_model


def near_duplicates(X_dev: pd.DataFrame, X_test: pd.DataFrame, tol: float = 0.05) -> dict:
    """Count test rows identical to a dev row, and test rows whose nearest dev row (standardised,
    median-imputed, Euclidean distance) is within `tol` × the median nearest-neighbour distance."""
    med = X_dev.median()
    A = X_dev.fillna(med).to_numpy(float)
    B = X_test.fillna(med).to_numpy(float)
    sc = StandardScaler().fit(A)
    A, B = sc.transform(A), sc.transform(B)
    d = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1))
    nn = d.min(axis=1)
    exact = int((nn < 1e-9).sum())
    within_dev = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(within_dev, np.inf)
    ref = float(np.median(within_dev.min(axis=1)))
    close = int((nn <= tol * ref).sum())
    return {"exact_matches": exact, "near_matches": close, "tol": tol,
            "median_nn_distance_test_to_dev": float(np.median(nn)), "median_nn_distance_within_dev": ref}


def permutation_test(name: str, params: dict, features: list[str], X_dev, y_dev, X_test, y_test,
                     n_repeats: int = 5, seed: int = 42) -> list[float]:
    """Refit with shuffled dev labels; test AUC should fall to about 0.5."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_repeats):
        yp = rng.permutation(y_dev)
        m = make_model(name, params, features, yp).fit(X_dev[features], yp)
        out.append(float(roc_auc_score(y_test, m.predict_proba(X_test[features])[:, 1])))
    return out


def drop_one(name: str, params: dict, features: list[str], X_dev, y_dev, X_test, y_test) -> pd.DataFrame:
    """Test AUC after removing each attribute in turn (fixed hyperparameters)."""
    base = make_model(name, params, features, y_dev).fit(X_dev[features], y_dev)
    full = roc_auc_score(y_test, base.predict_proba(X_test[features])[:, 1])
    rows = []
    for f in features:
        fs = [c for c in features if c != f]
        m = make_model(name, params, fs, y_dev).fit(X_dev[fs], y_dev)
        auc = roc_auc_score(y_test, m.predict_proba(X_test[fs])[:, 1])
        rows.append({"removed": f, "test_auc": auc, "delta_vs_full": auc - full})
    return pd.DataFrame(rows).sort_values("delta_vs_full").reset_index(drop=True)
