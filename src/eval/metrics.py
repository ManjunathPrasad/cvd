"""Evaluation metrics: discrimination, classification, calibration and bootstrap CIs."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

EPS = 1e-6


def youden_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    """Threshold that maximises Youden's J = sensitivity + specificity - 1."""
    fpr, tpr, thr = roc_curve(y_true, p)
    j = tpr - fpr
    t = float(thr[int(np.argmax(j))])
    return min(t, 1.0)


def sensitivity_threshold(y_true: np.ndarray, p: np.ndarray, target: float = 0.90) -> float:
    """Highest threshold whose sensitivity is at least `target`."""
    fpr, tpr, thr = roc_curve(y_true, p)
    ok = np.where(tpr >= target)[0]
    return min(float(thr[ok[0]]), 1.0)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """Calibration slope (logistic fit of y on logit p) and calibration-in-the-large
    intercept (logistic fit of y with logit p as a fixed offset). Ideal: intercept 0, slope 1."""
    lp = _logit(p)
    y = y.astype(float)

    # slope: y ~ a + b * lp  (Newton-Raphson)
    X = np.column_stack([np.ones_like(lp), lp])
    beta = np.zeros(2)
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-X @ beta))
        W = mu * (1 - mu) + 1e-12
        H = X.T @ (X * W[:, None])
        g = X.T @ (y - mu)
        beta = beta + np.linalg.solve(H + 1e-9 * np.eye(2), g)
    slope = float(beta[1])

    # intercept with offset: y ~ a + offset(lp)
    a = 0.0
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(a + lp)))
        a += float(np.sum(y - mu) / (np.sum(mu * (1 - mu)) + 1e-12))
    return a, slope


def all_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    """All point metrics for probabilities `p` dichotomised at `thr`."""
    yhat = (p >= thr).astype(int)
    tp = int(np.sum((yhat == 1) & (y == 1)))
    tn = int(np.sum((yhat == 0) & (y == 0)))
    fp = int(np.sum((yhat == 1) & (y == 0)))
    fn = int(np.sum((yhat == 0) & (y == 1)))
    ci, cs = calibration_intercept_slope(y, p)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y, yhat, zero_division=0),
        "mcc": matthews_corrcoef(y, yhat),
        "accuracy": (tp + tn) / len(y),
        "brier": brier_score_loss(y, p),
        "cal_intercept": ci,
        "cal_slope": cs,
        "threshold": thr,
    }


BOOT_KEYS = ["roc_auc", "pr_auc", "sensitivity", "specificity", "f1", "mcc", "brier"]


def bootstrap_ci(y: np.ndarray, p: np.ndarray, thr: float, n: int = 1000, seed: int = 42) -> dict[str, tuple[float, float]]:
    """Percentile 95% bootstrap CIs (threshold held fixed)."""
    rng = np.random.default_rng(seed)
    vals: dict[str, list[float]] = {k: [] for k in BOOT_KEYS}
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        yb, pb = y[b], p[b]
        yhat = (pb >= thr).astype(int)
        tp = np.sum((yhat == 1) & (yb == 1)); fn = np.sum((yhat == 0) & (yb == 1))
        tn = np.sum((yhat == 0) & (yb == 0)); fp = np.sum((yhat == 1) & (yb == 0))
        vals["roc_auc"].append(roc_auc_score(yb, pb))
        vals["pr_auc"].append(average_precision_score(yb, pb))
        vals["sensitivity"].append(tp / (tp + fn))
        vals["specificity"].append(tn / (tn + fp))
        vals["f1"].append(f1_score(yb, yhat, zero_division=0))
        vals["mcc"].append(matthews_corrcoef(yb, yhat))
        vals["brier"].append(brier_score_loss(yb, pb))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items()}


def paired_bootstrap_auc(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 1000, seed: int = 42) -> tuple[float, float, float, float]:
    """AUC(p1) - AUC(p2) with a paired-bootstrap 95% CI and two-sided p-value."""
    rng = np.random.default_rng(seed)
    d0 = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    ds = []
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        if len(np.unique(y[b])) < 2:
            continue
        ds.append(roc_auc_score(y[b], p1[b]) - roc_auc_score(y[b], p2[b]))
    ds = np.array(ds)
    pval = float(min(1.0, 2 * min(np.mean(ds <= 0), np.mean(ds >= 0))))
    return float(d0), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), pval
