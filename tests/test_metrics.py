import numpy as np
from sklearn.metrics import roc_auc_score

from src.metrics import all_metrics, calibration_intercept_slope, paired_bootstrap_auc, youden_threshold


def test_perfect_calibration_gives_zero_intercept_unit_slope():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    a, b = calibration_intercept_slope(y, p)
    assert abs(a) < 0.05 and abs(b - 1) < 0.08


def test_youden_separates_perfect_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = youden_threshold(y, p)
    m = all_metrics(y, p, t)
    assert m["sensitivity"] == 1 and m["specificity"] == 1 and m["roc_auc"] == 1


def test_paired_bootstrap_identical_models_is_zero():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300); p = rng.uniform(size=300)
    d, lo, hi, _ = paired_bootstrap_auc(y, p, p, n=100)
    assert d == 0 and lo == 0 and hi == 0
    assert np.isclose(roc_auc_score(y, p), all_metrics(y, p, 0.5)["roc_auc"])
