"""Decision-curve analysis (Vickers & Elkin): net benefit across threshold probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def net_benefit(y: np.ndarray, p: np.ndarray, t: float) -> float:
    """Net benefit of treating patients with predicted risk >= t."""
    n = len(y)
    treat = p >= t
    tp = np.sum(treat & (y == 1))
    fp = np.sum(treat & (y == 0))
    return float(tp / n - fp / n * t / (1 - t))


def decision_curve(y: np.ndarray, probs: dict[str, np.ndarray], lo: float = 0.05, hi: float = 0.50,
                   step: float = 0.01) -> pd.DataFrame:
    """Net benefit for each model, 'Treat all' and 'Treat none' at thresholds lo..hi."""
    ts = np.round(np.arange(lo, hi + 1e-9, step), 4)
    prev = y.mean()
    rows = []
    for t in ts:
        for name, p in probs.items():
            rows.append({"threshold": t, "strategy": name, "net_benefit": net_benefit(y, p, t)})
        rows.append({"threshold": t, "strategy": "Treat all", "net_benefit": float(prev - (1 - prev) * t / (1 - t))})
        rows.append({"threshold": t, "strategy": "Treat none", "net_benefit": 0.0})
    return pd.DataFrame(rows)
