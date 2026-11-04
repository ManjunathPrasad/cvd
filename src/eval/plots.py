"""Shared figure style and curve plots (ROC, PR, calibration, decision curves, AUC forest)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

# Fixed categorical order; colour follows the model, never its rank.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost", "EBM",
               "SoftVoting", "Stacking"]   # 8 fixed slots; TabPFN (if available) is drawn in grey


def color_of(name: str) -> str:
    """Stable colour per model name."""
    base = name.split(" ")[0]
    return COLORS[MODEL_ORDER.index(base)] if base in MODEL_ORDER else INK2


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2,
        "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    })


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], aucs: dict[str, float], title: str, out: Path) -> None:
    style()
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color_of(name), label=f"{name} (AUC {aucs[name]:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2, label="Chance")
    ax.set(xlabel="1 − specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.01), title=title)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(out); plt.close(fig)


def plot_pr(curves, aps: dict[str, float], title: str, out: Path) -> None:
    style()
    fig, ax = plt.subplots(figsize=(5.2, 5))
    prev = None
    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=color_of(name), label=f"{name} (AP {aps[name]:.3f})")
        prev = y.mean()
    ax.axhline(prev, ls="--", lw=1, color=INK2, label=f"Prevalence ({prev:.2f})")
    ax.set(xlabel="Recall (sensitivity)", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01), title=title)
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(out); plt.close(fig)


def plot_calibration(curves, briers: dict[str, float], title: str, out: Path, n_bins: int = 10) -> None:
    style()
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5.2, 6.2), height_ratios=[3, 1], sharex=True)
    for name, (y, p) in curves.items():
        frac, mean = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean, frac, marker="o", ms=4, color=color_of(name), label=f"{name} (Brier {briers[name]:.3f})")
        axh.hist(p, bins=20, range=(0, 1), histtype="step", lw=1.5, color=color_of(name))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2, label="Perfect calibration")
    ax.set(ylabel="Observed fraction with disease", xlim=(0, 1), ylim=(0, 1), title=title)
    ax.legend(loc="upper left", fontsize=8)
    axh.set(xlabel="Predicted probability", ylabel="Count")
    fig.savefig(out); plt.close(fig)


def plot_auc_forest(df: pd.DataFrame, title: str, out: Path, cv: pd.DataFrame | None = None) -> None:
    """df: columns model, roc_auc, roc_auc_lo, roc_auc_hi. cv (optional): model, roc_auc_mean, roc_auc_std."""
    style()
    t = df.sort_values("roc_auc").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(t) + 1.3))
    yy = np.arange(len(t))
    ax.errorbar(t.roc_auc, yy + 0.15, xerr=[t.roc_auc - t.roc_auc_lo, t.roc_auc_hi - t.roc_auc], fmt="o", ms=7,
                color=COLORS[0], elinewidth=2, capsize=0, label="Test ROC-AUC (95% bootstrap CI)")
    if cv is not None:
        c = cv.set_index("model").reindex(t.model)
        ax.errorbar(c.roc_auc_mean, yy - 0.15, xerr=c.roc_auc_std, fmt="o", ms=7, mfc=SURFACE, color=INK2,
                    elinewidth=1.5, capsize=0, label="Nested-CV ROC-AUC (mean ± SD)")
    for i, r in t.iterrows():
        ax.text(r.roc_auc_hi + 0.003, i + 0.15, f"{r.roc_auc:.3f}", va="center", fontsize=8)
    ax.set_yticks(yy, t.model)
    ax.set(xlabel="ROC-AUC", title=title)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=8)
    fig.savefig(out); plt.close(fig)


def plot_decision_curves(nb: pd.DataFrame, title: str, out: Path) -> None:
    """nb: columns threshold, strategy, net_benefit (strategies include 'Treat all', 'Treat none')."""
    style()
    fig, ax = plt.subplots(figsize=(6, 4.6))
    tiers = [s for s in nb.strategy.unique() if s not in ("Treat all", "Treat none")]
    for i, s in enumerate(tiers):
        d = nb[nb.strategy == s]
        ax.plot(d.threshold, d.net_benefit, color=COLORS[i], label=s)
    for s, ls in [("Treat all", ":"), ("Treat none", "--")]:
        d = nb[nb.strategy == s]
        ax.plot(d.threshold, d.net_benefit, color=INK2, ls=ls, lw=1.2, label=s)
    top = nb[nb.strategy != "Treat all"].net_benefit.max()
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", ylim=(-0.05, top + 0.05), title=title)
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(out); plt.close(fig)
