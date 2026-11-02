"""Figures for the youngsters benchmark. Reads only files written by src/run_youngsters.py.

Usage:  python -m src.plots
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "youngsters"
FIG = RES / "figures"

# Reference categorical palette (fixed order; colour follows the model, never its rank)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost", "EBM",
               "SoftVoting(top3)", "Stacking(top3)"]
COLOR = dict(zip(MODEL_ORDER, SERIES))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2,
    "legend.frameon": False, "figure.dpi": 150,
})


def _curve_models(test: pd.DataFrame, k: int = 5) -> list[str]:
    """Logistic regression (reference) plus the best models by test ROC-AUC, max k lines."""
    ranked = [m for m in test.sort_values("roc_auc", ascending=False).model if m in COLOR]
    chosen = ["LogisticRegression"] + [m for m in ranked if m != "LogisticRegression"][: k - 1]
    return [m for m in MODEL_ORDER if m in chosen]


def fig_roc(pred, test, models):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    y = pred.true_label.to_numpy()
    for m in models:
        fpr, tpr, _ = roc_curve(y, pred[f"prob_{m}"])
        auc = test.set_index("model").loc[m, "roc_auc"]
        ax.plot(fpr, tpr, color=COLOR[m], label=f"{m} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2, label="Chance")
    ax.set(xlabel="1 − specificity (false-positive rate)", ylabel="Sensitivity (true-positive rate)",
           xlim=(0, 1), ylim=(0, 1.01), title="Figure 1. ROC curves, held-out test set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig1_roc.png"); plt.close(fig)


def fig_pr(pred, test, models):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    y = pred.true_label.to_numpy()
    for m in models:
        prec, rec, _ = precision_recall_curve(y, pred[f"prob_{m}"])
        ap = test.set_index("model").loc[m, "pr_auc"]
        ax.plot(rec, prec, color=COLOR[m], label=f"{m} (AP {ap:.3f})")
    ax.axhline(y.mean(), ls="--", lw=1, color=INK2, label=f"Prevalence ({y.mean():.3f})")
    ax.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", xlim=(0, 1), ylim=(0, 1.01),
           title="Figure 2. Precision–recall curves, test set")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig2_pr.png"); plt.close(fig)


def fig_calibration(pred, test, models):
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5.2, 6.2), height_ratios=[3, 1], sharex=True)
    y = pred.true_label.to_numpy()
    for m in models:
        frac, mean = calibration_curve(y, pred[f"prob_{m}"], n_bins=10, strategy="quantile")
        b = test.set_index("model").loc[m, "brier"]
        ax.plot(mean, frac, marker="o", ms=4, color=COLOR[m], label=f"{m} (Brier {b:.3f})")
        axh.hist(pred[f"prob_{m}"], bins=30, range=(0, 1), histtype="step", lw=1.5, color=COLOR[m])
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2, label="Perfect calibration")
    ax.set(ylabel="Observed fraction positive", xlim=(0, 1), ylim=(0, 1),
           title="Figure 3. Calibration curves (10 quantile bins), test set")
    ax.legend(loc="upper left", fontsize=8)
    axh.set(xlabel="Predicted probability", ylabel="Count")
    fig.tight_layout(); fig.savefig(FIG / "fig3_calibration.png"); plt.close(fig)


def fig_auc_forest(test, cv):
    """Test ROC-AUC with 95% CI (dots) and nested-CV mean ± SD (hollow), one row per model."""
    t = test.sort_values("roc_auc").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(t) + 1.2))
    yy = np.arange(len(t))
    ax.errorbar(t.roc_auc, yy + 0.15, xerr=[t.roc_auc - t.roc_auc_lo, t.roc_auc_hi - t.roc_auc],
                fmt="o", ms=7, color=SERIES[0], ecolor=SERIES[0], elinewidth=2, capsize=0,
                label="Test ROC-AUC (95% bootstrap CI)")
    c = cv.set_index("model").reindex(t.model)
    ax.errorbar(c.roc_auc_mean, yy - 0.15, xerr=c.roc_auc_std, fmt="o", ms=7, mfc=SURFACE,
                color=INK2, ecolor=INK2, elinewidth=1.5, capsize=0, label="Nested-CV ROC-AUC (mean ± SD)")
    for i, r in t.iterrows():
        ax.text(r.roc_auc_hi + 0.004, i + 0.15, f"{r.roc_auc:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(0.5, ls="--", lw=1, color=INK2)
    ax.set_yticks(yy, t.model)
    ax.set(xlabel="ROC-AUC", title="Figure 4. Discrimination by model")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig4_auc_forest.png"); plt.close(fig)


def fig_metric_bars(test):
    """Sensitivity, specificity, F1 and MCC at the Youden threshold, one small panel per metric."""
    t = test[test.model != "Dummy(prior)"].set_index("model").reindex([m for m in MODEL_ORDER if m in set(test.model)])
    metrics = [("sensitivity", True), ("specificity", True), ("f1", False), ("mcc", False)]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
    yy = np.arange(len(t))
    for ax, (k, has_ci) in zip(axes, metrics):
        ax.barh(yy, t[k], height=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
        if has_ci:
            ax.errorbar(t[k], yy, xerr=[t[k] - t[f"{k}_lo"], t[f"{k}_hi"] - t[k]], fmt="none", ecolor=INK, elinewidth=1)
        right = t[f"{k}_hi"] if has_ci else t[k]
        for i, (v, r) in enumerate(zip(t[k], right)):
            ax.text(max(r, 0) + 0.02, i, f"{v:.2f}", va="center", fontsize=7.5, color=INK)
        ax.set_title(k.upper() if k in ("f1", "mcc") else k.capitalize())
        ax.set_xlim(min(0, t[k].min() - 0.05), 1.15)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(yy, t.index)
    fig.suptitle("Figure 5. Threshold metrics on the test set (Youden threshold from dev)", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "fig5_threshold_metrics.png"); plt.close(fig)


def fig_confusion(pred, best):
    y = pred.true_label.to_numpy()
    cm = confusion_matrix(y, pred[f"pred_{best}"])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}\n({v / cm[i].sum():.0%} of row)", ha="center", va="center",
                color="white" if v > cm.max() / 2 else INK, fontsize=9)
    ax.set_xticks([0, 1], ["Predicted No", "Predicted Yes"]); ax.set_yticks([0, 1], ["Actual No", "Actual Yes"])
    ax.grid(False)
    ax.set_title(f"Figure 6. Confusion matrix — {best}", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig6_confusion_best.png"); plt.close(fig)


def fig_association(assoc):
    a = assoc.sort_values("p_value", ascending=False)
    fig, ax = plt.subplots(figsize=(6.4, 0.28 * len(a) + 1.2))
    v = -np.log10(a.p_value.clip(lower=1e-300))
    ax.barh(np.arange(len(a)), v, height=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax.axvline(-np.log10(0.05 / len(a)), ls="--", lw=1, color=SERIES[7], label="Bonferroni p = 0.05")
    ax.axvline(-np.log10(0.05), ls=":", lw=1, color=INK2, label="Unadjusted p = 0.05")
    ax.set_yticks(np.arange(len(a)), a.feature, fontsize=8)
    ax.set(xlabel="−log10(p) of univariate association with the label",
           title="Figure 7. Feature–label association (χ² / Mann–Whitney)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig7_feature_association.png"); plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(RES / "test_predictions.csv")
    test = pd.read_csv(RES / "test_metrics.csv")
    cv = pd.read_csv(RES / "cv_metrics.csv").rename(columns={"Unnamed: 0": "model"})
    assoc = pd.read_csv(RES / "feature_association.csv")
    best = next(c[5:] for c in pred.columns if c.startswith("pred_"))
    models = _curve_models(test)
    fig_roc(pred, test, models)
    fig_pr(pred, test, models)
    fig_calibration(pred, test, models)
    fig_auc_forest(test, cv)
    fig_metric_bars(test)
    fig_confusion(pred, best)
    fig_association(assoc)
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
