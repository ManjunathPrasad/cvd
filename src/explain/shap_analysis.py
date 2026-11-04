"""SHAP explanations for a fitted tree-ensemble pipeline (preprocessor -> tree model)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from src.eval.plots import style  # noqa: E402
from src.models.zoo import CatBoostModel  # noqa: E402

TREE_MODELS = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]


def shap_explanation(pipe, X: pd.DataFrame) -> shap.Explanation:
    """SHAP values (positive class, log-odds or probability scale as the model provides) for rows X."""
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Xt = pre.transform(X)
    model = clf.model_ if isinstance(clf, CatBoostModel) else clf
    data = clf._prep(Xt) if isinstance(clf, CatBoostModel) else Xt
    ex = shap.TreeExplainer(model)
    sv = ex(data)
    if sv.values.ndim == 3:  # random forest: (n, features, classes)
        sv = sv[:, :, 1]
    base = np.asarray(sv.base_values, dtype=float)
    base = np.full(len(Xt), float(base)) if base.ndim == 0 else base.reshape(len(Xt), -1)[:, -1]
    return shap.Explanation(values=np.asarray(sv.values, dtype=float), base_values=base,
                            data=Xt.to_numpy(float), feature_names=list(Xt.columns))


def mean_abs_shap(exp: shap.Explanation) -> pd.Series:
    return pd.Series(np.abs(exp.values).mean(axis=0), index=exp.feature_names).sort_values(ascending=False)


def shap_figures(exp: shap.Explanation, model_name: str, fig_dir: Path, tp_idx: int | None, fn_idx: int | None,
                 prefix: str = "e1") -> list[str]:
    """Beeswarm (Figure 4), bar, dependence for top 3, waterfalls for one TP and one FN."""
    style()
    written = []
    plt.figure()
    shap.plots.beeswarm(exp, show=False, max_display=12)
    plt.title(f"Figure 4. SHAP summary — {model_name} (test set)", loc="left", fontsize=10)
    f = fig_dir / "fig4_shap_summary.png"; plt.savefig(f, bbox_inches="tight", dpi=150); plt.close("all"); written.append(f.name)

    plt.figure()
    shap.plots.bar(exp, show=False, max_display=12)
    plt.title(f"Mean |SHAP| — {model_name}", loc="left", fontsize=10)
    f = fig_dir / f"{prefix}_shap_bar.png"; plt.savefig(f, bbox_inches="tight", dpi=150); plt.close("all"); written.append(f.name)

    top = list(mean_abs_shap(exp).index[:3])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, feat in zip(axes, top):
        shap.plots.scatter(exp[:, feat], ax=ax, show=False, color="#2a78d6")
        ax.set_title(feat)
    fig.suptitle(f"SHAP dependence, top 3 attributes — {model_name}", x=0.01, ha="left")
    f = fig_dir / f"{prefix}_shap_dependence_top3.png"; fig.savefig(f, bbox_inches="tight", dpi=150); plt.close("all"); written.append(f.name)

    for label, idx in [("true_positive", tp_idx), ("false_negative", fn_idx)]:
        if idx is None:
            continue
        plt.figure()
        shap.plots.waterfall(exp[idx], show=False, max_display=12)
        plt.title(f"SHAP waterfall — {label.replace('_', ' ')} (test row {idx})", loc="left", fontsize=10)
        f = fig_dir / f"{prefix}_shap_waterfall_{label}.png"; plt.savefig(f, bbox_inches="tight", dpi=150); plt.close("all"); written.append(f.name)
    return written
