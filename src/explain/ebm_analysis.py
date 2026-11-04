"""EBM term importances and shape functions; logistic-regression coefficients; agreement table."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.eval.plots import COLORS, style  # noqa: E402


def ebm_importance(pipe) -> pd.Series:
    """Main-effect term importances (mean |score|) of a fitted EBM pipeline; interactions are summed
    into each participating attribute with half weight each."""
    ebm = pipe.named_steps["clf"]
    imp = pd.Series(ebm.term_importances(), index=ebm.term_names_)
    out: dict[str, float] = {}
    for term, v in imp.items():
        parts = [p.strip() for p in term.split("&")]
        for p in parts:
            out[p] = out.get(p, 0.0) + (v if len(parts) == 1 else v / len(parts))
    return pd.Series(out).sort_values(ascending=False)


def ebm_shape_figure(pipe, features: list[str], out: Path) -> None:
    """Shape functions (log-odds contribution) of the given main-effect terms."""
    style()
    ebm = pipe.named_steps["clf"]
    glob = ebm.explain_global()
    names = list(ebm.term_names_)
    fig, axes = plt.subplots(1, len(features), figsize=(4.2 * len(features), 3.6))
    for ax, f in zip(np.atleast_1d(axes), features):
        d = glob.data(names.index(f))
        x, s = d["names"], np.asarray(d["scores"])
        if len(x) == len(s) + 1:  # continuous: bin edges -> step plot
            ax.step(x[:-1], s, where="post", color=COLORS[0])
        else:
            ax.bar([str(v) for v in x], s, color=COLORS[0], edgecolor="white", linewidth=2)
        ax.axhline(0, color="#52514e", lw=1)
        ax.set(title=f, ylabel="Contribution (log-odds)")
    fig.suptitle("EBM shape functions, top 3 attributes", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def lr_importance(pipe) -> pd.Series:
    """|standardised coefficient| per original attribute (max over one-hot levels)."""
    pre, lr = pipe.named_steps["pre"], pipe.named_steps["clf"]
    names = pre.get_feature_names_out()
    coef = pd.Series(np.abs(lr.coef_[0]), index=names)
    out: dict[str, float] = {}
    for n, v in coef.items():
        base = n.rsplit("_", 1)[0] if n.rsplit("_", 1)[-1].replace(".0", "").lstrip("-").isdigit() and "_" in n else n
        out[base] = max(out.get(base, 0.0), float(v))
    return pd.Series(out).sort_values(ascending=False)


def agreement_table(shap_imp: pd.Series, lr_imp: pd.Series, ebm_imp: pd.Series | None) -> tuple[pd.DataFrame, dict]:
    """Attribute ranks under each method and pairwise Spearman rank correlations."""
    df = pd.DataFrame({"shap_mean_abs": shap_imp, "lr_abs_std_coef": lr_imp})
    if ebm_imp is not None:
        df["ebm_importance"] = ebm_imp
    for c in list(df.columns):
        df[c.split("_")[0] + "_rank"] = df[c].rank(ascending=False).astype(int)
    df = df.sort_values("shap_rank").rename_axis("attribute").reset_index()
    ranks = [c for c in df.columns if c.endswith("_rank")]
    corr = {}
    for i, a in enumerate(ranks):
        for b in ranks[i + 1:]:
            r, p = spearmanr(df[a], df[b])
            corr[f"{a} vs {b}"] = (float(r), float(p))
    return df, corr
