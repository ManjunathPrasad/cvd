"""Stage 1: data audit and EDA figures. Writes results/data_audit.md and results/figures/eda_*.png.

Usage:  python -m src.data.audit
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config, path  # noqa: E402
from src.data.load import clean, read_raw  # noqa: E402
from src.eval.plots import COLORS, style  # noqa: E402


def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def run_audit() -> dict:
    """Run the audit, write the report and figures, return the key counts."""
    cfg = load_config()
    raw = read_raw()
    cleaned, log = clean(raw)
    tgt = cfg["columns"]["target"]
    cat_cols = cfg["columns"]["binary"] + cfg["columns"]["categorical"] + cfg["columns"]["ordinal"] + [tgt]
    num_cols = cfg["columns"]["numeric"]

    out = ["# Data audit — Mendeley Indian dataset", "",
           "Written by `src/data/audit.py`. Source: DOI 10.17632/dzz48mvjht.1.", "",
           "## Shape and types", "",
           f"- Raw shape: {raw.shape[0]} rows × {raw.shape[1]} columns",
           f"- Missing values in the raw file: {int(raw.isna().sum().sum())}",
           f"- Exact duplicate rows (ignoring `patientid`): {log['duplicates_dropped']} (dropped)",
           f"- Duplicate `patientid` values: {int(raw.patientid.duplicated().sum())}",
           f"- Rows after cleaning: {log['rows_out']}", "",
           _md_table(pd.DataFrame({"column": raw.columns, "dtype": [str(t) for t in raw.dtypes]})), "",
           "## Class balance (`target`)", ""]
    vc = raw[tgt].value_counts().sort_index()
    out += [_md_table(pd.DataFrame({"target": vc.index, "n": vc.values,
                                    "percent": (100 * vc.values / vc.sum()).round(1)})), ""]
    minority = vc.min() / vc.sum()
    rule = cfg["imbalance"]["min_minority_fraction"]
    out += [f"Minority class = {minority:.1%}. Rule (PLAN.md Section 3): class weights only if below {rule:.0%} → "
            + ("**no re-weighting**." if minority >= rule else "**class weights used**."), ""]

    out += ["## Numeric columns (raw)", ""]
    desc = raw[num_cols].describe().T[["min", "50%", "max", "mean", "std"]].round(2).reset_index()
    desc.columns = ["column", "min", "median", "max", "mean", "sd"]
    out += [_md_table(desc), ""]

    out += ["## Physiologically impossible values", ""]
    for col in cfg["impossible_zero"]:
        n = log[f"{col}_zero_to_missing"]
        out.append(f"- `{col} == 0`: {n} rows ({n / len(raw):.1%}) → set to missing; imputed inside the pipeline.")
    chol0 = raw[raw.serumcholestrol == 0]
    if len(chol0):
        out.append(f"- Target rate among `serumcholestrol == 0` rows: {chol0[tgt].mean():.1%} "
                   f"(overall {raw[tgt].mean():.1%}).")
    out.append("")

    out += ["## Categorical columns: value counts and target rate", ""]
    undocumented = []
    for col in cat_cols:
        g = raw.groupby(col)[tgt].agg(["count", "mean"]).reset_index()
        g.columns = [col, "n", "target_rate"]
        g["target_rate"] = g["target_rate"].round(3)
        allowed = cfg["allowed_codes"].get(col)
        if allowed is not None:
            g["documented"] = g[col].isin(allowed).map({True: "yes", False: "**NO**"})
            undocumented += [(col, v, n) for v, n in zip(g[col], g["n"]) if v not in allowed]
        out += [f"### `{col}`", "", _md_table(g), ""]

    out += ["## Codes outside the documented coding", ""]
    if undocumented:
        for col, v, n in undocumented:
            out.append(f"- `{col} == {v}`: {n} rows. Kept as its own category for RC1 and RC2.")
        if any(c == "slope" for c, _, _ in undocumented):
            s0 = raw[raw.slope == 0][tgt].mean()
            mapping = cfg["e3"].get("indian_slope_mapping")
            rc3 = (f"RC3 maps Indian slope codes onto UCI slope 1–3 as {mapping} (None = missing; "
                   "`e3.indian_slope_mapping` in configs/config.yaml)." if mapping is not None else
                   "**Decision needed before RC3:** how to map Indian slope codes onto UCI slope 1–3 "
                   "(set `e3.indian_slope_mapping` in configs/config.yaml).")
            out.append(f"- Target rate when `slope == 0`: {s0:.1%}. {rc3}")
    else:
        out.append("- None.")
    out.append("")

    fig_dir = path("figures")
    _eda_figures(raw, num_cols, cat_cols, tgt, fig_dir)
    out += ["## Figures", "", "- `results/figures/eda_numeric.png`: distributions of numeric attributes by target",
            "- `results/figures/eda_categorical.png`: target rate by category", ""]

    (path("results") / "data_audit.md").write_text("\n".join(out) + "\n")
    summary = {**log, "minority_fraction": float(minority),
               "undocumented": [(c, int(v), int(n)) for c, v, n in undocumented]}
    return summary


def _eda_figures(raw, num_cols, cat_cols, tgt, fig_dir) -> None:
    style()
    fig, axes = plt.subplots(1, len(num_cols), figsize=(3.2 * len(num_cols), 3))
    for ax, c in zip(axes, num_cols):
        for k, lab in [(0, "No disease"), (1, "Disease")]:
            ax.hist(raw.loc[raw[tgt] == k, c], bins=25, histtype="step", lw=2, color=COLORS[k], label=lab)
        ax.set_title(c); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Patients"); axes[0].legend(fontsize=8)
    fig.suptitle("Numeric attributes by target (raw values; zeros included)", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(fig_dir / "eda_numeric.png"); plt.close(fig)

    cols = [c for c in cat_cols if c != tgt]
    fig, axes = plt.subplots(1, len(cols), figsize=(2.4 * len(cols), 3), sharey=True)
    for ax, c in zip(axes, cols):
        g = raw.groupby(c)[tgt].mean()
        ax.bar(g.index.astype(str), g.values, color=COLORS[0], edgecolor="white", linewidth=2)
        ax.axhline(raw[tgt].mean(), ls="--", lw=1, color="#52514e")
        ax.set_title(c); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Disease rate")
    fig.suptitle("Disease rate by category (dashed = overall rate)", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(fig_dir / "eda_categorical.png"); plt.close(fig)


if __name__ == "__main__":
    s = run_audit()
    for k, v in s.items():
        print(f"{k}: {v}")
