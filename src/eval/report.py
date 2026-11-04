"""Formatting helpers for result tables (CSV, Markdown and LaTeX)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def ci(row: pd.Series, k: str, d: int = 3) -> str:
    """'0.912 (0.871–0.948)' from columns k, k_lo, k_hi."""
    return f"{row[k]:.{d}f} ({row[k + '_lo']:.{d}f}–{row[k + '_hi']:.{d}f})"


def to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def to_latex(df: pd.DataFrame, caption: str, label: str, out: Path) -> None:
    """Minimal booktabs LaTeX table (no pandas Styler/jinja2 dependency)."""
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$").replace("–", "--")
    cols = "l" + "c" * (len(df.columns) - 1)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(v) for v in r) + r" \\" for r in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines) + "\n")
