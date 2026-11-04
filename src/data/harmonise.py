"""Harmonise the UCI Heart Disease files to the Indian column names and codes (PLAN.md Section 2.2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, path

HARMONISATION = [
    ("sex", "gender", "same coding (1 male, 0 female)"),
    ("cp", "chestpain", "1–4 → 0–3 (subtract 1): typical, atypical, non-anginal, asymptomatic"),
    ("trestbps", "restingBP", "0 → missing"),
    ("chol", "serumcholestrol", "0 → missing (Switzerland codes all cholesterol as 0)"),
    ("fbs", "fastingbloodsugar", "same coding (1 if > 120 mg/dL)"),
    ("restecg", "restingrelectro", "same coding (0 normal, 1 ST-T abnormality, 2 LVH)"),
    ("thalach", "maxheartrate", "unchanged"),
    ("exang", "exerciseangia", "same coding (1 yes, 0 no)"),
    ("oldpeak", "oldpeak", "unchanged"),
    ("slope", "slope", "1 up, 2 flat, 3 down (kept); Indian codes mapped via config e3.indian_slope_mapping"),
    ("num", "target", "num > 0 → 1"),
    ("ca", "noofmajorvessels", "kept only for the India–Cleveland 12-attribute sensitivity analysis"),
    ("thal", "—", "dropped (not in the Indian data)"),
]
RANGES = {"gender": {0, 1}, "chestpain": {0, 1, 2, 3}, "fastingbloodsugar": {0, 1}, "restingrelectro": {0, 1, 2},
          "exerciseangia": {0, 1}, "slope": {1, 2, 3}, "target": {0, 1}, "noofmajorvessels": {0, 1, 2, 3}}


def read_uci(file: Path) -> pd.DataFrame:
    """Read one processed.*.data file (no header, '?' = missing)."""
    cols = load_config()["e3"]["uci_columns"]
    df = pd.read_csv(file, header=None, names=cols, na_values="?")
    if df.shape[1] != len(cols):
        raise ValueError(f"{file}: expected {len(cols)} columns, found {df.shape[1]}")
    return df


def harmonise(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Rename and recode UCI columns to the Indian schema; validate value ranges."""
    out = pd.DataFrame({
        "gender": df.sex, "chestpain": df.cp - 1, "restingBP": df.trestbps.replace(0, np.nan),
        "serumcholestrol": df.chol.replace(0, np.nan), "fastingbloodsugar": df.fbs, "restingrelectro": df.restecg,
        "maxheartrate": df.thalach, "exerciseangia": df.exang, "oldpeak": df.oldpeak, "slope": df.slope,
        "noofmajorvessels": df.ca, "target": (df.num > 0).astype(int), "age": df.age,
    })
    for col, ok in RANGES.items():
        bad = set(out[col].dropna().unique()) - ok
        if bad:
            raise ValueError(f"{cohort}: column {col} has values {sorted(bad)} outside {sorted(ok)}")
    out.insert(0, "cohort", cohort)
    return out


def map_indian_slope(df: pd.DataFrame, mapping: dict | None) -> pd.DataFrame:
    """Apply the agreed Indian→UCI slope mapping (values mapped to None become missing)."""
    if mapping is None:
        raise ValueError("e3.indian_slope_mapping is not set in configs/config.yaml. PLAN.md requires agreeing "
                         "how Indian slope codes (including 0) map onto UCI slope 1–3 before RC3.")
    out = df.copy()
    out["slope"] = out["slope"].map({int(k): (np.nan if v is None else v) for k, v in mapping.items()})
    return out


def load_uci() -> dict[str, pd.DataFrame]:
    """All four harmonised UCI cohorts. Raises FileNotFoundError listing any missing file."""
    cfg = load_config()["e3"]
    ext = path("external_dir")
    missing = [f for f in cfg["uci_files"].values() if not (ext / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing UCI files in {ext}: {missing}. Download them manually (PLAN.md 2.2).")
    return {c: harmonise(read_uci(ext / f), c) for c, f in cfg["uci_files"].items()}


def write_harmonisation_table(cohorts: dict[str, pd.DataFrame]) -> None:
    lines = ["# Harmonisation of UCI Heart Disease to the Indian schema", "",
             "| UCI column | Becomes | Rule |", "|---|---|---|"]
    lines += [f"| `{a}` | `{b}` | {r} |" for a, b, r in HARMONISATION]
    lines += ["", "## Cohort sizes and missingness after harmonisation", "",
              "| Cohort | Rows | Disease rate | " + " | ".join(f"missing {c}" for c in ["restingBP", "serumcholestrol", "slope", "noofmajorvessels"]) + " |",
              "|---|---|---|---|---|---|---|"]
    for c, d in cohorts.items():
        miss = " | ".join(f"{d[k].isna().mean():.0%}" for k in ["restingBP", "serumcholestrol", "slope", "noofmajorvessels"])
        lines.append(f"| {c} | {len(d)} | {d.target.mean():.1%} | {miss} |")
    (path("results") / "harmonisation_table.md").write_text("\n".join(lines) + "\n")
