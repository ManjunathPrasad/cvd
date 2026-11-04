"""Load and clean the Mendeley Indian dataset (PLAN.md Section 2.1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, path

EXPECTED_COLUMNS = [
    "patientid", "age", "gender", "chestpain", "restingBP", "serumcholestrol", "fastingbloodsugar",
    "restingrelectro", "maxheartrate", "exerciseangia", "oldpeak", "slope", "noofmajorvessels", "target",
]


class DataValidationError(ValueError):
    """Raised when the CSV does not match PLAN.md Section 2.1."""


def read_raw(csv: Path | None = None) -> pd.DataFrame:
    """Read the raw CSV and validate its columns and codes. Raises DataValidationError on mismatch."""
    csv = csv or path("raw")
    if not Path(csv).exists():
        raise FileNotFoundError(f"{csv} not found. Download it manually (PLAN.md Section 2.1).")
    df = pd.read_csv(csv)
    validate_columns(df)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Check column names and that coded columns only contain known codes (slope 0 is allowed)."""
    if list(df.columns) != EXPECTED_COLUMNS:
        missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
        extra = sorted(set(df.columns) - set(EXPECTED_COLUMNS))
        raise DataValidationError(f"Column mismatch. Missing: {missing}. Unexpected: {extra}.")
    allowed = load_config()["allowed_codes"]
    for col, codes in allowed.items():
        ok = set(codes) | ({0} if col == "slope" else set())  # slope 0 is audited, not rejected
        bad = set(df[col].dropna().unique()) - ok
        if bad:
            raise DataValidationError(f"Column {col} has undocumented codes {sorted(bad)}.")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicates (ignoring patientid) and set impossible zeros to missing.
    Returns the cleaned frame and a log of what was changed."""
    cfg = load_config()
    log: dict = {"rows_in": len(df)}
    feats = [c for c in df.columns if c != cfg["columns"]["id"]]
    dup = df.duplicated(subset=feats)
    log["duplicates_dropped"] = int(dup.sum())
    out = df.loc[~dup].copy()
    for col in cfg["impossible_zero"]:
        n = int((out[col] == 0).sum())
        log[f"{col}_zero_to_missing"] = n
        out[col] = out[col].astype(float).where(out[col] != 0, np.nan)
    log["rows_out"] = len(out)
    return out.reset_index(drop=True), log


def load_xy(features: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Cleaned features (default: all 12) and target as a numpy array."""
    cfg = load_config()
    df, _ = clean(read_raw())
    features = features or list(cfg["tiers"]["T4"])
    return df[features].copy(), df[cfg["columns"]["target"]].to_numpy().astype(int)


def dev_test_split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stratified 80/20 dev/test split of row indices (seed from config). Used by every experiment."""
    from sklearn.model_selection import train_test_split

    cfg = load_config()
    idx = np.arange(len(y))
    dev, test = train_test_split(idx, test_size=cfg["split"]["test_size"], stratify=y, random_state=cfg["seed"])
    return np.sort(dev), np.sort(test)
