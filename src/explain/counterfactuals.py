"""DiCE counterfactuals restricted to modifiable attributes (PLAN.md Section 5, E2)."""
from __future__ import annotations

import warnings

import dice_ml
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _dice_frame(X: pd.DataFrame, y: np.ndarray | None = None) -> pd.DataFrame:
    df = X.copy()
    df = df.fillna(df.median(numeric_only=True))
    if y is not None:
        df["target"] = y
    return df


def make_explainer(pipe, X_dev: pd.DataFrame, y_dev: np.ndarray, seed: int = 42) -> dice_ml.Dice:
    """DiCE explainer over a fitted sklearn pipeline. All attributes are declared continuous so DiCE passes
    numeric values straight to the pipeline; only the allowed attributes are ever changed."""
    data = dice_ml.Data(dataframe=_dice_frame(X_dev, y_dev), continuous_features=list(X_dev.columns),
                        outcome_name="target")
    model = dice_ml.Model(model=pipe, backend="sklearn")
    return dice_ml.Dice(data, model, method="random")


def counterfactuals(exp: dice_ml.Dice, query: pd.DataFrame, vary: list[str], ranges: dict[str, list],
                    total_cfs: int = 3, seed: int = 42) -> pd.DataFrame:
    """Counterfactuals that flip the prediction to class 0, changing only `vary` within `ranges`."""
    np.random.seed(seed)
    res = exp.generate_counterfactuals(query, total_CFs=total_cfs, desired_class=0, features_to_vary=vary,
                                       permitted_range=ranges, random_seed=seed)
    out = []
    for i, cf in enumerate(res.cf_examples_list):
        df = cf.final_cfs_df
        if df is None or len(df) == 0:
            continue
        df = df.copy(); df.insert(0, "query", i)
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
