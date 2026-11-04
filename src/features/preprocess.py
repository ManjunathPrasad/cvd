"""Preprocessors. Everything that learns from data lives here and is fitted inside a Pipeline."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_config


def split_columns(features: list[str]) -> tuple[list[str], list[str]]:
    """Return (categorical, other) columns among `features`. Categorical = one-hot for logistic regression."""
    cats = [c for c in load_config()["columns"]["categorical"] if c in features]
    return cats, [c for c in features if c not in cats]


def build_preprocessor(model_family: str, features: list[str]) -> ColumnTransformer | Pipeline:
    """Preprocessing for a model family.

    - "linear": median imputation + scaling for numeric/binary/ordinal columns;
      mode imputation + one-hot for chestpain, restingrelectro and slope.
    - "tree": imputation only (median for numeric, mode for categorical); keeps column names.
    """
    cats, others = split_columns(features)
    if model_family == "linear":
        num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        return ColumnTransformer([("num", num, others), ("cat", cat, cats)], verbose_feature_names_out=False)
    if model_family == "tree":
        ct = ColumnTransformer([("num", SimpleImputer(strategy="median"), others),
                                ("cat", SimpleImputer(strategy="most_frequent"), cats)],
                               verbose_feature_names_out=False)
        return ct.set_output(transform="pandas")
    raise ValueError(f"Unknown model family: {model_family}")
