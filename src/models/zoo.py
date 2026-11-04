"""Model zoo: one function that builds any of the nine models as an unfitted scikit-learn estimator."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import load_config
from src.features.preprocess import build_preprocessor, split_columns

SEED = load_config()["seed"]
TUNED = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]


def needs_class_weights(y: np.ndarray) -> bool:
    """PLAN.md Section 3: re-weight only if the minority class is below the configured fraction."""
    frac = min(np.mean(y), 1 - np.mean(y))
    return frac < load_config()["imbalance"]["min_minority_fraction"]


class CatBoostModel(ClassifierMixin, BaseEstimator):
    """CatBoost with native categorical handling, wrapped so scikit-learn can clone it."""

    def __init__(self, cat_features: tuple = (), params: dict | None = None, balanced: bool = False, n_jobs: int = 1):
        self.cat_features = cat_features
        self.params = params
        self.balanced = balanced
        self.n_jobs = n_jobs

    def _prep(self, X):
        X = X.copy()
        for c in self.cat_features:
            X[c] = X[c].astype(int)
        return X

    def fit(self, X, y):
        extra = {"auto_class_weights": "Balanced"} if self.balanced else {}
        self.model_ = CatBoostClassifier(cat_features=list(self.cat_features), random_seed=SEED, verbose=0,
                                         thread_count=self.n_jobs, allow_writing_files=False,
                                         **extra, **(self.params or {}))
        self.model_.fit(self._prep(X), y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._prep(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def tabpfn_status() -> tuple[bool, str]:
    """Whether TabPFN can be used here, and why not if it cannot."""
    try:
        from tabpfn import TabPFNClassifier  # noqa: F401
    except Exception as e:  # pragma: no cover - depends on environment
        return False, f"tabpfn could not be imported: {type(e).__name__}: {e}"
    try:
        import numpy as _np
        from tabpfn import TabPFNClassifier
        m = TabPFNClassifier(device="cpu", random_state=SEED)
        Xs = _np.random.default_rng(0).normal(size=(40, 3)); ys = (Xs[:, 0] > 0).astype(int)
        m.fit(Xs, ys); m.predict_proba(Xs)
    except Exception as e:  # pragma: no cover
        return False, f"TabPFN weights could not be loaded or run: {type(e).__name__}: {str(e)[:300]}"
    return True, "ok"


def make_model(name: str, params: dict | None, features: list[str], y_train: np.ndarray, n_jobs: int = 1):
    """Unfitted estimator (Pipeline with its preprocessor) for one of the single models."""
    params = dict(params or {})
    balanced = needs_class_weights(y_train)
    cw = "balanced" if balanced else None
    cats, _ = split_columns(features)
    if name == "LogisticRegression":
        return Pipeline([("pre", build_preprocessor("linear", features)),
                         ("clf", LogisticRegression(max_iter=5000, class_weight=cw, random_state=SEED, **params))])
    if name == "RandomForest":
        return Pipeline([("pre", build_preprocessor("tree", features)),
                         ("clf", RandomForestClassifier(class_weight=cw, random_state=SEED, n_jobs=n_jobs, **params))])
    if name == "XGBoost":
        spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1)) if balanced else 1.0
        return Pipeline([("pre", build_preprocessor("tree", features)),
                         ("clf", XGBClassifier(scale_pos_weight=spw, tree_method="hist", eval_metric="logloss",
                                               random_state=SEED, n_jobs=n_jobs, verbosity=0, **params))])
    if name == "LightGBM":
        if "subsample" in params:
            params.setdefault("subsample_freq", 1)
        return Pipeline([("pre", build_preprocessor("tree", features)),
                         ("clf", LGBMClassifier(class_weight=cw, random_state=SEED, n_jobs=n_jobs, verbose=-1,
                                                **params))])
    if name == "CatBoost":
        return Pipeline([("pre", build_preprocessor("tree", features)),
                         ("clf", CatBoostModel(cat_features=tuple(cats), params=params, balanced=balanced,
                                               n_jobs=n_jobs))])
    if name == "EBM":
        return Pipeline([("pre", build_preprocessor("tree", features)),
                         ("clf", ExplainableBoostingClassifier(random_state=SEED, n_jobs=n_jobs, **params))])
    if name == "TabPFN":
        from tabpfn import TabPFNClassifier
        return Pipeline([("pre", build_preprocessor("tree", features)),
                         ("clf", TabPFNClassifier(device="cpu", random_state=SEED, **params))])
    raise KeyError(f"Unknown model {name}")


def make_stacking(base: list[tuple[str, dict]], features: list[str], y_train: np.ndarray) -> StackingClassifier:
    """Stacking ensemble with a logistic-regression meta-learner and cv=5 (PLAN.md Section 3)."""
    ests = [(n, make_model(n, p, features, y_train)) for n, p in base]
    return StackingClassifier(ests, final_estimator=LogisticRegression(max_iter=5000), stack_method="predict_proba",
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED))


class StackedModel:
    """Fitted stacking ensemble: base pipelines + logistic-regression meta-learner on their probabilities."""

    def __init__(self, bases: list, meta: LogisticRegression):
        self.bases, self.meta = bases, meta

    def predict_proba(self, X):
        Z = np.column_stack([b.predict_proba(X)[:, 1] for b in self.bases])
        return self.meta.predict_proba(Z)
