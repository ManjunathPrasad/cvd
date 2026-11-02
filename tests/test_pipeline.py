import numpy as np
import pytest

from src.run_youngsters import DATA, load_data, make_model

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="dataset not present")


def test_columns_and_label():
    X, y, cat, num = load_data()
    assert X.shape == (10000, 26) and set(np.unique(y)) == {0, 1}
    assert "Systolic BP" in num and "Gender" in cat


def test_preprocessor_fitted_on_training_rows_only():
    X, y, cat, num = load_data()
    tr = np.arange(500)
    m = make_model("LogisticRegression", {}, y[tr], cat, num).fit(X.iloc[tr], y[tr])
    scaler = m.named_steps["pre"].named_transformers_["num"]
    assert np.allclose(scaler.mean_, X.iloc[tr][num].mean().to_numpy())


def test_every_model_can_be_cloned_into_stacking():
    from sklearn.base import clone

    from src.run_youngsters import stacking_model

    X, y, cat, num = load_data()
    names = ["CatBoost", "RandomForest", "XGBoost", "LightGBM", "LogisticRegression", "EBM"]
    st = stacking_model([(n, {}) for n in names], y, cat, num)
    clone(st)
    small = stacking_model([("CatBoost", {"iterations": 20}), ("LogisticRegression", {})], y[:600], cat, num)
    p = small.fit(X.iloc[:600], y[:600]).predict_proba(X.iloc[600:700])[:, 1]
    assert p.shape == (100,)
