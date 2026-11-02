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
