"""Tests required by PLAN.md Section 9: column validation, tier definitions, no leakage."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config, path
from src.data.load import DataValidationError, clean, dev_test_split, validate_columns, EXPECTED_COLUMNS

HAS_DATA = path("raw").exists()


def _frame(n=20):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "patientid": np.arange(n), "age": rng.integers(20, 80, n), "gender": rng.integers(0, 2, n),
        "chestpain": rng.integers(0, 4, n), "restingBP": rng.integers(94, 200, n),
        "serumcholestrol": rng.integers(0, 600, n), "fastingbloodsugar": rng.integers(0, 2, n),
        "restingrelectro": rng.integers(0, 3, n), "maxheartrate": rng.integers(71, 202, n),
        "exerciseangia": rng.integers(0, 2, n), "oldpeak": rng.uniform(0, 6, n).round(1),
        "slope": rng.integers(0, 4, n), "noofmajorvessels": rng.integers(0, 4, n), "target": rng.integers(0, 2, n),
    })[EXPECTED_COLUMNS]


def test_validation_rejects_wrong_columns():
    with pytest.raises(DataValidationError):
        validate_columns(_frame().drop(columns=["slope"]))
    with pytest.raises(DataValidationError):
        validate_columns(_frame().rename(columns={"oldpeak": "st_depression"}))


def test_validation_rejects_undocumented_codes_but_allows_slope_zero():
    df = _frame(); df.loc[0, "slope"] = 0
    validate_columns(df)
    df.loc[0, "chestpain"] = 7
    with pytest.raises(DataValidationError):
        validate_columns(df)


def test_clean_drops_duplicates_and_sets_zero_cholesterol_missing():
    df = _frame(); df.loc[1, df.columns[1:]] = df.loc[0, df.columns[1:]].values; df.loc[2, "serumcholestrol"] = 0
    out, log = clean(df)
    assert log["duplicates_dropped"] >= 1
    assert out.serumcholestrol.isna().sum() == (df.drop_duplicates(subset=df.columns[1:]).serumcholestrol == 0).sum()


def test_tiers_are_cumulative_and_match_plan():
    t = load_config()["tiers"]
    assert t["T1"] == ["age", "gender", "chestpain", "restingBP"]
    assert t["T2"] == t["T1"] + ["serumcholestrol", "fastingbloodsugar", "restingrelectro"]
    assert t["T3"] == t["T2"] + ["maxheartrate", "exerciseangia", "oldpeak", "slope"]
    assert t["T4"] == t["T3"] + ["noofmajorvessels"]
    assert set(t["T4"]) == set(EXPECTED_COLUMNS) - {"patientid", "target"}


@pytest.mark.parametrize("name", ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost", "EBM"])
def test_preprocessor_fitted_on_training_rows_only(name):
    """Imputation statistics must come from the training rows, never from validation/test rows."""
    from src.models.zoo import make_model

    df = _frame(200).astype(float)
    df.loc[df.index[:100], "serumcholestrol"] = 100.0   # training rows
    df.loc[df.index[100:], "serumcholestrol"] = 500.0   # held-out rows
    df.loc[[3, 150], "serumcholestrol"] = np.nan
    feats = load_config()["tiers"]["T4"]
    tr = np.arange(100)
    m = make_model(name, {}, feats, df.target.to_numpy()[tr]).fit(df.loc[tr, feats], df.target.to_numpy()[tr])
    pre = m.named_steps["pre"]
    num = pre.named_transformers_["num"]
    imputer = num.named_steps["impute"] if hasattr(num, "named_steps") else num
    others = [c for c in feats if c not in load_config()["columns"]["categorical"]]
    assert imputer.statistics_[others.index("serumcholestrol")] == 100.0


def test_dev_test_split_is_stratified_and_disjoint():
    y = np.array([0] * 420 + [1] * 580)
    dev, test = dev_test_split(y)
    assert len(set(dev) & set(test)) == 0 and len(test) == 200
    assert abs(y[test].mean() - 0.58) < 0.01


def test_net_benefit_treat_all_matches_formula():
    from src.eval.dca import decision_curve
    y = np.array([1, 1, 0, 0, 0]); p = np.ones(5)
    d = decision_curve(y, {"all": p}, 0.2, 0.2)
    nb_model = d[d.strategy == "all"].net_benefit.iloc[0]
    nb_all = d[d.strategy == "Treat all"].net_benefit.iloc[0]
    assert np.isclose(nb_model, nb_all) and np.isclose(nb_all, 0.4 - 0.6 * 0.25)


@pytest.mark.skipif(not HAS_DATA, reason="Mendeley CSV not present")
def test_real_data_loads_with_expected_shape():
    from src.data.load import load_xy
    X, y = load_xy()
    assert X.shape == (1000, 12) and set(np.unique(y)) == {0, 1}
