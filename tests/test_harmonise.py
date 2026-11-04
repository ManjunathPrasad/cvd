"""Harmonisation of UCI rows to the Indian schema (synthetic rows; no real data needed)."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.harmonise import harmonise, map_indian_slope


def _uci():
    cols = load_config()["e3"]["uci_columns"]
    rows = [[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
            [67, 1, 4, 0, 0, 0, 2, 108, 1, 1.5, 2, np.nan, 3, 2],
            [41, 0, 2, 130, 204, 0, 0, 172, 0, 1.4, 1, 0, 3, 0]]
    return pd.DataFrame(rows, columns=cols)


def test_harmonise_recodes_and_renames():
    h = harmonise(_uci(), "Cleveland")
    assert list(h.chestpain) == [0, 3, 1]
    assert list(h.target) == [0, 1, 0]
    assert np.isnan(h.restingBP.iloc[1]) and np.isnan(h.serumcholestrol.iloc[1])
    assert set(load_config()["e3"]["shared_features"]) <= set(h.columns)
    assert "thal" not in h.columns


def test_harmonise_rejects_out_of_range_codes():
    bad = _uci(); bad.loc[0, "restecg"] = 5
    with pytest.raises(ValueError):
        harmonise(bad, "X")


def test_slope_mapping_must_be_agreed():
    df = pd.DataFrame({"slope": [0, 1, 2, 3]})
    with pytest.raises(ValueError):
        map_indian_slope(df, None)
    out = map_indian_slope(df, {0: None, 1: 1, 2: 2, 3: 3})
    assert np.isnan(out.slope.iloc[0]) and list(out.slope.iloc[1:]) == [1, 2, 3]
