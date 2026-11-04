"""Load configs/config.yaml and configs/search_spaces.yaml."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import os

import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Return the project configuration as a dict."""
    with open(ROOT / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    if os.environ.get("CVD_QUICK") == "1":  # smoke-test mode (inherited by parallel workers)
        cfg["tuning"]["n_trials"], cfg["tuning"]["n_trials_override"] = 2, {}
        cfg["cv"]["outer_folds"] = cfg["cv"]["inner_folds"] = 3
        cfg["evaluation"]["bootstrap_resamples"] = 100
        cfg["e3"]["adaptation_repeats"] = 2
        cfg["paths"]["results"] = "results/_quick"
        cfg["paths"]["figures"] = "results/_quick/figures"
        cfg["paths"]["models"] = "models/_quick"
    return cfg


@lru_cache(maxsize=1)
def load_search_spaces() -> dict:
    """Return the Optuna search spaces as a dict."""
    with open(ROOT / "configs" / "search_spaces.yaml") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for an entry under `paths:` in the config, created if it is a directory."""
    p = ROOT / load_config()["paths"][key]
    if p.suffix == "":
        p.mkdir(parents=True, exist_ok=True)
    return p


def all_features() -> list[str]:
    """The 12 modelling attributes (T4)."""
    return list(load_config()["tiers"]["T4"])
