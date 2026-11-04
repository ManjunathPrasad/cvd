"""E3 (RC3): cross-population transfer, recalibration, local refitting and fairness. Writes Table 6.

Needs: the four UCI files in data/external/ and e3.indian_slope_mapping set in configs/config.yaml.
Usage:  python -m src.run_e3
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import load_config, path
from src.data.harmonise import load_uci, map_indian_slope, write_harmonisation_table
from src.data.load import clean, read_raw
from src.eval.metrics import calibration_intercept_slope, youden_threshold
from src.eval.report import to_latex
from src.models.tune import tune
from src.models.zoo import make_model

SINGLE = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost", "EBM"]


def _metrics(y, p) -> dict:
    a, s = calibration_intercept_slope(y, p)
    return {"auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p), "cal_intercept": a, "cal_slope": s}


def fairness(y, p, thr, df) -> list[dict]:
    """Sensitivity, specificity and calibration-in-the-large by sex and age band; equalised-odds difference."""
    cfg = load_config()["e3"]
    a1, a2 = cfg["age_bands"]
    groups = {"sex": df.gender.map({0: "female", 1: "male"}),
              "age": pd.cut(df.age, [-np.inf, a1 - 1e-9, a2, np.inf], labels=[f"<{a1}", f"{a1}-{a2}", f">{a2}"])}
    yhat = (p >= thr).astype(int)
    rows = []
    for attr, g in groups.items():
        tprs, fprs = [], []
        for level in g.dropna().unique():
            m = (g == level).to_numpy()
            yy, hh, pp = y[m], yhat[m], p[m]
            tpr = hh[yy == 1].mean() if (yy == 1).any() else np.nan
            fpr = hh[yy == 0].mean() if (yy == 0).any() else np.nan
            tprs.append(tpr); fprs.append(fpr)
            rows.append({"attribute": attr, "group": str(level), "n": int(m.sum()), "prevalence": yy.mean(),
                         "sensitivity": tpr, "specificity": 1 - fpr if not np.isnan(fpr) else np.nan,
                         "observed_minus_expected": yy.mean() - pp.mean()})
        eo = max(np.nanmax(tprs) - np.nanmin(tprs), np.nanmax(fprs) - np.nanmin(fprs))
        for r in rows:
            if r["attribute"] == attr:
                r["equalised_odds_difference"] = eo
    return rows


def transfer(name, src: pd.DataFrame, tgt: pd.DataFrame, feats, label: str, rng_seed: int) -> tuple[dict, list, list]:
    """Train on src (tuned by inner CV), evaluate internally (inner OOF) and on tgt; then adapt."""
    cfg = load_config()
    e3 = cfg["e3"]
    Xs, ys, Xt, yt = src[feats], src.target.to_numpy(), tgt[feats], tgt.target.to_numpy()
    params, oof = tune(name, Xs, ys, feats)
    m = make_model(name, params, feats, ys).fit(Xs, ys)
    thr = youden_threshold(ys, oof)
    p = m.predict_proba(Xt)[:, 1]
    row = {"direction": label, "model": name, "n_source": len(ys), "n_target": len(yt),
           **{f"internal_{k}": v for k, v in _metrics(ys, oof).items()},
           **{f"external_{k}": v for k, v in _metrics(yt, p).items()}}
    row["delta_auc"] = row["external_auc"] - row["internal_auc"]

    adapt = []
    for frac in e3["adaptation_fractions"]:
        for r in range(e3["adaptation_repeats"]):
            lab, rest = train_test_split(np.arange(len(yt)), train_size=frac, stratify=yt, random_state=rng_seed + r)
            pl, pr = p[lab], p[rest]
            lp = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
            platt = LogisticRegression().fit(lp(pl).reshape(-1, 1), yt[lab])
            iso = IsotonicRegression(out_of_bounds="clip").fit(pl, yt[lab])
            Xr = pd.concat([Xs, Xt.iloc[lab]]); yr = np.r_[ys, yt[lab]]
            refit = make_model(name, params, feats, yr).fit(Xr, yr)
            for method, q in [("none", pr), ("platt", platt.predict_proba(lp(pr).reshape(-1, 1))[:, 1]),
                              ("isotonic", iso.predict(pr)), ("refit", refit.predict_proba(Xt.iloc[rest])[:, 1])]:
                adapt.append({"direction": label, "model": name, "fraction": frac, "repeat": r, "method": method,
                              **_metrics(yt[rest], q)})
    fair = [{"direction": label, "model": name, **f} for f in fairness(yt, p, thr, tgt)]
    return row, adapt, fair


def main(log=print) -> None:
    t0 = time.time()
    cfg = load_config()
    e3 = cfg["e3"]
    res_dir = path("results")
    uci = load_uci()
    write_harmonisation_table(uci)
    india, _ = clean(read_raw())
    india = map_indian_slope(india, e3["indian_slope_mapping"]).assign(cohort="India")
    cohorts = {"India": india, **uci}
    feats = list(e3["shared_features"])
    summary = json.loads((res_dir / "e1_summary.json").read_text())
    models = [m for m in summary["cv_ranking"] if m in SINGLE][: e3["n_models"]]
    log(f"E3: models {models}; cohorts {[(c, len(d)) for c, d in cohorts.items()]}")

    rows, adapt, fair = [], [], []
    pooled = pd.concat(list(uci.values()), ignore_index=True)
    plan = [(f"India → {c}", india, d) for c, d in uci.items()] + [("Pooled UCI → India", pooled, india)]
    for held in cohorts:
        src = pd.concat([d for c, d in cohorts.items() if c != held], ignore_index=True)
        plan.append((f"LOCO: rest → {held}", src, cohorts[held]))
    for name in models:
        for label, src, tgt in plan:
            r, a, f = transfer(name, src, tgt, feats, label, cfg["seed"])
            rows.append(r); adapt += a; fair += f
            log(f"  {name:18s} {label:28s} internal {r['internal_auc']:.3f} → external {r['external_auc']:.3f}")
        # sensitivity analysis: India ↔ Cleveland with all 12 attributes
        f12 = list(cfg["tiers"]["T4"])
        r, _, _ = transfer(name, india, uci["Cleveland"], f12, "India → Cleveland (12 attributes)", cfg["seed"])
        rows.append(r)

    t6 = pd.DataFrame(rows)
    t6.to_csv(res_dir / "table6_transfer.csv", index=False)
    ad = pd.DataFrame(adapt)
    ad.to_csv(res_dir / "e3_adaptation_runs.csv", index=False)
    ad.groupby(["direction", "model", "fraction", "method"])[["auc", "brier", "cal_slope", "cal_intercept"]] \
      .agg(["mean", "std"]).to_csv(res_dir / "e3_adaptation_summary.csv")
    pd.DataFrame(fair).to_csv(res_dir / "e3_fairness.csv", index=False)
    disp = pd.DataFrame({"Direction": t6.direction, "Model": t6.model,
                         "Internal AUC": t6.internal_auc.round(3), "External AUC": t6.external_auc.round(3),
                         "ΔAUC": t6.delta_auc.round(3), "Cal. slope": t6.external_cal_slope.round(2),
                         "Brier": t6.external_brier.round(3)})
    to_latex(disp, "Cross-population transfer: internal vs external performance.", "tab:transfer",
             res_dir / "table6_transfer.tex")
    log(f"E3 done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
