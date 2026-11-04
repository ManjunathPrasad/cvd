"""E2 (RC2): test availability tiers, decision curves and counterfactuals. Writes Table 5, Figure 5.

Usage:  python -m src.run_e2      (needs E1 results: python -m src.run_e1)
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, path
from src.data.load import dev_test_split, load_xy
from src.eval import plots
from src.eval.dca import decision_curve, net_benefit
from src.eval.metrics import paired_bootstrap_auc
from src.eval.nested_cv import ProtocolResult, run_protocol, test_table
from src.eval.report import ci, to_latex, to_markdown
from src.explain.counterfactuals import counterfactuals, make_explainer

SINGLE = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost", "TabPFN", "EBM"]


def subset(res: ProtocolResult, models: list[str]) -> ProtocolResult:
    """Keep only `models` from a protocol result (used to reuse E1 as the T4 tier)."""
    keep = lambda d: {k: v for k, v in d.items() if k in models}
    return ProtocolResult(features=res.features, cv_rows=[r for r in res.cv_rows if r["model"] in models],
                          fold_params=keep(res.fold_params), final_params=keep(res.final_params),
                          thresholds=keep(res.thresholds), sens90=keep(res.sens90), dev_oof=keep(res.dev_oof),
                          test_probs=keep(res.test_probs), fitted=keep(res.fitted), timings=keep(res.timings))


def main(log=print) -> dict:
    t0 = time.time()
    cfg = load_config()
    e2 = cfg["e2"]
    res_dir, fig_dir, model_dir = path("results"), path("figures"), path("models")
    e1_file = model_dir / "e1_protocol_result.joblib"
    if not e1_file.exists():
        raise FileNotFoundError("Run E1 first: python -m src.run_e1")
    e1 = joblib.load(e1_file)
    summary = json.loads((res_dir / "e1_summary.json").read_text())
    models = [m for m in summary["cv_ranking"] if m in SINGLE][: e2["n_models"]]
    log(f"E2: best {len(models)} E1 models by CV ROC-AUC: {models}")

    X, y = load_xy()
    dev, tst = dev_test_split(y)
    X_dev, y_dev, X_test, y_test = X.iloc[dev].reset_index(drop=True), y[dev], X.iloc[tst].reset_index(drop=True), y[tst]

    results: dict[str, ProtocolResult] = {}
    for tier, feats in cfg["tiers"].items():
        if tier == "T4":
            results[tier] = subset(e1, models)
            log(f"  {tier}: reused from E1")
            continue
        log(f"  {tier}: {len(feats)} attributes")
        results[tier] = run_protocol(models, X_dev, y_dev, X_test, list(feats), ensembles=False, log=log)

    # ---- Table 5
    rows, nb_rows = [], []
    n_boot, seed = cfg["evaluation"]["bootstrap_resamples"], cfg["seed"]
    for tier, res in results.items():
        cvt = res.cv_table().set_index("model")
        tt = test_table(res, y_test).set_index("model")
        for m in models:
            r = tt.loc[m]
            d, lo, hi, pv = (0.0, 0.0, 0.0, 1.0) if tier == "T4" else paired_bootstrap_auc(
                y_test, res.test_probs[m], results["T4"].test_probs[m], n=n_boot, seed=seed)
            rows.append({"tier": tier, "n_attributes": len(res.features), "model": m,
                         "cv_auc_mean": cvt.loc[m, "roc_auc_mean"], "cv_auc_std": cvt.loc[m, "roc_auc_std"],
                         **{k: r[k] for k in ["roc_auc", "roc_auc_lo", "roc_auc_hi", "pr_auc", "sensitivity",
                                              "specificity", "brier", "cal_intercept", "cal_slope", "mcc"]},
                         "delta_auc_vs_T4": d, "delta_lo": lo, "delta_hi": hi, "delta_p": pv,
                         "net_benefit_at_%.2f" % e2["dca_prespecified"]: net_benefit(y_test, res.test_probs[m],
                                                                                     e2["dca_prespecified"])})
    t5 = pd.DataFrame(rows)
    t5.to_csv(res_dir / "table5_tiers.csv", index=False)
    disp = pd.DataFrame({
        "Tier": t5.tier, "Model": t5.model,
        "CV ROC-AUC": [f"{a:.3f} ± {b:.3f}" for a, b in zip(t5.cv_auc_mean, t5.cv_auc_std)],
        "Test ROC-AUC (95% CI)": [ci(r, "roc_auc") for _, r in t5.iterrows()],
        "ΔAUC vs T4 (95% CI)": ["—" if r.tier == "T4" else f"{r.delta_auc_vs_T4:+.3f} ({r.delta_lo:+.3f} to {r.delta_hi:+.3f})"
                                for _, r in t5.iterrows()],
        "Brier": t5.brier.round(3), "Cal. slope": t5.cal_slope.round(2),
        "NB@%.2f" % e2["dca_prespecified"]: t5["net_benefit_at_%.2f" % e2["dca_prespecified"]].round(3),
    })
    disp.to_csv(res_dir / "table5_tiers_display.csv", index=False)
    to_latex(disp, "Performance by feature tier on the Indian test set; ΔAUC by paired bootstrap.",
             "tab:tiers", res_dir / "table5_tiers.tex")

    # ---- Figure 5: decision curves of the best model (by T4 CV AUC) at each tier
    best = models[0]
    probs = {f"{tier} ({len(r.features)} attributes)": r.test_probs[best] for tier, r in results.items()}
    lo, hi = e2["dca_thresholds"]
    dc = decision_curve(y_test, probs, lo, hi)
    dc.to_csv(res_dir / "decision_curves.csv", index=False)
    plots.plot_decision_curves(dc, f"Figure 5. Decision curves by tier — {best}, test set", fig_dir / "fig5_decision_curves.png")
    auc_rows = t5[t5.model == best][["tier", "roc_auc", "roc_auc_lo", "roc_auc_hi"]].rename(columns={"tier": "model"})
    plots.plot_auc_forest(auc_rows, f"E2: test ROC-AUC by tier — {best}", fig_dir / "e2_auc_by_tier.png")

    # ---- Counterfactuals on the best T2 model
    t2 = results["T2"]
    t2_cv = t2.cv_table()
    best_t2 = t2_cv.model.iloc[0]
    pipe, thr, feats2 = t2.fitted[best_t2], t2.thresholds[best_t2], t2.features
    cf_md = counterfactual_report(pipe, thr, feats2, best_t2, X_dev, y_dev, X_test, tst, e2, log)
    (res_dir / "counterfactual_examples.md").write_text(cf_md)

    joblib.dump({"model": pipe, "name": best_t2, "features": feats2, "threshold": thr, "tier": "T2"},
                model_dir / "demo_T2.joblib")
    with open(res_dir / "e2_summary.json", "w") as f:
        json.dump({"models": models, "dca_model": best, "best_T2": best_t2,
                   "final_params": {t: r.final_params for t, r in results.items()},
                   "thresholds": {t: r.thresholds for t, r in results.items()},
                   "runtime_min": (time.time() - t0) / 60}, f, indent=2, default=float)
    log(f"E2 done in {(time.time() - t0) / 60:.1f} min. Best T2 model: {best_t2}")
    return {"table5": t5, "best_T2": best_t2}


def counterfactual_report(pipe, thr, feats, name, X_dev, y_dev, X_test, test_rows, e2, log=print) -> str:
    """DiCE counterfactuals for test patients predicted positive. Only the modifiable attributes may change,
    and only downwards (lower BP, lower cholesterol, normal fasting sugar)."""
    vary = list(e2["counterfactual_features"])
    Xd, Xt = X_dev[feats], X_test[feats]
    exp = make_explainer(pipe, Xd, y_dev)
    p = pipe.predict_proba(Xt)[:, 1]
    lows = {"restingBP": float(Xd.restingBP.min()), "serumcholestrol": float(Xd.serumcholestrol.min()),
            "fastingbloodsugar": 0.0}
    cand = [i for i in np.argsort(p) if p[i] >= thr and not Xt.iloc[i][vary].isna().any()]
    lines = [f"# Counterfactual examples — {name} (T2 screening model)", "",
             "Generated by `src/run_e2.py` with DiCE (random method). Only resting blood pressure, serum cholesterol "
             "and fasting blood sugar may change, and only towards healthier values (each value may fall to the "
             "lowest value seen in the development data). Research prototype, not for clinical use.", "",
             f"Decision threshold (Youden, from development data): {thr:.3f}.", ""]
    found = 0
    for i in cand:
        q = Xt.iloc[[i]].copy()
        ranges = {f: [lows[f], float(q[f].iloc[0])] for f in vary}
        if all(r[0] >= r[1] for r in ranges.values()):
            continue
        try:
            cfs = counterfactuals(exp, q, vary, ranges, total_cfs=3)
        except Exception as e:  # DiCE raises when no counterfactual exists in range
            log(f"  DiCE: test row {int(test_rows[i])}: {type(e).__name__}")
            continue
        if cfs.empty:
            continue
        cfs = cfs.drop(columns=["query"])
        cfs["predicted_risk"] = pipe.predict_proba(cfs[feats])[:, 1].round(3)
        found += 1
        lines += [f"## Example {found}: test patient (CSV row {int(test_rows[i])}), predicted risk {p[i]:.3f}", "",
                  "Patient:", "", to_markdown(q.reset_index(drop=True)), "",
                  "Counterfactuals (changed attributes only):", ""]
        changed = cfs[vary + ["predicted_risk"]].copy()
        for f in vary:
            changed[f] = [f"{v:g} (was {q[f].iloc[0]:g})" if v != q[f].iloc[0] else f"{v:g}" for v in changed[f]]
        lines += [to_markdown(changed), ""]
        if found >= e2["n_counterfactual_patients"]:
            break
    if found < e2["n_counterfactual_patients"]:
        lines += [f"Only {found} predicted-positive test patients had a counterfactual within the allowed ranges: "
                  "for the others, the modifiable attributes alone cannot lower the predicted risk below the threshold.", ""]
    log(f"  counterfactuals: {found} patients")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
