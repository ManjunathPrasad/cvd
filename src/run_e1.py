"""E1 (RC1): model comparison on all 12 attributes. Writes Table 4, Figures 1–4, sanity checks and explanations.

Usage:  python -m src.run_e1
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, path
from src.data.load import dev_test_split, load_xy
from src.eval import plots, sanity
from src.eval.metrics import paired_bootstrap_auc
from src.eval.nested_cv import ProtocolResult, run_protocol, test_table
from src.eval.report import ci, to_latex, to_markdown
from src.explain.ebm_analysis import agreement_table, ebm_importance, ebm_shape_figure, lr_importance
from src.explain.shap_analysis import TREE_MODELS, mean_abs_shap, shap_explanation, shap_figures
from src.models.zoo import tabpfn_status


def available_models(log=print) -> list[str]:
    """Models from config; TabPFN is dropped (and the reason logged) if it cannot run here."""
    models = list(load_config()["models"])
    if "TabPFN" in models:
        ok, why = tabpfn_status()
        (path("results") / "tabpfn_status.md").write_text(
            "# TabPFN status\n\n" + ("TabPFN ran.\n" if ok else f"TabPFN was **skipped**: {why}\n\n"
            "PLAN.md Section 3: if installation or the weights licence blocks, skip it and log why. "
            "The TabPFN weights are gated on Hugging Face: accept the licence at "
            "https://huggingface.co/Prior-Labs and set HF_TOKEN, then `pip install tabpfn` and re-run.\n"))
        if not ok:
            log(f"  TabPFN skipped: {why[:120]}")
            models.remove("TabPFN")
    return models


def table4(test: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    """Paper Table 4 (display strings)."""
    cvi = cv.set_index("model")
    rows = []
    for _, r in test.iterrows():
        c = cvi.loc[r.model]
        rows.append({
            "Model": r.model,
            "CV ROC-AUC": f"{c.roc_auc_mean:.3f} ± {c.roc_auc_std:.3f}",
            "Test ROC-AUC (95% CI)": ci(r, "roc_auc"),
            "PR-AUC (95% CI)": ci(r, "pr_auc"),
            "Sensitivity (95% CI)": ci(r, "sensitivity"),
            "Specificity (95% CI)": ci(r, "specificity"),
            "F1 (95% CI)": ci(r, "f1"),
            "MCC (95% CI)": ci(r, "mcc"),
            "Brier (95% CI)": ci(r, "brier"),
            "Cal. intercept": f"{r.cal_intercept:.2f}",
            "Cal. slope": f"{r.cal_slope:.2f}",
        })
    return pd.DataFrame(rows)


def explain(res: ProtocolResult, cvt: pd.DataFrame, X_test: pd.DataFrame, y_test: np.ndarray, log=print) -> dict:
    """SHAP for the best tree ensemble; EBM shape functions; LR coefficients; agreement table."""
    fig_dir = path("figures")
    tree_rank = [m for m in cvt.model if m in TREE_MODELS]
    best_tree = tree_rank[0]
    pipe = res.fitted[best_tree]
    exp = shap_explanation(pipe, X_test[res.features])
    p, thr = res.test_probs[best_tree], res.thresholds[best_tree]
    yhat = (p >= thr).astype(int)
    tp = np.where((yhat == 1) & (y_test == 1))[0]
    fn = np.where((yhat == 0) & (y_test == 1))[0]
    tp_idx = int(tp[np.argmax(p[tp])]) if len(tp) else None
    fn_idx = int(fn[np.argmin(p[fn])]) if len(fn) else None
    written = shap_figures(exp, best_tree, fig_dir, tp_idx, fn_idx)

    shap_imp = mean_abs_shap(exp)
    lr_imp = lr_importance(res.fitted["LogisticRegression"])
    ebm_imp = None
    if "EBM" in res.fitted:
        ebm_imp = ebm_importance(res.fitted["EBM"])
        top_main = [t for t in ebm_imp.index if t in res.features][:3]
        ebm_shape_figure(res.fitted["EBM"], top_main, fig_dir / "e1_ebm_shape_functions.png")
        written.append("e1_ebm_shape_functions.png")
    agree, corr = agreement_table(shap_imp, lr_imp, ebm_imp)
    agree.to_csv(path("results") / "feature_importance_agreement.csv", index=False)
    log(f"  SHAP on {best_tree}; top attributes: {list(shap_imp.index[:5])}")
    return {"best_tree": best_tree, "tp_idx": tp_idx, "fn_idx": fn_idx, "figures": written,
            "rank_correlations": corr, "shap_top": list(shap_imp.index[:5])}


def run_sanity(res: ProtocolResult, best: str, test: pd.DataFrame, X_dev, y_dev, X_test, y_test, log=print) -> None:
    """PLAN.md Section 4 step 7. Always run; required if any test AUC >= the configured trigger."""
    cfg = load_config()
    trig = cfg["evaluation"]["sanity_auc_trigger"]
    feats = res.features
    params = res.final_params[best]
    nd = sanity.near_duplicates(X_dev[feats], X_test[feats])
    perm = sanity.permutation_test(best, params, feats, X_dev, y_dev, X_test, y_test,
                                   n_repeats=cfg["evaluation"]["permutation_repeats"], seed=cfg["seed"])
    drop = sanity.drop_one(best, params, feats, X_dev, y_dev, X_test, y_test)
    drop.to_csv(path("results") / "sanity_drop_one.csv", index=False)
    raw = X_dev.assign(target=y_dev)
    lines = ["# Sanity checks (PLAN.md Section 4, step 7)", "",
             f"Highest test ROC-AUC: {test.roc_auc.max():.3f} ({test.iloc[0].model}). Trigger: {trig}. "
             + ("**Checks required.**" if test.roc_auc.max() >= trig else "Checks run for completeness."), "",
             f"Checks use `{best}` (best single model by nested-CV ROC-AUC) with its tuned hyperparameters.", "",
             "## 1. Near-duplicate rows shared between dev and test", "",
             f"- Test rows identical to a dev row (all 12 attributes): {nd['exact_matches']}",
             f"- Test rows within {nd['tol']} × the typical nearest-neighbour distance of a dev row: {nd['near_matches']}",
             f"- Median distance test→nearest dev row: {nd['median_nn_distance_test_to_dev']:.3f}; "
             f"within dev: {nd['median_nn_distance_within_dev']:.3f} (standardised units)", "",
             "## 2. Label-permutation test", "",
             f"- Test ROC-AUC after training on shuffled dev labels ({len(perm)} repeats): "
             f"mean {np.mean(perm):.3f}, range {min(perm):.3f}–{max(perm):.3f} (expected ≈ 0.5)", "",
             "## 3. Removing one attribute at a time", "",
             to_markdown(drop.assign(test_auc=drop.test_auc.round(3), delta_vs_full=drop.delta_vs_full.round(3))), "",
             "## 4. Attributes that nearly determine the label (dev set)", ""]
    for col in ["slope", "chestpain", "noofmajorvessels", "restingrelectro"]:
        if col in feats:
            g = raw.groupby(col)["target"].agg(["count", "mean"]).round(3).reset_index()
            g.columns = [col, "n", "disease_rate"]
            lines += [f"`{col}`:", "", to_markdown(g), ""]
    lines += [f"Rows with cholesterol recorded as 0 (set to missing before modelling): disease rate "
              f"{raw[raw.serumcholestrol.isna()].target.mean():.1%} (n = {int(raw.serumcholestrol.isna().sum())}).", ""]
    worst = drop.iloc[0]
    lines += ["## Interpretation (generated from the numbers above)", "",
              f"- The largest single-attribute drop comes from removing `{worst.removed}` "
              f"(ΔAUC {worst.delta_vs_full:+.3f}).",
              f"- Permutation AUC ≈ {np.mean(perm):.2f} shows the pipeline does not leak labels; "
              f"{nd['exact_matches']} exact dev–test duplicates.", ""]
    (path("results") / "sanity.md").write_text("\n".join(lines) + "\n")
    log(f"  sanity: perm AUC {np.mean(perm):.3f}, exact dups {nd['exact_matches']}, "
        f"largest drop {worst.removed} {worst.delta_vs_full:+.3f}")


def main(log=print) -> dict:
    t0 = time.time()
    cfg = load_config()
    res_dir, fig_dir, model_dir = path("results"), path("figures"), path("models")
    X, y = load_xy()
    dev, tst = dev_test_split(y)
    X_dev, y_dev, X_test, y_test = X.iloc[dev].reset_index(drop=True), y[dev], X.iloc[tst].reset_index(drop=True), y[tst]
    feats = list(cfg["tiers"]["T4"])
    log(f"E1: dev {len(y_dev)}, test {len(y_test)}, prevalence {y.mean():.3f}")

    models = available_models(log)
    res = run_protocol(models, X_dev, y_dev, X_test, feats, log=log)
    cvt = res.cv_table()
    test = test_table(res, y_test)

    pd.DataFrame(res.cv_rows).to_csv(res_dir / "cv_metrics_per_fold.csv", index=False)
    cvt.to_csv(res_dir / "cv_metrics.csv", index=False)
    test.to_csv(res_dir / "test_metrics.csv", index=False)
    t4 = table4(test, cvt)
    t4.to_csv(res_dir / "table4_models.csv", index=False)
    to_latex(t4[["Model", "CV ROC-AUC", "Test ROC-AUC (95% CI)", "PR-AUC (95% CI)", "Sensitivity (95% CI)",
                 "Specificity (95% CI)", "MCC (95% CI)", "Brier (95% CI)", "Cal. slope"]],
             "Discrimination and calibration of all models on the held-out Indian test set (n = %d); "
             "95%% bootstrap CIs." % len(y_test), "tab:models", res_dir / "table4_models.tex")

    best = next(m for m in cvt.model if m in models)  # best single model by CV AUC
    comp = []
    for ref in ["LogisticRegression", "RandomForest"]:
        if ref != best:
            d, lo, hi, pv = paired_bootstrap_auc(y_test, res.test_probs[best], res.test_probs[ref],
                                                 n=cfg["evaluation"]["bootstrap_resamples"], seed=cfg["seed"])
            comp.append({"comparison": f"{best} − {ref}", "delta_auc": d, "ci_lo": lo, "ci_hi": hi, "p_value": pv})
    pd.DataFrame(comp).to_csv(res_dir / "model_comparison.csv", index=False)

    preds = pd.DataFrame({"test_row": tst, "target": y_test})
    for m, p in res.test_probs.items():
        preds[f"prob_{m}"] = np.round(p, 5)
    preds.to_csv(res_dir / "test_predictions_e1.csv", index=False)

    # Figures 1–3: LR (reference) + the four best other models by CV AUC
    shown = ["LogisticRegression"] + [m for m in cvt.model if m != "LogisticRegression"][:4]
    shown = [m for m in plots.MODEL_ORDER if m in shown] + [m for m in shown if m not in plots.MODEL_ORDER]
    curves = {m: (y_test, res.test_probs[m]) for m in shown}
    ti = test.set_index("model")
    plots.plot_roc(curves, ti.roc_auc.to_dict(), "Figure 1. ROC curves, Indian test set", fig_dir / "fig1_roc.png")
    plots.plot_pr(curves, ti.pr_auc.to_dict(), "Figure 2. Precision–recall curves, test set", fig_dir / "fig2_pr.png")
    plots.plot_calibration(curves, ti.brier.to_dict(), "Figure 3. Calibration curves, test set",
                           fig_dir / "fig3_calibration.png")
    plots.plot_auc_forest(test, "E1: ROC-AUC by model", fig_dir / "e1_auc_forest.png", cv=cvt)

    run_sanity(res, best, test, X_dev, y_dev, X_test, y_test, log)
    ex = explain(res, cvt, X_test, y_test, log)

    # Demo model (full T4): the explained tree ensemble
    joblib.dump({"model": res.fitted[ex["best_tree"]], "name": ex["best_tree"], "features": feats,
                 "threshold": res.thresholds[ex["best_tree"]], "tier": "T4"}, model_dir / "demo_T4.joblib")
    with open(res_dir / "e1_summary.json", "w") as f:
        json.dump({"models": models, "best_by_cv": best, "top3": res.top3, "final_params": res.final_params,
                   "thresholds": res.thresholds, "sens90_thresholds": res.sens90, "explanations": ex,
                   "cv_ranking": list(cvt.model), "timings_s": res.timings, "runtime_min": (time.time() - t0) / 60},
                  f, indent=2, default=float)
    joblib.dump(res, model_dir / "e1_protocol_result.joblib")
    log(f"E1 done in {(time.time() - t0) / 60:.1f} min. Best by CV: {best}; "
        f"best test AUC {test.roc_auc.max():.3f} ({test.iloc[0].model}).")
    return {"cv": cvt, "test": test, "best": best, "top": list(cvt.model)}


if __name__ == "__main__":
    main()
