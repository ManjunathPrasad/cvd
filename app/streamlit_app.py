"""Streamlit demo: CVD risk from the full (T4) or screening (T2) model.

Run:  streamlit run app/streamlit_app.py
Research prototype, not for clinical use. All metrics shown are read from results/ (never hard-coded).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import load_config, path  # noqa: E402
from src.data.load import dev_test_split, load_xy  # noqa: E402

st.set_page_config(page_title="CVD risk — research prototype", layout="wide")
st.error("**Research prototype, not for clinical use.** Predictions are illustrative and must not guide care.")

CFG = load_config()
LABELS = {
    "age": "Age (years)", "gender": "Sex", "chestpain": "Chest-pain type", "restingBP": "Resting blood pressure (mm Hg)",
    "serumcholestrol": "Serum cholesterol (mg/dL)", "fastingbloodsugar": "Fasting blood sugar > 120 mg/dL",
    "restingrelectro": "Resting ECG", "maxheartrate": "Maximum heart rate, exercise test (bpm)",
    "exerciseangia": "Exercise-induced angina", "oldpeak": "ST depression, exercise vs rest (mm)",
    "slope": "Slope of peak-exercise ST segment", "noofmajorvessels": "Major vessels coloured by fluoroscopy",
}
CHOICES = {
    "gender": {"Male": 1, "Female": 0},
    "chestpain": {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3},
    "fastingbloodsugar": {"No": 0, "Yes": 1},
    "restingrelectro": {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2},
    "exerciseangia": {"No": 0, "Yes": 1},
    "slope": {"Upsloping (1)": 1, "Flat (2)": 2, "Downsloping (3)": 3, "Code 0 (undocumented in source)": 0},
    "noofmajorvessels": {"0": 0, "1": 1, "2": 2, "3": 3},
}
NUMERIC = {"age": (20, 80, 50, 1), "restingBP": (94, 200, 130, 1), "serumcholestrol": (85, 602, 240, 1),
           "maxheartrate": (71, 202, 150, 1), "oldpeak": (0.0, 6.2, 1.0, 0.1)}


@st.cache_resource
def load_bundle(tier: str):
    f = path("models") / f"demo_{tier}.joblib"
    return joblib.load(f) if f.exists() else None


@st.cache_data
def dev_data():
    X, y = load_xy()
    dev, _ = dev_test_split(y)
    return X.iloc[dev].reset_index(drop=True), y[dev]


def metrics_for(bundle) -> pd.DataFrame | None:
    """Test metrics with CIs for the selected model, read from results/."""
    res = path("results")
    if bundle["tier"] == "T4" and (res / "table4_models.csv").exists():
        t = pd.read_csv(res / "table4_models.csv")
        row = t[t.Model == bundle["name"]]
        if len(row):
            return row[["Test ROC-AUC (95% CI)", "Sensitivity (95% CI)", "Specificity (95% CI)", "Brier (95% CI)"]].T
    if (res / "table5_tiers.csv").exists():
        t = pd.read_csv(res / "table5_tiers.csv")
        row = t[(t.model == bundle["name"]) & (t.tier == bundle["tier"])]
        if len(row):
            r = row.iloc[0]
            return pd.DataFrame({"value": [f"{r.roc_auc:.3f} ({r.roc_auc_lo:.3f}–{r.roc_auc_hi:.3f})",
                                           f"{r.sensitivity:.3f}", f"{r.specificity:.3f}", f"{r.brier:.3f}"]},
                                index=["Test ROC-AUC (95% CI)", "Sensitivity", "Specificity", "Brier"])
    return None


tier_label = st.radio("Model", ["Full model (T4: all 12 attributes, needs fluoroscopy)",
                                "Screening model (T2: history, vitals, basic labs and resting ECG)"], horizontal=True)
tier = "T4" if tier_label.startswith("Full") else "T2"
bundle = load_bundle(tier)
if bundle is None:
    st.warning(f"models/demo_{tier}.joblib not found. Run `python -m src.run_all --stages audit e1 e2` first.")
    st.stop()
feats = bundle["features"]

with st.sidebar:
    st.header("About")
    st.markdown("**Data:** Indian multispecialty-hospital dataset, 1,000 patients (Mendeley Data, "
                "DOI 10.17632/dzz48mvjht.1). Model trained on the 80% development split.")
    st.markdown(f"**Model:** {bundle['name']} ({bundle['tier']}, {len(feats)} attributes)")
    m = metrics_for(bundle)
    if m is not None:
        st.markdown("**Held-out test performance** (from `results/`):")
        st.dataframe(m, use_container_width=True)
    st.markdown("**Limitations:** single-centre retrospective data; not externally validated; the source data "
                "contains codes (e.g. ST slope 0) that almost determine the label, so performance may not transfer "
                "to other hospitals. Risk bands are illustrative, not clinical thresholds.")

st.subheader("Patient attributes")
cols = st.columns(3)
values = {}
for i, f in enumerate(feats):
    with cols[i % 3]:
        if f in CHOICES:
            opt = st.selectbox(LABELS[f], list(CHOICES[f]))
            values[f] = CHOICES[f][opt]
        else:
            lo, hi, default, step = NUMERIC[f]
            values[f] = st.number_input(LABELS[f], min_value=lo, max_value=hi, value=default, step=step)
x = pd.DataFrame([values])[feats].astype(float)

model = bundle["model"]
p = float(model.predict_proba(x)[:, 1][0])
bands = CFG["demo"]["risk_bands"]
band = "Low" if p < bands["low"] else ("Moderate" if p <= bands["high"] else "High")

c1, c2 = st.columns([1, 2])
with c1:
    st.metric("Predicted probability of heart disease", f"{p:.1%}")
    st.markdown(f"**Illustrative risk band:** {band}  \n(Low < {bands['low']:.0%}, Moderate {bands['low']:.0%}–"
                f"{bands['high']:.0%}, High > {bands['high']:.0%}; illustrative only)")
    st.caption(f"Model decision threshold (Youden, from development data): {bundle['threshold']:.2f}")

with c2:
    st.markdown("**Why this prediction (SHAP, probability scale)**")
    Xd, yd = dev_data()
    bg = Xd[feats].fillna(Xd[feats].median()).sample(100, random_state=CFG["seed"])
    f_ = lambda a: model.predict_proba(pd.DataFrame(a, columns=feats))[:, 1]
    explainer = shap.Explainer(f_, bg, algorithm="permutation", seed=CFG["seed"])
    sv = explainer(x.to_numpy(), max_evals=4 * len(feats) + 1)
    sv.feature_names = [LABELS[f] for f in feats]
    fig = plt.figure()
    shap.plots.waterfall(sv[0], show=False, max_display=12)
    st.pyplot(plt.gcf(), clear_figure=True)
    plt.close("all")

if tier == "T2":
    st.subheader("What could lower the predicted risk? (counterfactual)")
    if p < bundle["threshold"]:
        st.info("The prediction is already below the model's decision threshold.")
    elif x[["restingBP", "serumcholestrol", "fastingbloodsugar"]].isna().any(axis=None):
        st.info("Counterfactuals need blood pressure, cholesterol and fasting sugar.")
    else:
        from src.explain.counterfactuals import counterfactuals, make_explainer

        vary = list(CFG["e2"]["counterfactual_features"])
        lows = {"restingBP": float(Xd.restingBP.min()), "serumcholestrol": float(Xd.serumcholestrol.min()),
                "fastingbloodsugar": 0.0}
        ranges = {f: [lows[f], float(x[f].iloc[0])] for f in vary}
        try:
            cfs = counterfactuals(make_explainer(model, Xd[feats], yd), x, vary, ranges, total_cfs=3)
        except Exception:
            cfs = pd.DataFrame()
        if cfs.empty:
            st.info("No combination of lower blood pressure, lower cholesterol and normal fasting sugar moves this "
                    "patient below the decision threshold.")
        else:
            cfs = cfs[feats].copy()
            cfs["predicted probability"] = model.predict_proba(cfs)[:, 1].round(3)
            show = cfs[vary + ["predicted probability"]].rename(columns=LABELS)
            st.dataframe(show, use_container_width=True)
            st.caption("Only resting blood pressure, cholesterol and fasting blood sugar may change, and only towards "
                       "healthier values. Illustrative; not a treatment recommendation.")
