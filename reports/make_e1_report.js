// Builds reports/E1_results.docx from the files in results/ (run: node reports/make_e1_report.js).
// Every number in the tables is read from results/*.csv; numbers quoted in the prose were copied from the same files.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, AlignmentType, HeadingLevel,
  WidthType, ShadingType, BorderStyle, LevelFormat, PageNumber, Footer, Header, PageBreak,
} = require("docx");

const ROOT = path.join(__dirname, "..");
const R = (f) => path.join(ROOT, "results", f);
const FIG = (f) => path.join(ROOT, "results", "figures", f);

function csv(file) {
  const [head, ...rows] = fs.readFileSync(R(file), "utf8").trim().split(/\r?\n/);
  const cols = head.split(",");
  return rows.map((r) => Object.fromEntries(r.split(",").map((v, i) => [cols[i], v])));
}
const f3 = (x) => Number(x).toFixed(3);
const f2 = (x) => Number(x).toFixed(2);
const pct = (x) => (100 * Number(x)).toFixed(1) + "%";

const test = csv("test_metrics.csv");
const cv = Object.fromEntries(csv("cv_metrics.csv").map((r) => [r.model, r]));
const imp = csv("feature_importance_agreement.csv");
const drop = csv("sanity_drop_one.csv");
const NAME = { LogisticRegression: "Logistic regression", RandomForest: "Random forest", SoftVoting: "Soft voting",
  Stacking: "Stacking", LightGBM: "LightGBM", XGBoost: "XGBoost", CatBoost: "CatBoost", EBM: "EBM" };

// ---------- styling helpers
const FONT = "Times New Roman";
const PAGE_W = 11906, MARGIN = 1134, CONTENT = PAGE_W - 2 * MARGIN; // A4, 2 cm margins
const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const borders = { top: border, bottom: border, left: border, right: border };

const p = (text, opts = {}) => new Paragraph({ spacing: { after: 120, line: 276 }, alignment: opts.align,
  children: (Array.isArray(text) ? text : [text]).map((t) => (typeof t === "string" ? new TextRun(t) : t)) });
const b = (t) => new TextRun({ text: t, bold: true });
const i = (t) => new TextRun({ text: t, italics: true });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (parts) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60, line: 276 },
  children: (Array.isArray(parts) ? parts : [parts]).map((t) => (typeof t === "string" ? new TextRun(t) : t)) });
const caption = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200 },
  children: [new TextRun({ text: t, italics: true, size: 24, color: "000000" })] });

function table(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, c) => a + c, 0);
  const cell = (text, isHead, w, bold) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: undefined,
    margins: { top: 50, bottom: 50, left: 70, right: 70 },
    children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: String(text), bold: isHead || bold, size: 24, color: "000000", font: FONT })] })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, k) => cell(h, true, widths[k])) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, k) => cell(c, false, widths[k], opts.highlight && opts.highlight(ri))) }))],
  });
}

function figure(file, widthPx, caps) {
  const buf = fs.readFileSync(FIG(file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20); // PNG header
  const maxW = Math.min(widthPx, 620);
  return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: maxW, height: Math.round(maxW * h / w) },
      altText: { title: file, description: caps, name: file } })] }), caption(caps)];
}

// ---------- tables
const order = test.map((r) => r.model);
const t1 = test.map((r) => [NAME[r.model], `${f3(cv[r.model].roc_auc_mean)} ± ${f3(cv[r.model].roc_auc_std)}`,
  `${f3(r.roc_auc)} (${f3(r.roc_auc_lo)}–${f3(r.roc_auc_hi)})`, pct(r.accuracy),
  `${f3(r.sensitivity)}`, `${f3(r.specificity)}`, f3(r.f1), f3(r.mcc), f3(r.brier)]);
const t2 = test.map((r) => [NAME[r.model], f3(r.threshold), f2(r.cal_intercept), f2(r.cal_slope),
  f3(r.sens90_threshold), f3(r.sens90_sensitivity), f3(r.sens90_specificity)]);
const t3 = order.map((m) => { const c = cv[m]; return [NAME[m], f3(c.roc_auc_mean), f3(c.pr_auc_mean), pct(c.accuracy_mean),
  f3(c.sensitivity_mean), f3(c.specificity_mean), f3(c.mcc_mean), f3(c.brier_mean)]; });
const t4 = imp.map((r) => [r.attribute, r.shap_rank, r.ebm_rank, r.lr_rank, f2(r.shap_mean_abs)]);
const t5 = drop.slice(0, 6).map((r) => [r.removed, f3(r.test_auc), Number(r.delta_vs_full).toFixed(3)]);

const sv = test.find((r) => r.model === "SoftVoting");
const ebm = test.find((r) => r.model === "EBM");
const lr = test.find((r) => r.model === "LogisticRegression");

// ---------- document
const children = [
  new Paragraph({ heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER, children: [new TextRun("Experiment E1: Results and Discussion")] }),
  p([new TextRun({ text: "Interpretable machine learning for cardiovascular disease prediction in Indian patients", size: 24, color: "000000" })]),
  p([new TextRun({ text: "Manjunath Prasad · Department of AIML, MITE · Research prototype, not for clinical use", size: 24, color: "000000" })]),

  h1("1. What E1 asks"),
  p("E1 (research contribution RC1) compares six single models and two ensembles for predicting cardiovascular disease from the 12 routinely recorded attributes of the Indian multispecialty-hospital dataset (Mendeley Data, DOI 10.17632/dzz48mvjht.1; 1,000 patients, 580 with disease), and asks which attributes drive the predictions."),

  h2("1.1 Protocol"),
  bullet([b("Split: "), "stratified 80/20 split into a development set (800 patients) and a held-out test set (200 patients: 116 with disease, 84 without). The test set was used once, at the end."]),
  bullet([b("Models: "), "logistic regression (LR), random forest (RF), XGBoost, LightGBM, CatBoost and the Explainable Boosting Machine (EBM); soft voting and stacking of the three best single models by cross-validated ROC-AUC (CatBoost, LightGBM, random forest)."]),
  bullet([b("Internal validation: "), "nested cross-validation on the development set: 5 outer folds for performance, and inside each, 5 inner folds with 40 Optuna (TPE) trials maximising ROC-AUC."]),
  bullet([b("No leakage: "), "imputation, scaling and one-hot encoding are fitted inside each training fold as part of a scikit-learn Pipeline. Cholesterol values of 0 (53 patients) were treated as missing."]),
  bullet([b("Thresholds: "), "Youden's J on the out-of-fold predictions of the development set (never the test set), plus a second threshold that gives sensitivity ≥ 0.90 on the development data."]),
  bullet([b("Metrics: "), "ROC-AUC, PR-AUC, accuracy, sensitivity, specificity, F1, MCC, Brier score, calibration intercept and slope. 95% confidence intervals come from 1,000 bootstrap resamples of the test set. Random seed 42 throughout."]),
  bullet([b("Explanations: "), "SHAP (TreeExplainer) on the best single model, EBM shape functions, and standardised LR coefficients."]),

  h1("2. Results"),
  h2("2.1 Test-set performance"),
  p("Table 1 gives the performance of every model on the 200 held-out patients at its Youden threshold, alongside its nested-CV ROC-AUC on the development set."),
  table(["Model", "CV ROC-AUC (mean ± SD)", "Test ROC-AUC (95% CI)", "Acc.", "Sens.", "Spec.", "F1", "MCC", "Brier"],
    t1, [1500, 1250, 1650, 850, 850, 850, 850, 850, 988], { highlight: (k) => k < 2 }),
  caption("Table 1. E1 performance, Indian test set (n = 200), at each model's Youden threshold. Acc. = accuracy, Sens. = sensitivity, Spec. = specificity. Sorted by test ROC-AUC; the first two rows (bold) are the ensembles."),
  p([b("Discrimination is near-perfect for every model. "), `All eight models reach a test ROC-AUC of at least ${f3(lr.roc_auc)}, and accuracy between ${pct(Math.min(...test.map((r) => r.accuracy)))} and ${pct(Math.max(...test.map((r) => r.accuracy)))}. Soft voting and stacking have the highest ROC-AUC (${f3(sv.roc_auc)}, 95% CI ${f3(sv.roc_auc_lo)}–${f3(sv.roc_auc_hi)}); each misclassifies 2 of 200 test patients (1 missed case and 1 false alarm). EBM has the highest accuracy (${pct(ebm.accuracy)}): it finds all 116 patients with disease and raises 1 false alarm.`]),
  p([b("The differences between models are not meaningful. "), "The confidence intervals of all eight models overlap almost completely. A paired bootstrap of the ROC-AUC difference found no significant gap between the best single model (CatBoost) and logistic regression (ΔAUC +0.0005, 95% CI −0.0002 to +0.0019, p = 0.43) or random forest (ΔAUC +0.0001, 95% CI −0.0016 to +0.0019, p = 1.0). A simple logistic regression is therefore as good as the gradient-boosted and ensemble models on this dataset."]),

  h2("2.2 Cross-validation on the development set"),
  table(["Model", "ROC-AUC", "PR-AUC", "Acc.", "Sens.", "Spec.", "MCC", "Brier"], t3,
    [1600, 1150, 1150, 1150, 1150, 1150, 1100, 1188], { highlight: (k) => k < 2 }),
  caption("Table 2. Nested 5-fold cross-validation (means over outer folds), development set (n = 800). Abbreviations as in Table 1."),
  p("Cross-validated performance is slightly lower than test performance (ROC-AUC 0.991–0.997, accuracy 95.8%–97.5%), which is expected: the CV estimate averages five smaller validation folds, while the test set is a single 200-patient sample on which most models make only 1–3 errors. The CV estimate is the more conservative figure to quote. The model ranking by CV (ensembles first, then CatBoost, LightGBM, random forest and EBM tied at 0.996, then XGBoost and LR) is the same as on the test set within the uncertainty."),

  h2("2.3 Calibration and operating points"),
  table(["Model", "Youden threshold", "Cal. intercept", "Cal. slope", "Sens. ≥ 0.90 threshold", "Sens. there", "Spec. there"], t2,
    [1600, 1250, 1250, 1150, 1500, 1400, 1488]),
  caption("Table 3. Calibration on the test set, and the high-sensitivity operating point chosen on the development data."),
  p("Brier scores are all below 0.02, so the predicted probabilities are close to the observed outcomes. The calibration slopes vary widely (from 0.88 for LightGBM to 3.29 for random forest) because almost every test patient receives a probability near 0 or near 1; with so few patients in between, the slope is estimated from a handful of points and is unstable. The Brier score and the calibration curve (Figure 3) are more reliable guides here. Using the threshold chosen on the development data to reach sensitivity ≥ 0.90, every model keeps sensitivity between 0.93 and 0.98 on the test set with specificity of 0.99–1.00."),

  h2("2.4 Which attributes drive the predictions"),
  table(["Attribute", "SHAP rank (CatBoost)", "EBM rank", "LR rank", "Mean |SHAP|"], t4, [2300, 1900, 1700, 1700, 2038]),
  caption("Table 4. Attribute importance from three independent methods (1 = most important)."),
  p([b("All three methods agree that the ST-segment slope is by far the most important attribute. "), "Its mean absolute SHAP value (3.15) is more than twice that of the next attribute. Resting blood pressure and chest-pain type follow in SHAP and EBM; resting ECG, oldpeak and sex complete the top six. Fasting blood sugar, age and exercise-induced angina contribute least in all three methods."]),

  h2("2.5 Sanity checks"),
  p("Because test ROC-AUC exceeded the pre-specified trigger of 0.98, the checks in PLAN.md Section 4 step 7 were run (using CatBoost, the best single model by CV):"),
  bullet([b("No duplicated patients: "), "0 test patients are identical to, or within 0.05 of the typical nearest-neighbour distance of, any development patient."]),
  bullet([b("No label leakage: "), "after training on randomly shuffled labels, test ROC-AUC averaged 0.497 over 20 repeats (range 0.302–0.686), which is chance level. The pipeline cannot reproduce the result without the true labels."]),
  bullet([b("No single attribute is responsible: "), "removing any one attribute and retraining lowers the test ROC-AUC by at most 0.004 (Table 5)."]),
  table(["Attribute removed", "Test ROC-AUC", "Change vs all 12"], t5, [3200, 3000, 3000]),
  caption("Table 5. Test ROC-AUC after removing one attribute at a time (six largest drops)."),
  bullet([b("A data artefact: "), "in the development set, every patient with slope = 0 (144 patients) is disease-free and every patient with slope = 3 (163) has disease; all 43 development patients with cholesterol recorded as 0 have disease. Slope 0 is not described in the dataset's documentation."]),

  h1("3. Figures and how to read them"),
  ...figure("fig1_roc.png", 330, "Figure 1. ROC curves on the test set."),
  p("Each curve shows the trade-off between sensitivity (patients with disease correctly flagged) and 1 − specificity (healthy patients wrongly flagged) as the threshold moves. A useless model follows the dashed diagonal; a perfect one runs up the left edge and along the top. All curves hug the top-left corner: the models can rank almost every patient with disease above every patient without it (AUC 0.999–1.000)."),
  ...figure("fig2_pr.png", 330, "Figure 2. Precision–recall curves on the test set."),
  p("Precision is the fraction of flagged patients who truly have disease; recall is sensitivity. The dashed line is the precision of flagging everyone (the 58% disease prevalence). All curves stay at precision ≈ 1.0 until recall approaches 1.0 (average precision 0.999–1.000): almost no false alarms are needed to catch nearly all cases."),
  ...figure("fig3_calibration.png", 330, "Figure 3. Calibration curves (top) and distribution of predicted probabilities (bottom)."),
  p("If a model predicts 80% risk for a group of patients, about 80% of them should have disease; a perfectly calibrated model lies on the dashed diagonal. All curves lie close to it (Brier 0.006–0.013). The histogram shows why the calibration slope is unstable: almost all predictions sit below 0.05 or above 0.95, so the middle of the curve is estimated from very few patients."),
  ...figure("e1_auc_forest.png", 560, "Figure 4. ROC-AUC by model: test set (filled, 95% bootstrap CI) and nested CV (open, mean ± SD)."),
  p("This forest plot puts all eight models on one axis. Every interval overlaps, so no model is clearly better than another. The CV estimates (open circles) sit slightly to the left of the test results and have wider spread, especially for logistic regression and XGBoost; they are the more conservative estimate of performance on new patients from the same hospital."),
  ...figure("fig4_shap_summary.png", 480, "Figure 5. SHAP summary for CatBoost on the test set."),
  p("Each dot is one test patient. Its horizontal position is how much that attribute pushed the prediction towards disease (right) or away from it (left), on the log-odds scale; colour is the attribute's value (red high, blue low). Slope separates patients most strongly: low slope values (blue) push risk sharply down and high values (red) push it up by about +3. High resting blood pressure, chest-pain types 1–3, abnormal resting ECG and high cholesterol also raise risk. Oldpeak runs the opposite way to clinical expectation: high values (red) slightly lower predicted risk in this dataset."),
  ...figure("e1_shap_dependence_top3.png", 620, "Figure 6. SHAP dependence for the three most important attributes (CatBoost)."),
  p("These plots show how the contribution of each attribute changes with its value. Slope has four distinct levels: 0 lowers the log-odds by about 4.5, 1 by about 2.5, while 2 and 3 raise them by about 3, a near step-function that reflects the artefact noted in Section 2.5. Resting blood pressure lowers risk by a roughly constant amount below about 125 mmHg, its contribution rises steeply between 130 and 160 mmHg and then plateaus. Chest-pain type 0 (typical angina) lowers risk and types 1–3 raise it."),
  ...figure("e1_ebm_shape_functions.png", 620, "Figure 7. EBM shape functions for its three most important attributes."),
  p("The EBM is a glass-box model: its prediction is the sum of one curve per attribute, so these curves are the model itself rather than an approximation of it. They agree with SHAP on CatBoost: slope is a staircase (codes 0, 1, 2 and 3 correspond to the steps at 0.5, 1.5 and 2.5 on the axis), and resting blood pressure raises risk sharply above about 145 mmHg. The oldpeak curve confirms the counter-intuitive direction seen in Figure 5: values above about 2.3 lower the predicted risk."),
  ...figure("e1_shap_waterfall_true_positive.png", 470, "Figure 8a. SHAP explanation for one correctly detected patient (true positive)."),
  ...figure("e1_shap_waterfall_false_negative.png", 470, "Figure 8b. SHAP explanation for one missed patient (false negative)."),
  p("A waterfall explains one prediction. It starts from the average model output E[f(X)] = 1.08 log-odds and adds each attribute's contribution to reach the patient's score f(x). In the true positive (8a), slope = 3 (+2.86), an abnormal resting ECG (+1.51), chest-pain type 2 (+1.12) and resting blood pressure 162 mmHg (+1.12) push the score to 8.85, a near-certain prediction of disease. In the missed case (8b), chest-pain type 2 raises the score (+2.92), but slope = 1 (−2.08), a normal resting blood pressure of 126 mmHg (−1.68) and a normal resting ECG (−0.73) outweigh it, leaving the score at −0.33, just on the 'no disease' side. This is the kind of explanation the demo shows for each patient."),

  h1("4. Discussion"),
  h2("4.1 Main findings"),
  p("On this dataset, every model, from a regularised logistic regression to stacked gradient-boosting ensembles, separates patients with and without cardiovascular disease almost perfectly (test ROC-AUC 0.999–1.000, accuracy 98.5%–99.5%, nested-CV ROC-AUC 0.991–0.997). The differences between models are smaller than their uncertainty, and the paired bootstrap finds no significant advantage of the best boosting model over logistic regression. For deployment, this argues for the simplest and most transparent model that performs as well: logistic regression or the EBM, whose shape functions can be shown to a clinician directly."),
  p("The explanations are consistent across three independent methods (SHAP on CatBoost, EBM, LR coefficients): ST-segment slope dominates, followed by resting blood pressure, chest-pain type and resting ECG. Most directions agree with clinical knowledge (high blood pressure, symptomatic chest pain, abnormal ECG and high cholesterol increase risk)."),

  h2("4.2 Why the scores are so high, and what that means"),
  p("Near-perfect discrimination is unusual for cardiovascular risk prediction from routine attributes, so the sanity checks matter. They rule out the common technical causes: there are no duplicated patients between development and test data, the shuffled-label test gives chance-level performance, and all pre-processing is fitted inside the training folds. The high scores are therefore a property of the data, not of a leak in the pipeline."),
  p("The most likely property is that some attributes encode the outcome almost directly. In the development set, slope = 0 always means no disease, slope = 3 always means disease, and cholesterol = 0 always means disease. Slope 0 is not defined in the dataset's documentation. Yet removing slope costs only 0.004 in ROC-AUC, so the remaining attributes still separate the classes almost perfectly. The dataset may have been curated or partly synthesised in a way that makes the classes unusually separable. The counter-intuitive oldpeak direction (higher ST depression slightly lowering predicted risk) points the same way, since in clinical data larger ST depression is associated with more disease."),
  p("The practical consequence is that E1's numbers describe how well the models fit this dataset, not how well they would perform on new Indian patients. That is the question E3 addresses by testing transfer to four independent UCI cohorts (Cleveland, Hungary, Switzerland and VA Long Beach)."),

  h2("4.3 Limitations"),
  bullet("Single source: all 1,000 patients come from one hospital dataset, with no information on collection dates, inclusion criteria or how the diagnosis was made."),
  bullet("Undocumented codes and artefacts: slope = 0 and cholesterol = 0 align perfectly with the outcome; the results should be read with this in mind."),
  bullet("Small test set: with 200 patients and 1–3 errors per model, differences between models cannot be resolved; the bootstrap intervals reflect this."),
  bullet("Calibration slope is unstable when nearly all predictions are near 0 or 1; Brier scores and calibration curves are reported alongside it."),
  bullet("TabPFN, listed in the plan, was not evaluated because its model weights require a separate licence."),
  bullet("Research prototype only: none of these models has been validated for clinical use."),

  h2("4.4 Conclusion"),
  p("E1 shows that interpretable models match complex ensembles on the Indian dataset: logistic regression and the EBM reach the same near-perfect discrimination as stacked gradient boosting, with explanations that agree across methods. The sanity checks exclude pipeline leakage but reveal dataset artefacts, especially in the slope attribute, that make the classes unusually separable. External validation (E3) is needed before any claim about real-world performance."),

  h2("Reproducibility"),
  p([new TextRun("All numbers in this document come from "), new TextRun({ text: "python -m src.run_all --stages audit e1", font: FONT, size: 24 }),
    new TextRun(" (seed 42), which writes results/table4_models.csv, test_metrics.csv, cv_metrics.csv, model_comparison.csv, feature_importance_agreement.csv, sanity.md and the figures in results/figures/. This document is regenerated by "),
    new TextRun({ text: "node reports/make_e1_report.js", font: "Consolas", size: 18 }), new TextRun(".")]),
];

const doc = new Document({
  creator: "Manjunath Prasad",
  title: "Experiment E1: Results and Discussion",
  styles: {
    default: { document: { run: { font: FONT, size: 24, color: "000000" } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 24, bold: true, color: "000000" }, paragraph: { spacing: { before: 0, after: 120 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 24, bold: true, color: "000000" }, paragraph: { spacing: { before: 300, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 24, bold: true, color: "000000" }, paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "E1 results · CVD prediction in Indian patients", size: 24, color: "000000" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 24, color: "000000" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, "reports", "E1_results.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
