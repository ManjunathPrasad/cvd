// Build the accuracy table .docx from rows extracted from results/youngsters/*.csv
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, ShadingType, BorderStyle, VerticalAlign,
} = require("docx");

const [rowsPath, outPath] = process.argv.slice(2);
const rows = JSON.parse(fs.readFileSync(rowsPath, "utf8"));

const FONT = "Calibri";
const widths = [3000, 1400, 1400, 1400, 1826]; // sums to 9026 = A4 width minus 1" margins
const header = ["Model", "Test accuracy", "Sensitivity", "Specificity", "CV accuracy (mean ± SD)"];
const line = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: line, bottom: line, left: line, right: line };

function cell(text, i, { bold = false, italic = false, fill = null } = {}) {
  return new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    shading: fill ? { type: ShadingType.CLEAR, color: "auto", fill } : undefined,
    children: [new Paragraph({
      alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [new TextRun({ text, bold, italic, font: FONT, size: 20 })],
    })],
  });
}

const table = new Table({
  width: { size: 9026, type: WidthType.DXA },
  columnWidths: widths,
  rows: [
    new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, { bold: true, fill: "D9E2F3" })) }),
    ...rows.map((r, k) => new TableRow({
      children: r.map((v, i) => cell(v, i, k === 0 ? { italic: true, fill: "F2F2F2" } : {})),
    })),
  ],
});

const p = (text, opts = {}) => new Paragraph({
  spacing: { after: opts.after ?? 120 },
  alignment: opts.align,
  children: [new TextRun({ text, font: FONT, size: opts.size ?? 20, bold: opts.bold, italic: opts.italic })],
});

const doc = new Document({
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children: [
      p("Prediction accuracy — heart_attack_youngsters_india.csv", { size: 28, bold: true, after: 200 }),
      p("Table 1. Accuracy, sensitivity and specificity on the held-out test set (n = 2,000) and accuracy from 5-fold nested cross-validation on the development set (n = 8,000).", { bold: true, after: 120 }),
      table,
      p("", { after: 60 }),
      p("Notes", { bold: true, after: 60 }),
      p("Sensitivity and specificity are on the test set, at each model’s Youden-optimal threshold chosen on development data only. Ensembles combine the three best single models in cross-validation (CatBoost, random forest, XGBoost).", { size: 18 }),
      p("Only 20.4% of patients are positive, so predicting “No” for everyone gives 79.6% accuracy with no predictive information. No model exceeds this baseline; test ROC-AUC is 0.49–0.52 for all models, and every 95% confidence interval includes 0.50. Accuracy should therefore not be reported on its own.", { size: 18 }),
      p("Source: results/youngsters/test_metrics.csv and cv_metrics.csv, written by src/run_youngsters.py. Research prototype, not for clinical use.", { size: 18, italic: true }),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(outPath, buf); console.log("wrote", outPath); });
