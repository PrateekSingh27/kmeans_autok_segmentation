const fs = require("fs");
const D = require("docx");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, HeadingLevel,
  AlignmentType, BorderStyle, WidthType, ShadingType, LevelFormat, PageBreak } = D;
const NAVY = "1F3864", LIGHT = "E8EDF7", ALT = "F3F6FB";
const R = (p) => `${__dirname}/../results/${p}`;
const W = 9026; // A4 content width (DXA, 1" margins)

const p = (t, o = {}) => new Paragraph({ spacing: { after: 120 }, ...o, children: Array.isArray(t) ? t : [new TextRun(t)] });
const b = (t) => new TextRun({ text: t, bold: true });
const code = (t) => new Paragraph({ spacing: { after: 60 }, shading: { type: ShadingType.CLEAR, fill: LIGHT },
  children: [new TextRun({ text: t, font: "Consolas", size: 19 })] });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)],
  border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: NAVY, space: 2 } } });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bl = (t) => new Paragraph({ numbering: { reference: "bl", level: 0 }, spacing: { after: 60 },
  children: Array.isArray(t) ? t : [new TextRun(t)] });
const img = (path, w, h, cap) => [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 40 },
  children: [new ImageRun({ type: path.endsWith("jpg") ? "jpg" : "png", data: fs.readFileSync(path),
    transformation: { width: w, height: h }, altText: { title: cap, description: cap, name: cap } })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: cap, italics: true, size: 18 })] })];

function table(head, rows, widths) {
  const bd = { style: BorderStyle.SINGLE, size: 4, color: "B0B8C8" };
  const borders = { top: bd, bottom: bd, left: bd, right: bd };
  const cell = (t, i, fill, bold, color) => new TableCell({ borders, width: { size: widths[i], type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill }, margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: String(t), bold, color, size: 19 })] })] });
  return new Table({ width: { size: widths.reduce((a, c) => a + c), type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: head.map((t, i) => cell(t, i, NAVY, true, "FFFFFF")) }),
      ...rows.map((r, j) => new TableRow({ children: r.map((t, i) => cell(t, i, j % 2 ? ALT : "FFFFFF", false)) }))] });
}

// ---- data ------------------------------------------------------------
const names = ["astronaut", "chelsea", "coffee", "colorwheel", "hubble_deep_field", "immunohistochemistry",
  "retina", "rocket", "synthetic_k3", "synthetic_k5", "synthetic_k7"];
const demo = names.map((n) => { const r = JSON.parse(fs.readFileSync(R(`${n}/report.json`)));
  const c = r.selection.chosen; return [n, c.elbow, c.silhouette, c.davies_bouldin, c.calinski_harabasz, c.gap,
    r.k, r.metrics.psnr_db.toFixed(1), r.metrics.silhouette.toFixed(3), r.metrics.compression_ratio + "x"]; });
const bench = fs.readFileSync(R("benchmark/benchmark.md"), "utf8").split("\n");
const parseTbl = (start) => { const out = []; for (let i = start; bench[i] && bench[i].startsWith("|"); i++) out.push(bench[i].split("|").slice(1, -1).map((s) => s.trim())); return out.slice(2); };
const t1 = parseTbl(0);
const t2start = bench.findIndex((l) => l.startsWith("| Method | noise"));
const t2 = parseTbl(t2start), t2head = bench[t2start].split("|").slice(1, -1).map((s) => s.trim());
const ari = bench.find((l) => l.startsWith("Mean ARI")).split(": ")[1];

const C = [];
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1800, after: 200 },
  children: [new TextRun({ text: "K-Means Image Segmentation", size: 52, bold: true, color: NAVY })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
  children: [new TextRun({ text: "with Automatic K Selection", size: 40, bold: true, color: NAVY })] }));
C.push(p("Project #7  |  Course Outcomes: CO2, CO3  |  Bloom level: BTL-5/6", { alignment: AlignmentType.CENTER }));
C.push(p([b("Team: "), new TextRun("Prawahan Katyal, Prateek Singh")], { alignment: AlignmentType.CENTER }));
C.push(p("Project Report", { alignment: AlignmentType.CENTER }));
C.push(new Paragraph({ children: [new PageBreak()] }));

C.push(h1("1. Problem Statement and Objectives"));
C.push(p("Implement K-Means clustering for image segmentation and investigate the automatic selection of the number of clusters using quantitative criteria."));
C.push(bl("Implement K-Means (Lloyd's algorithm, k-means++ seeding) from scratch with NumPy."));
C.push(bl("Segment colour images by clustering pixels in RGB or perceptual CIE-Lab space, optionally with spatial coordinates."));
C.push(bl("Implement five quantitative criteria for choosing K: elbow (WCSS), silhouette, Davies-Bouldin, Calinski-Harabasz and the gap statistic."));
C.push(bl("Combine criteria through majority voting and Borda rank aggregation."));
C.push(bl("Evaluate on synthetic images with known K (accuracy of K recovery, Adjusted Rand Index) and on natural images (PSNR, silhouette, compression ratio)."));

C.push(h1("2. Theory"));
C.push(h2("2.1 K-Means for segmentation"));
C.push(p("Each pixel is a feature vector x (colour, optionally position). K-Means finds K centroids that minimise the within-cluster sum of squares (WCSS):"));
C.push(code("J = sum_{j=1..K} sum_{x in C_j} || x - mu_j ||^2"));
C.push(p("Lloyd's algorithm alternates an assignment step (nearest centroid) and an update step (mean of members) until the centroid shift falls below a tolerance. k-means++ seeding picks each new centre with probability proportional to the squared distance to the nearest existing centre, which gives an O(log K)-competitive solution and faster convergence. The best of n_init restarts (lowest J) is kept. Every pixel is finally replaced by the mean colour of its cluster."));
C.push(h2("2.2 Criteria for choosing K"));
C.push(table(["Criterion", "Definition", "Rule"], [
  ["Elbow (WCSS)", "J(K) decreases monotonically with K", "Knee = point farthest below the chord joining the end points of the normalised curve"],
  ["Silhouette", "s(i) = (b(i) - a(i)) / max(a(i), b(i)); a = mean intra-cluster distance, b = mean distance to nearest other cluster", "Maximise mean s"],
  ["Davies-Bouldin", "DB = (1/K) sum_i max_{j!=i} (S_i + S_j) / M_ij", "Minimise"],
  ["Calinski-Harabasz", "CH = [B/(K-1)] / [W/(N-K)]  (between / within dispersion)", "Maximise"],
  ["Gap statistic", "Gap(K) = E*[log W_K] - log W_K, reference = uniform data in PCA-aligned box (B = 10)", "Smallest K with Gap(K) >= Gap(K+1) - s(K+1)"],
], [1800, 4226, 3000]));
C.push(p(""));
C.push(h2("2.3 Combining criteria"));
C.push(bl([b("Vote: "), new TextRun("majority K among the five criteria; ties broken by the silhouette score.")]));
C.push(bl([b("Borda: "), new TextRun("every K is ranked by silhouette, DB and CH; the K with the smallest rank sum wins. This is the default strategy because it was the most accurate in our benchmark.")]));

C.push(h1("3. Implementation"));
C.push(table(["Module", "Responsibility"], [
  ["kmseg/kmeans.py", "K-Means: vectorised chunked distances, k-means++ / random init, restarts, empty-cluster repair"],
  ["kmseg/colorspace.py", "sRGB <-> CIE-Lab (D65) conversion"],
  ["kmseg/metrics.py", "Silhouette, DB, CH, gap, elbow, ARI, MSE, PSNR (all from scratch)"],
  ["kmseg/selection.py", "Evaluate K range, per-criterion choice, vote / borda strategies"],
  ["kmseg/segmentation.py", "AutoKSegmenter: features, sampling, final fit, labelling, quality metrics"],
  ["kmseg/synthetic.py", "Ground-truth test images (warped Voronoi regions + Gaussian noise)"],
  ["kmseg/visualize.py", "Selection plots, segmentation panels, K galleries"],
  ["main.py / app.py / benchmark.py", "CLI, Streamlit GUI, benchmark experiment"],
  ["tests/", "18 pytest tests, including cross-checks against scikit-learn"],
], [3000, 6026]));
C.push(p(""));
C.push(h2("Design decisions"));
C.push(bl("Fitting and K selection run on a random sample of at most 20,000 pixels; the full image is then labelled with the learned centroids, so large images stay fast."));
C.push(bl("The silhouette is O(n^2), so it is computed on a 2,000-point subsample; the gap statistic uses 3,000 points."));
C.push(bl("Clusters are re-labelled by decreasing size so palettes are reproducible."));
C.push(bl("CIE-Lab is the default colour space because Euclidean distance there approximates perceived colour difference."));
C.push(bl("Correctness: our K-Means inertia and silhouette / DB / CH / ARI values match scikit-learn in unit tests."));

C.push(h1("4. Experiments and Results"));
C.push(h2("4.1 Worked example: synthetic image with 7 regions"));
C.push(...img(R("synthetic_k7/k_selection.png"), 600, 320, "Figure 1. All criteria on a synthetic image with true K = 7."));
C.push(p("Silhouette, Davies-Bouldin, Calinski-Harabasz and the gap statistic all recover K = 7, whereas the elbow rule proposes 5 because the WCSS curve has no sharp knee when many regions are present."));
C.push(...img(R("synthetic_k7/segmentation.png"), 600, 200, "Figure 2. Segmentation result for the synthetic image."));

C.push(h2("4.2 Benchmark on synthetic images with known K"));
C.push(p(`${t1.length > 0 ? "105" : ""} images were generated (K_true = 2..8, noise std = 8, 20, 35, five seeds each). Table below gives the exact-match accuracy of each criterion.`));
C.push(table(["Method", "Exact-match accuracy (%)", "Mean abs. error"], t1, [3000, 3500, 2526]));
C.push(p(""));
C.push(table(t2head, t2, [2400, 2200, 2200, 2226]));
C.push(p(""));
C.push(p(`Mean Adjusted Rand Index of the final segmentation (voted K) against ground truth: ${ari}.`));
C.push(...img(R("benchmark/benchmark.png"), 600, 214, "Figure 3. Accuracy of each criterion and accuracy versus number of true segments."));
C.push(p("Note: the table was produced with the voting strategy as the baseline run; the Borda strategy is evaluated in the same runs and is listed in the table."));

C.push(h2("4.3 Natural images"));
C.push(table(["Image", "Elbow", "Sil.", "DB", "CH", "Gap", "Final K", "PSNR (dB)", "Silhouette", "Compr."], demo,
  [1900, 600, 600, 600, 600, 600, 900, 1050, 1100, 1076]));
C.push(p(""));
C.push(...img(R("coffee/k_gallery.png"), 600, 200, "Figure 4. Segmentation of the coffee image for K = 2, 3, 4, 5, 6, 8 (star = automatically chosen K)."));
C.push(...img(R("chelsea/segmentation.png"), 600, 200, "Figure 5. Chelsea (cat) segmentation with automatically selected K."));

C.push(h1("5. Discussion"));
C.push(bl([b("Clean, well-separated colours: "), new TextRun("silhouette, DB, CH and gap agree and recover the true K almost always (>85% at low noise).")]));
C.push(bl([b("Noise: "), new TextRun("accuracy falls for every criterion as noise grows because clusters overlap in colour space; Calinski-Harabasz and the Borda combination degrade most gracefully.")]));
C.push(bl([b("Elbow: "), new TextRun("the least reliable as an automatic rule because WCSS curves are smooth; it is best used as a visual aid.")]));
C.push(bl([b("Natural photographs: "), new TextRun("colour distributions are continuous, so there is no single 'true' K. Silhouette and DB are biased toward small K (often 2), while CH and gap favour larger K. Aggregation (Borda) gives a compromise, and PSNR / compression ratio show the trade-off between fidelity and number of segments.")]));
C.push(bl([b("Colour space and spatial weight: "), new TextRun("Lab usually yields perceptually cleaner segments; a small spatial weight (0.2-0.5) encourages compact regions but may merge distant same-colour objects.")]));
C.push(bl([b("Limitations: "), new TextRun("K-Means assumes roughly spherical clusters of similar size, is sensitive to initialisation (mitigated by k-means++ and restarts), and is purely colour-based unless spatial features are added.")]));

C.push(h1("6. Conclusion"));
C.push(p("A complete K-Means segmentation system with five quantitative K-selection criteria and two aggregation strategies was implemented, validated against scikit-learn, and evaluated quantitatively on synthetic and natural images. No single criterion is universally best; rank aggregation over several indices was the most reliable automatic strategy in our experiments."));

C.push(h1("7. How to Run"));
["pip install -r requirements.txt", "python main.py --demo", "python main.py --image path/to/photo.jpg --kmax 12",
  "python main.py --synthetic 5 --noise 15", "python benchmark.py", "streamlit run app.py", "pytest"].forEach((l) => C.push(code(l)));

C.push(h1("References"));
["Lloyd, S. (1982). Least squares quantization in PCM. IEEE Trans. Inf. Theory.",
 "Arthur, D. & Vassilvitskii, S. (2007). k-means++: the advantages of careful seeding. SODA.",
 "Rousseeuw, P. (1987). Silhouettes: a graphical aid to the interpretation of cluster analysis. J. Comput. Appl. Math.",
 "Davies, D. & Bouldin, D. (1979). A cluster separation measure. IEEE TPAMI.",
 "Calinski, T. & Harabasz, J. (1974). A dendrite method for cluster analysis. Comm. Stat.",
 "Tibshirani, R., Walther, G. & Hastie, T. (2001). Estimating the number of clusters via the gap statistic. JRSS-B.",
 "Hubert, L. & Arabie, P. (1985). Comparing partitions. J. Classification."].forEach((t) => C.push(bl(t)));

const doc = new Document({
  styles: { default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Arial", size: 32, bold: true, color: NAVY }, paragraph: { spacing: { before: 320, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Arial", size: 26, bold: true, color: NAVY }, paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 1 } }] },
  numbering: { config: [{ reference: "bl", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } }, children: C }],
});
Packer.toBuffer(doc).then((buf) => fs.writeFileSync(`${__dirname}/Project_Report.docx`, buf));
