# K-Means Image Segmentation with Automatic K Selection

**Project #7 · CO2, CO3 · BTL-5/6 · Prawahan Katyal & Prateek Singh**

> Implement K-Means clustering for image segmentation and investigate automatic
> selection of the number of clusters using quantitative criteria.

Everything is implemented from scratch with NumPy (K-Means, k-means++, CIE-Lab conversion,
all validity indices, gap statistic, ARI, PSNR). scikit-learn is used **only in the tests**
to cross-validate our implementations.

## Quick start

```bash
pip install -r requirements.txt
python main.py --demo                              # all images in ./samples -> ./results/<name>/
python main.py --image path/to/photo.jpg           # automatic K
python main.py --image photo.jpg --k 5             # fixed K
python main.py --image photo.jpg --kmin 3 --kmax 12 --color-space lab --strategy borda
python main.py --synthetic 5 --noise 15            # synthetic image with known K + ARI
python main.py --image photo.jpg --gallery 2 3 4 6 8   # compare several K side by side
streamlit run app.py                               # interactive GUI
python benchmark.py                                # accuracy of every criterion (~2 min)
pytest                                             # 18 tests
```

Outputs per image (`results/<name>/`): `k_selection.png` (all criteria), `segmentation.png`,
`segmented.png`, `original.png`, `labels.npy`, `selection.csv/json`, `report.json`
(PSNR, silhouette, DB, CH, compression ratio, ARI when ground truth is known), optional `k_gallery.png`.

## Methods

| Criterion | Rule |
|---|---|
| Elbow (WCSS) | knee of the inertia curve (max distance to chord) |
| Silhouette | maximise |
| Davies-Bouldin | minimise |
| Calinski-Harabasz | maximise |
| Gap statistic | Tibshirani 1-SE rule, PCA-aligned uniform reference |

Strategies (`--strategy`): `borda` (default, rank aggregation of silhouette/DB/CH),
`vote` (majority of the five), or a single criterion name.

## Project layout

```
main.py            CLI                       app.py        Streamlit GUI
benchmark.py       K-recovery experiment     requirements.txt
kmseg/             kmeans.py  colorspace.py  metrics.py  selection.py
                   segmentation.py  synthetic.py  visualize.py
tests/             unit + pipeline tests
samples/           8 natural + 3 synthetic images
results/           generated demo outputs + benchmark/
docs/              Project_Report.docx (+ build_report.js)
```

## Python API

```python
from kmseg import AutoKSegmenter, load_image
img = load_image("samples/chelsea.jpg", max_side=256)
res = AutoKSegmenter(k_min=2, k_max=10, color_space="lab").segment(img)
print(res.k, res.metrics["psnr_db"], res.selection.chosen)
```

## Results summary (synthetic images, known K, 105 runs)

| Method | Exact-match acc. | MAE |
|---|---|---|
| elbow | 33.3 % | 1.53 |
| silhouette | 61.9 % | 1.21 |
| davies_bouldin | 58.1 % | 1.11 |
| calinski_harabasz | 65.7 % | 0.81 |
| gap | 49.5 % | 0.97 |
| vote | 66.7 % | 0.94 |
| **borda** | **71.4 %** | 0.84 |

Accuracy of the best methods exceeds 94 % at low noise and drops as noise grows.

## Notes / tips

* On natural photos there is no true K; silhouette/DB lean to K = 2. Use `--kmin 3`, `--strategy elbow`
  or compare with `--gallery`. PSNR and compression ratio quantify the fidelity/size trade-off.
* `--spatial-weight 0.3` adds (x, y) features for more compact segments.
* Large images are down-scaled (`--max-side`, default 256; `0` = keep) and fitted on a pixel sample for speed.
