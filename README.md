# Walk-LCP: Least-cost walking paths over a DEM with Tobler's hiking function

[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[!\[Python ≥ 3.10](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/)

This repository is the reproducibility package for the analysis presented in:

the article "LiDAR-Integrated GIS Modelling of Urban Thresholds and Extramural Funerary Landscapes at Parion" by Malgil and Tugrul which is under review.



It computes least-cost walking paths (LCPs) and travel times across a digital
elevation model (DEM) using Dijkstra's algorithm with edge costs derived from
Tobler's hiking function (Tobler 1993). Given any number of start and stop
points, the package returns the time-optimal walking route for every
(start, stop) pair as a polyline layer with distance and time attributes.

## 

## Contents

```
walk\\\_lcp\\\_repro/
├── README.md              ← this file
├── LICENSE                ← MIT license
├── CITATION.cff           ← machine-readable citation
├── requirements.txt       ← pip dependencies (pinned)
├── environment.yml        ← conda environment specification
├── walk\\\_lcp.py            ← STANDALONE script (recommended for review)
├── walk\\\_lcp\\\_qgis.py       ← QGIS-runnable script (convenience)
├── viewshed\\\_null.py       ← viewshed overlap vs. matched null model
├── viewshed\\\_foci\\\_parion.json ← foci and pairs config for the manuscript
├── demo\\\_test\\\_run.ipynb    ← Jupyter notebook: synthetic ridge-and-gap DEM
├── demo\\\_test\\\_run\\\_v2.ipynb  ← Jupyter notebook: alternative complex multi-ridge DEM
├── test\\\_walk\\\_lcp.py       ← unit tests (least-cost paths)
├── test\\\_viewshed\\\_null.py  ← unit tests (viewshed null model)
└── .gitignore
```

This package covers **both** quantitative analyses in the manuscript:

|Analysis|Script|Manuscript output|
|-|-|-|
|Least-cost walking paths|`walk_lcp.py`|Figure 6|
|Pairwise viewshed overlap vs. matched null model|`viewshed_null.py`|Table 1, Figure 10|

## 

## Quick start (standalone, recommended)

### 

### 1\. Create the environment

Using conda (preferred — GDAL is easier on conda):

```bash
conda env create -f environment.yml
conda activate walk-lcp
```

Or using pip:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\\\\Scripts\\\\activate
pip install -r requirements.txt
```

### 

### 2\. Run the analysis

```bash
python walk_lcp.py \
    --dem    "02_processed/dem 04 Low Pass Filter.tif" \
    --start  03_gis_inputs/nodes_start.gpkg \
    --stop   03_gis_inputs/nodes_stop.gpkg \
    --output ./my_lcp_results.gpkg \
    --manifest ./manifest.json
```

Optional flags:

* `--export-cumulative` — also writes one cumulative travel-time GeoTIFF per
start point, next to the output file.
* `--log-level DEBUG` — verbose logging.
* `--version` — print version and exit.

### 

### 3\. Verify the run

The `--manifest` flag produces a JSON file containing:

* Run timestamp (UTC)
* Python and library versions
* Platform information
* Algorithm parameters
* SHA-256 hashes of every input and output file

A reviewer can re-run the command, point `--manifest` at a new path, and
compare the two manifests to confirm the inputs are byte-identical and the
software environment matches.

## 

## Demo notebook (no real data required)

`demo\\\_test\\\_run.ipynb` is a self-contained Jupyter notebook that exercises
the full pipeline on a small synthetic DEM (no GIS data needed). It is the
fastest way for a reviewer to confirm the algorithm works on their machine
before pointing it at the manuscript's real inputs.

```bash
pip install matplotlib jupyter   # in addition to requirements.txt
jupyter notebook demo\\\_test\\\_run.ipynb
```

The notebook builds a 100 × 100 m DEM with a diagonal ridge and a gap,
runs Dijkstra with Tobler's hiking function, plots the cumulative
travel-time field, reconstructs and overlays the optimal walking paths,
and compares them against a naive "5 km/h on flat ground" estimate. It
runs in a few seconds and requires only `numpy` and `matplotlib` —
none of `gdal`, `geopandas`, `shapely`, or QGIS.

The shipped copy of the notebook is **pre-executed**, so reviewers can
scan through the figures and printed outputs directly on GitHub or Zenodo
without running anything.

A second notebook, `demo_test_run_v2.ipynb`, demonstrates the same
pipeline on a more complex synthetic topography — multiple Gaussian hills
and depressions with smoothed random noise — to show that the algorithm
generalises beyond the simple ridge-and-gap case. It can be opened and
run the same way:

```bash
jupyter notebook demo_test_run_v2.ipynb
```

## 

## Viewshed overlap against a matched null model

`viewshed_null.py` produces the two viewshed inference results reported in
the manuscript:

* the **union test** — the combined visible area of a pair of foci, in
  hectares (the manuscript's Table 1);
* the **overlap test** — shared visible ground as a percentage of the
  smaller of the two viewsheds (the manuscript's Figure 10).

Each observed statistic is compared against a null distribution of 999
random pairs drawn from the same DEM, matched to the observed pair on
elevation (±5 m) and on separation (±10 %). Significance is
`p = (r + 1) / (n + 1)` in the upper tail, Holm-corrected across the seven
tests. All test directions are fixed a priori.

### Running it

```bash
python viewshed_null.py \
    --dem          "02_processed/dem 04 Low Pass Filter.tif" \
    --config       viewshed_foci_parion.json \
    --observed-dir 04_results/Viewshed/ \
    --output       ./null_model_results.csv \
    --output-json  ./null_model_draws.json
```

Fill in the real focus coordinates in `viewshed_foci_parion.json` first —
the script refuses to run while any `"FILL"` placeholder remains.

### Viewshed backends

|Backend|Flag|Use|
|-|-|-|
|GRASS `r.viewshed`|`--backend grass` (default)|Reproduces the published results. Needs `grass` on PATH.|
|Internal NumPy R2|`--backend internal`|No external software. Fast, approximate, for testing.|

The two must not be mixed within a run: if the observed viewsheds came
from GRASS, the null draws must too, or the comparison is not
like-for-like. The CLI refuses the invalid combination and tells you to
add `--recompute-observed`.

> **Runtime warning.** The null model computes 2 × 999 viewsheds per pair.
> With the GRASS backend and seven pairs that is a long run — hours to
> days depending on the machine, because each call spawns a GRASS
> session. Use `--backend internal` with a small `--n-null` first to check
> the wiring, then run the real thing once.

### Key options

* `--n-null` — number of matched random pairs (default 999).
* `--elev-tol` / `--dist-tol` — matching tolerances (default 5 m / 10 %).
* `--aggregate` — block factor to the analysis grid (default 5, i.e.
  0.4 m → 2 m).
* `--aggregate-rule` — `majority` (published), `any` or `all` for
  sensitivity checks.
* `--seed` — fixes the null draws, so a reviewer gets byte-identical
  numbers.

## 

## Inputs

|Input|Format|Requirements|
|-|-|-|
|DEM|GeoTIFF|Single band, projected CRS in metres (e.g. EPSG:5253 TUREF/TM27). Pixel size known; NoData defined.|
|Start points|Shapefile or GeoPackage|Point geometry. Same CRS as DEM.|
|Stop points|Shapefile or GeoPackage|Point geometry. Same CRS as DEM.|

All three layers must share the same CRS. The script does not reproject.

## 

## Outputs

|Output|Format|Contents|
|-|-|-|
|Routes|GeoPackage (recommended) or Shapefile|LineString features, one per (start, stop) pair, with attributes `route\\\_id`, `start\\\_id`, `stop\\\_id`, `dist\\\_m`, `time\\\_s`, `time\\\_min`, `time\\\_hr`.|
|Cumulative time rasters (optional)|GeoTIFF|One per start point. Each pixel = travel time in seconds from that start. NoData = -9999.|
|Manifest (optional)|JSON|Run metadata for reproducibility.|

## 

## Algorithm

The cost surface is the DEM itself; edge costs are constructed on-the-fly
from the elevation difference and pixel size:

1. The DEM is rasterised into a grid graph with 8-connectivity (queen
neighbourhood). Each cell connects to its eight neighbours.
2. The **slope** along each edge is `(z\\\_target − z\\\_source) / horizontal\\\_distance`,
where horizontal\_distance is computed from the DEM's pixel size.
3. The **walking speed** along that edge is given by Tobler's hiking function:

```
   v(S) = 6 · exp(−3.5 · |S + 0.05|)        \\\[km/h]
   ```

Peak speed (\~6 km/h) occurs at S = −0.05 (5 % downhill). Speed is
clamped to ≥ 0.01 m/s to avoid infinities.

4. The **edge cost** is `horizontal\\\_distance / v`, in seconds.
5. **Dijkstra's algorithm** is run from every start point. For each stop
point, the time-optimal path is reconstructed using the predecessor
array.

   The implementation uses Python's `heapq` for the priority queue and NumPy
for the elevation grid. Runtime is roughly `O(N log N)` where `N` is the
number of DEM cells.

## 

   ## Testing

   Run the unit tests:

   ```bash
python -m pytest test\\\_walk\\\_lcp.py test\\\_viewshed\\\_null.py -v
```

   Neither test file needs input data, GDAL or GRASS — only `numpy` and
`pytest`. They run in under a second.

   `test_walk_lcp.py` covers:

* Tobler's hiking function at canonical slope values (peak, flat, steep).
* Dijkstra on a synthetic flat 5 × 5 DEM (verifies straight-line paths).
* Dijkstra on a synthetic ridge DEM (verifies the path goes around).
* Path reconstruction round-trips.
* NoData handling.

   `test_viewshed_null.py` covers:

* Block aggregation to the 2 m analysis grid (majority / any / all).
* The two test statistics, including the "percentage of the *smaller*
viewshed" convention and the empty-viewshed edge case.
* `p = (r + 1)/(n + 1)`, including the 0.001 floor at 999 draws and the
treatment of ties.
* Holm correction (the seven-test case that turns p = 0.001 into 0.007).
* Matched sampling: elevation and distance constraints are actually
honoured, draws are reproducible from a seed, and impossible constraints
raise rather than silently returning a short null distribution.
* The internal viewshed: flat-plain visibility, wall shadows, the
distance cut-off, and observer height.
* The whole driver end-to-end on a synthetic hilly DEM.

## 

  ## QGIS alternative

  If you prefer to run inside QGIS rather than from a terminal, open
`walk\\\_lcp\\\_qgis.py` in the QGIS Python Console / Script Editor, set the
four paths at the top of the file, and run it. The mathematics are
identical to the standalone version.

  QGIS version tested: ≥ 3.28 LTR. Requires the `gdal`, `numpy` and `qgis`
modules that ship with QGIS.

## 

  ## Dependencies

  Standalone version:

|Package|Version|Purpose|
|-|-|-|
|Python|≥ 3.10|Interpreter|
|numpy|≥ 1.24|Array operations|
|gdal (osgeo)|≥ 3.6|Raster I/O|
|geopandas|≥ 0.13|Vector I/O|
|shapely|≥ 2.0|Geometry construction|
|pytest (dev only)|≥ 7|Tests|

Pinned exact versions used for the published results are in
`requirements.txt`.

## 

## Reproducing the published results

1. Download the dataset from https://doi.org/10.5281/zenodo.20384339
(concept DOI — always resolves to the newest version).
2. Place the files in the expected layout (see README in the data archive).
3. Run the `walk_lcp.py` command above with the dataset paths.
4. Compare the SHA-256 hashes in the produced manifest against those in
`published\\\_manifest.json` (included in the data archive).
5. For the viewshed inference, run the `viewshed_null.py` command above
with `--backend grass` and the default `--seed`.

## 

## Data availability

The W25 / Parion LiDAR dataset used to produce the results in the
manuscript is openly archived on Zenodo:

> Malgil, I. (2026). *W25 LiDAR Survey Dataset: DEM, Orthophoto, LAS
> Point Cloud Samples, and GIS Analysis Inputs/Outputs* [Data set].
> Zenodo. https://doi.org/10.5281/zenodo.20384339

License: Creative Commons Attribution 4.0 International (CC BY 4.0).


## 

## Citation

If you use this code, please cite both the manuscript and the software:

**Software**

> Malgil, I. (2026). *Walk-LCP: Least-cost walking paths over a DEM with
> Tobler's hiking function* [Software]. Zenodo.
> https://doi.org/10.5281/zenodo.20381420

That is the concept DOI and resolves to the newest release. To cite the
exact version behind a given result, use that release's own version DOI,
shown on its Zenodo record.

**Manuscript**

> Malgil, I. and Tugrul, S. "LiDAR-Integrated GIS Modelling of Urban
> Thresholds and Extramural Funerary Landscapes at Parion." Manuscript
> under editorial review at *Archaeological Prospection*.

A machine-readable citation is in `CITATION.cff`.

## License

MIT — see `LICENSE`.

## 

## References

* Tobler, W. (1993). *Three Presentations on Geographical Analysis and
Modeling*. NCGIA Technical Report 93-1, University of California,
Santa Barbara.
* Dijkstra, E. W. (1959). A note on two problems in connexion with
graphs. *Numerische Mathematik* 1: 269–271.

