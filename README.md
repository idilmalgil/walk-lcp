# Walk-LCP: Least-cost walking paths over a DEM with Tobler's hiking function

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python ≥ 3.10](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/)

This repository is the reproducibility package for the analysis presented in:

> [FILL: full citation of your manuscript, e.g. Surname, F., Surname, F. (2026).
> Title. *Journal*, vol(issue), pages. DOI:xxxxx]

It computes least-cost walking paths (LCPs) and travel times across a digital
elevation model (DEM) using Dijkstra's algorithm with edge costs derived from
Tobler's hiking function (Tobler 1993). Given any number of start and stop
points, the package returns the time-optimal walking route for every
(start, stop) pair as a polyline layer with distance and time attributes.

## Contents

```
walk_lcp_repro/
├── README.md              ← this file
├── LICENSE                ← MIT license
├── CITATION.cff           ← machine-readable citation
├── requirements.txt       ← pip dependencies (pinned)
├── environment.yml        ← conda environment specification
├── walk_lcp.py            ← STANDALONE script (recommended for review)
├── walk_lcp_qgis.py       ← QGIS-runnable script (convenience)
├── demo_test_run.ipynb    ← Jupyter notebook: full pipeline on synthetic data
├── test_walk_lcp.py       ← unit tests
└── .gitignore
```

## Quick start (standalone, recommended)

### 1. Create the environment

Using conda (preferred — GDAL is easier on conda):

```bash
conda env create -f environment.yml
conda activate walk-lcp
```

Or using pip:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the analysis

```bash
python walk_lcp.py \
    --dem    path/to/dem.tif \
    --start  path/to/start_points.gpkg \
    --stop   path/to/stop_points.gpkg \
    --output path/to/lcp_all_routes.gpkg \
    --manifest path/to/manifest.json
```

Optional flags:

- `--export-cumulative` — also writes one cumulative travel-time GeoTIFF per
  start point, next to the output file.
- `--log-level DEBUG` — verbose logging.
- `--version` — print version and exit.

### 3. Verify the run

The `--manifest` flag produces a JSON file containing:

- Run timestamp (UTC)
- Python and library versions
- Platform information
- Algorithm parameters
- SHA-256 hashes of every input and output file

A reviewer can re-run the command, point `--manifest` at a new path, and
compare the two manifests to confirm the inputs are byte-identical and the
software environment matches.

## Demo notebook (no real data required)

`demo_test_run.ipynb` is a self-contained Jupyter notebook that exercises
the full pipeline on a small synthetic DEM (no GIS data needed). It is the
fastest way for a reviewer to confirm the algorithm works on their machine
before pointing it at the manuscript's real inputs.

```bash
pip install matplotlib jupyter   # in addition to requirements.txt
jupyter notebook demo_test_run.ipynb
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

## Inputs

| Input | Format | Requirements |
|---|---|---|
| DEM | GeoTIFF | Single band, projected CRS in metres (e.g. EPSG:5253 TUREF/TM27). Pixel size known; NoData defined. |
| Start points | Shapefile or GeoPackage | Point geometry. Same CRS as DEM. |
| Stop points  | Shapefile or GeoPackage | Point geometry. Same CRS as DEM. |

All three layers must share the same CRS. The script does not reproject.

## Outputs

| Output | Format | Contents |
|---|---|---|
| Routes | GeoPackage (recommended) or Shapefile | LineString features, one per (start, stop) pair, with attributes `route_id`, `start_id`, `stop_id`, `dist_m`, `time_s`, `time_min`, `time_hr`. |
| Cumulative time rasters (optional) | GeoTIFF | One per start point. Each pixel = travel time in seconds from that start. NoData = -9999. |
| Manifest (optional) | JSON | Run metadata for reproducibility. |

## Algorithm

The cost surface is the DEM itself; edge costs are constructed on-the-fly
from the elevation difference and pixel size:

1. The DEM is rasterised into a grid graph with 8-connectivity (queen
   neighbourhood). Each cell connects to its eight neighbours.
2. The **slope** along each edge is `(z_target − z_source) / horizontal_distance`,
   where horizontal_distance is computed from the DEM's pixel size.
3. The **walking speed** along that edge is given by Tobler's hiking function:

   ```
   v(S) = 6 · exp(−3.5 · |S + 0.05|)        [km/h]
   ```

   Peak speed (~6 km/h) occurs at S = −0.05 (5 % downhill). Speed is
   clamped to ≥ 0.01 m/s to avoid infinities.
4. The **edge cost** is `horizontal_distance / v`, in seconds.
5. **Dijkstra's algorithm** is run from every start point. For each stop
   point, the time-optimal path is reconstructed using the predecessor
   array.

The implementation uses Python's `heapq` for the priority queue and NumPy
for the elevation grid. Runtime is roughly `O(N log N)` where `N` is the
number of DEM cells.

## Testing

Run the unit tests:

```bash
python -m pytest test_walk_lcp.py -v
```

The tests cover:

- Tobler's hiking function at canonical slope values (peak, flat, steep).
- Dijkstra on a synthetic flat 5 × 5 DEM (verifies straight-line paths).
- Dijkstra on a synthetic ridge DEM (verifies the path goes around).
- Path reconstruction round-trips.
- NoData handling.

## QGIS alternative

If you prefer to run inside QGIS rather than from a terminal, open
`walk_lcp_qgis.py` in the QGIS Python Console / Script Editor, set the
four paths at the top of the file, and run it. The mathematics are
identical to the standalone version.

QGIS version tested: ≥ 3.28 LTR. Requires the `gdal`, `numpy` and `qgis`
modules that ship with QGIS.

## Dependencies

Standalone version:

| Package | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 | Interpreter |
| numpy | ≥ 1.24 | Array operations |
| gdal (osgeo) | ≥ 3.6 | Raster I/O |
| geopandas | ≥ 0.13 | Vector I/O |
| shapely | ≥ 2.0 | Geometry construction |
| pytest (dev only) | ≥ 7 | Tests |

Pinned exact versions used for the published results are in
`requirements.txt`.

## Reproducing the published results

1. Download the dataset from [FILL: Zenodo / Figshare / repository DOI].
2. Place the files in the expected layout (see README in the data archive).
3. Run the command above with the dataset paths.
4. Compare the SHA-256 hashes in the produced manifest against those in
   `published_manifest.json` (included in the data archive).

## Data availability

[FILL: Where the input data are archived. Example:
*The DEM derived from the W25 LiDAR survey and the start/stop point
layers used in this study are archived at Zenodo,
DOI: 10.5281/zenodo.xxxxxxx, under a CC-BY-4.0 licence.*]

## Citation

If you use this code, please cite both the manuscript and the software:

```
[FILL: software citation, e.g. via CITATION.cff —
 Surname, F. (2026). Walk-LCP: Least-cost walking paths over a DEM
 with Tobler's hiking function. Version 1.0.0. Zenodo.
 https://doi.org/10.5281/zenodo.xxxxxxx]
```

A machine-readable citation is in `CITATION.cff`.

## License

MIT — see `LICENSE`.

## Acknowledgements

[FILL: funders, data providers, colleagues, e.g.
 *This work was supported by grant XYZ. The W25 LiDAR survey was
 conducted by …. We thank … for comments on an earlier draft.*]

## References

- Tobler, W. (1993). *Three Presentations on Geographical Analysis and
  Modeling*. NCGIA Technical Report 93-1, University of California,
  Santa Barbara.
- Dijkstra, E. W. (1959). A note on two problems in connexion with
  graphs. *Numerische Mathematik* 1: 269–271.
