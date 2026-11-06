# Walk-LCP: Least-cost walking paths and viewshed null models for Parion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python ≥ 3.10](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/)

This repository is the reproducibility package for the analyses in the article
"LiDAR-Integrated GIS Modelling of Urban Thresholds and Extramural Funerary
Landscapes at Parion" by Malgil and Tugrul (under review, *Archaeological Prospection*).

It computes least-cost walking paths (LCPs) and travel times across a digital
elevation model (DEM) with Dijkstra's algorithm and Tobler's hiking function, and
tests pairwise viewshed overlap and combined visible area against a matched null
model of random point pairs drawn from the same terrain.

**Version 2 (revision, October 2026).** The sea and harbour basins are now
excluded from both analyses through a land/sea mask on the DEM grid:
water is impassable in the LCP graph, and it is neither counted in the
viewsheds nor sampled by the null model. The null model reports two-sided
p-values with Holm correction, a symmetric Jaccard index, exceedance counts,
direct line of sight in both directions, and a 0.4 m resolution check.

## Contents

```
walk-lcp-main/
├── README.md                 ← this file
├── LICENSE                   ← MIT license
├── CITATION.cff              ← machine-readable citation
├── requirements.txt          ← pip dependencies
├── environment.yml           ← conda environment specification
├── walk_lcp.py               ← least-cost paths (standalone, recommended)
├── walk_lcp_qgis.py          ← QGIS-runnable LCP script (convenience)
├── viewshed_null.py          ← viewshed overlap / union vs. matched null model
├── viewshed_foci_parion.json ← the 11 foci (coordinates, CRS) and 7 tested pairs
├── water_mask_lcp.py         ← builds the land/sea mask on the DEM grid (QGIS Python)
├── reproduce_parion_results.ipynb ← notebook: checks the reported results against the deposit
├── demo_test_run.ipynb       ← notebook: synthetic ridge-and-gap DEM
├── demo_test_run_v2.ipynb    ← notebook: synthetic multi-ridge DEM
├── analysis_scripts/         ← final tables, line of sight, 2 m surface, route comparison
├── test_walk_lcp.py          ← unit tests (least-cost paths)
└── test_viewshed_null.py     ← unit tests (viewshed null model)
```

| Analysis | Script | Manuscript output |
|-|-|-|
| Least-cost walking paths, water impassable | `walk_lcp.py` | Figure 6 |
| Pairwise viewshed overlap vs. matched null model | `viewshed_null.py` | Table 1, Figure 10, Figure S7 |
| Combined (union) visible area vs. matched null model | `viewshed_null.py` | Table S5, Figures S8 and S9 |
| 0.4 m re-test of three pairs | `viewshed_null.py` | Table S8 |
| Final 0.4 m viewsheds | `04_results/Viewshed_landmasked_0.4m_rerun/make_viewsheds_0.4m_rerun.py` (data deposit; uses `viewshed_null.py`) | Tables S3 and S4, Figures 9 and S6, 0.4 m column of Table 1 |
| Land/sea mask | `water_mask_lcp.py` | Methods; input to both analyses |
| Direct line of sight, all pairs | `analysis_scripts/line_of_sight_all_pairs.py`, `line_of_sight_required_height.py` | Table 1, Section 4.5 |
| Final results tables | `analysis_scripts/assemble_final_tables.py` | `FINAL_*.csv` in `04_results/Null-model/` |
| 2 m cost surface and route comparison | `analysis_scripts/make_2m_cost_surface.py`, `compare_routes.py` | Section 3.2.2, Figure 6 |
| Check of all the above | `reproduce_parion_results.ipynb` | Tables 1, S3, S4, S5, S8; Sections 3.1, 3.2.2, 4.5 |

## Quickest check

Open `reproduce_parion_results.ipynb`. It runs in under a minute or two, recomputes the p-values of Tables 1, S5 and S8 from the deposited draws and the areas of Tables S3 and S4 from the final rasters, repeats the line-of-sight tests, checks the routes and stops with an error if any number differs from the manuscript. It makes 73 checks and needs no GRASS. The sections below regenerate the results from the start.

## Quick start

### 1. Environment

```bash
conda env create -f environment.yml
conda activate walk-lcp
```

or `python -m venv .venv` and `pip install -r requirements.txt` for the unit tests and
the two demo notebooks only.

The viewshed null model needs GRASS GIS. Use the Python and GRASS 8.4.2 that ship with
**QGIS 3.44.9**: they produced the deposited results, and on 6 November 2026 every command
below, run in them, reproduced the deposited routes, tables, draws and rasters exactly.
The conda environment was tested on Windows for the unit tests, all three notebooks and
`walk_lcp.py` (routes identical to the deposit); its GRASS 8.5 did not start there, so run
`viewshed_null.py` from QGIS on Windows. Other operating systems were not tested.

### 2. Least-cost paths (water impassable)

Run from the folder that holds both `W25_data_archive/` and `walk-lcp-main/`:

```bash
python walk-lcp-main/walk_lcp.py \
    --dem    "W25_data_archive/02_processed/dem 04 Low Pass Filter_water_nodata.tif" \
    --start  W25_data_archive/03_gis_inputs/nodes_start.gpkg \
    --stop   W25_data_archive/03_gis_inputs/nodes_stop.gpkg \
    --output ./lcp_routes_watermasked.gpkg \
    --manifest ./manifest.json \
    --name-field location --snap-to-valid 5 --workers 8
```

Options:

* `--snap-to-valid M` — move a node that sits on NoData (water) to the nearest
  walkable cell within M metres; every move is logged and written to the manifest.
  At Parion this affects only the Commercial Harbour node (1.3–1.7 m).
* `--name-field location` — copy node names to `start_name` / `stop_name`.
* `--workers N` — run the per-start Dijkstra searches in parallel.
* `--export-cumulative` — one cumulative travel-time GeoTIFF per start point.

Runtime: about 7 minutes with 8 workers.

Pairs with no walkable connection are listed in `<output>_unreachable.csv`
(none at Parion). Route ids are fixed by node position
(`route_id = start_index × n_stops + stop_index + 1`).

With the original node files, running the same command on `dem 04 Low Pass Filter.tif`
without `--snap-to-valid` reproduces the 238 routes of the first submission exactly
(`04_results/_superseded/LCP_v2_watermasked_before_node_alignment/regression_check.json`).
The nodes at the viewshed foci now use the observer coordinates in
`viewshed_foci_parion.json`.

### 3. Viewshed null model (land-masked pool model, Table 1 / Table S5)

```bash
python walk-lcp-main/viewshed_null.py \
    --dem        "W25_data_archive/02_processed/dem 04 Low Pass Filter.tif" \
    --water-mask W25_data_archive/02_processed/water_mask_0.4m.tif \
    --config     walk-lcp-main/viewshed_foci_parion.json \
    --outdir     ./null_landmask_2m \
    --pool-size 4000 --n-null 999 --seed 20260814 \
    --grass-binary grass --workers 8
```

On Windows with QGIS use `--grass-binary "C:\Program Files\QGIS 3.44.9\bin\grass84.bat"`
and run the script with QGIS's Python (`python-qgis-ltr.bat`). Runtime: about 6 minutes on
16 workers.

0.4 m re-test of the three pairs with a result beyond all random pairs, on the same
mask and the same pool points (about 106 minutes on 16 workers; Table S8):

```bash
python walk-lcp-main/viewshed_null.py ... --aggregate 1 \
    --pool-from ./null_landmask_2m/pool_points.csv --pool-subset-bands \
    --only-pairs "TavsanTum:SouthGate,CircularTower:TavsanTum,CircularTower:TasKule" \
    --outdir ./null_landmask_04m_fragile
```

The scripts in `analysis_scripts/` run from the same folder and write to `analysis_out/`
(or `--outdir`), so the deposited files are not overwritten. The final 0.4 m viewsheds
are made by `04_results/Viewshed_landmasked_0.4m_rerun/make_viewsheds_0.4m_rerun.py`.

Method:

1. Aggregate the 0.4 m DEM to 2 m (block mean, `--aggregate 5`) and the water
   mask by majority. The **analysis region** is valid DEM cells that are land.
2. Draw a pool of random land cells (`--pool-size`, seeded) and compute one
   viewshed per pool point and per focus with GRASS `r.viewshed -b`
   (observer 1.7 m, target 0 m, 3,000 m), in persistent GRASS sessions run in
   parallel (`--workers`). Sight lines may cross water; only land cells count.
3. For each pair (A, B), form all pool pairs within ±5 m of z(A) and z(B) and
   within ±10 % of the separation, and draw 999 of them.
4. Statistics: shared land (ha), shared land as % of the smaller viewshed,
   Jaccard index, union (ha). p-values: upper `(r≥ + 1)/(n + 1)`, lower
   `(r≤ + 1)/(n + 1)`, **two-sided `min(1, 2·min(upper, lower))`**; Holm across
   the seven pairs, per statistic. `n_null_ge_observed` / `beyond_all_draws`
   show results beyond every draw.
5. Direct line of sight between the two foci, both directions, target 1.7 m
   (and 0 m), both by an exact profile test and by `r.viewshed`.
6. Truncation check: how much of the 3,000 m disc the survey covers and whether
   each viewshed reaches the edge of the DEM.

Outputs in `--outdir`: `results_table.csv`, `results.json`, `null_draws.npz`
(every draw: pool pair indices and the four statistics), `pool_points.csv`,
`line_of_sight.csv`, `viewshed_truncation.csv`. Fixed `--seed` gives identical draws.

`--backend internal` uses a NumPy R2 viewshed instead of GRASS (no external
software; approximate; for testing). Observed and null viewsheds always come
from the same backend. `--mode direct` keeps the version-1 per-pair sampler.

### 4. Land/sea mask

`water_mask_lcp.py` (QGIS Python Console or `python-qgis-ltr.bat`) builds
`water_mask_0.4m.tif` and the water-impassable DEM on exactly the LCP grid from
the DEM, the DJI Terra orthophoto (3.38 cm, averaged) and DSM (6.76 cm, third
quartile), all in the data deposit (`01_raw/dom.tif`, `01_raw/dsm.tif`). Set the paths
at the top of the file.

## Testing

```bash
python -m pytest test_walk_lcp.py test_viewshed_null.py -v
```

Only `numpy` and `pytest` are needed (no GDAL, GRASS or data); 66 tests run in
under a second. Last verified on 6 November 2026 in the conda environment of
`environment.yml` (Python 3.12.16, numpy 2.5.4).

`test_walk_lcp.py`: Tobler function, Dijkstra on flat and ridge DEMs, path
reconstruction, NoData handling, snapping of nodes on NoData.

`test_viewshed_null.py`: block aggregation, the statistics (including Jaccard and
the bit-packed versions), one- and two-sided p-values and exceedance counts,
Holm correction, matched sampling (direct and pool), line of sight, the internal
viewshed, both drivers end-to-end on synthetic DEMs (the pool model on a DEM with
a "sea" strip, checking that water is never counted or sampled), and the shipped
configuration file.

## Notebooks

`reproduce_parion_results.ipynb` checks the reported results against the deposit (see above).
`demo_test_run.ipynb` and `demo_test_run_v2.ipynb` are worked examples of the least-cost path
method on synthetic terrain; they use no project data and need only numpy and matplotlib. All
three were executed on 6 November 2026 and are saved with their outputs.

## Algorithm (LCP)

1. 8-connected grid graph; each cell connects to its eight neighbours.
2. Edge slope `S = (z_target − z_source) / horizontal_distance` (anisotropic).
3. Tobler speed `v(S) = 6 · exp(−3.5 · |S + 0.05|)` km/h, floor 0.01 m/s.
4. Edge cost `horizontal_distance / v` in seconds; NoData cells are impassable.
5. Dijkstra from every start point; paths reconstructed from the predecessor array.

## Reproducing the revised results

1. Download the dataset (https://doi.org/10.5281/zenodo.20384339, concept DOI).
2. For a quick check, run `reproduce_parion_results.ipynb`. To regenerate the results, run
   the commands in sections 2 and 3 above.
3. Compare with `04_results/LCP_watermasked/` and `04_results/Null-model/`. The LCP
   manifests record parameters, software versions and SHA-256 hashes of the inputs;
   `results.json` records parameters, seed and software versions.

## Data availability

> Malgil, I. (2026). *W25 LiDAR Survey Dataset: DEM, Orthophoto, LAS Point Cloud
> Samples, and GIS Analysis Inputs/Outputs* [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.20384339 — CC BY 4.0.

What the deposit reproduces: the LCP routes (Figure 6), the land-masked null model
results (Table 1, Tables S5 and S8, Figure 10), the final viewsheds (Tables S3 and
S4) and the water mask, from the archived DEM, orthophoto, DSM, nodes and foci. What
it does not: the full point cloud is not deposited (only six LAS sample tiles from a
May 2026 reprocessing, and 1 m density grids for Figure S11), so the DEM itself can be
inspected but not regenerated from raw data. The observer coordinates were recovered
from the viewshed rasters of the first submission, which used superseded parameters
and are no longer deposited (see `viewshed_foci_parion.json`).

## Citation

> Malgil, I. (2026). *Walk-LCP: Least-cost walking paths over a DEM with
> Tobler's hiking function* [Software]. Zenodo. https://doi.org/10.5281/zenodo.20381420

Concept DOI (newest release). A machine-readable citation is in `CITATION.cff`.

## License

MIT — see `LICENSE`.

## References

* Tobler, W. (1993). *Three Presentations on Geographical Analysis and Modeling*.
  NCGIA Technical Report 93-1, University of California, Santa Barbara.
* Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
  *Numerische Mathematik* 1: 269–271.
* Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics* 6: 65–70.
