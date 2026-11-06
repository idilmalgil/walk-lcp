#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viewshed_null.py — Pairwise viewshed overlap against a matched null model.

Companion to ``walk_lcp.py`` in the reproducibility package accompanying the
manuscript. Where ``walk_lcp.py`` produces the least-cost routes, this module
produces the viewshed inference results reported in the manuscript
(Table 1 and Table S5):

* the **overlap test** — the shared (intersection) visible ground, as a
  percentage of the smaller of the two viewsheds and as a symmetric
  Jaccard index (intersection / union);
* the **union test** — the combined (union) visible area of a pair of foci,
  in hectares (Table S5);
* direct **line of sight** between the two foci of every pair, in both
  directions.

Method (version 2, land-masked pool model — the method behind Table 1/S5)
------------------------------------------------------------------------
1. Aggregate the 0.4 m DEM to the analysis grid (block mean, factor 5 ->
   2 m by default; factor 1 keeps 0.4 m for the resolution check).
2. Build the **analysis region**: valid DEM cells that are *land*, i.e. not
   sea or harbour basin in the supplied water mask (``--water-mask``, a
   0/1 raster on the DEM grid; 1 = water). Viewsheds are computed over the
   full terrain (sight lines may cross water) but only land cells are
   counted, and random points are drawn from land cells only.
3. Draw a **pool** of random land cells (``--pool-size``, seeded) and
   compute one viewshed per pool point and per focus with the same engine
   and parameters (GRASS ``r.viewshed -b``, observer 1.7 m, target 0 m,
   3,000 m).
4. For each tested pair (A, B), form every pool pair (a', b') with
   ``|z(a') - z(A)| <= elev_tol``, ``|z(b') - z(B)| <= elev_tol`` and
   separation within ``dist_tol`` of d(A, B), and draw ``n`` of them.
5. Compare each observed statistic with its matched null distribution:
   upper-tail p = (r_ge + 1)/(n + 1), lower-tail p = (r_le + 1)/(n + 1),
   **two-sided p = min(1, 2 * min(upper, lower))**; Holm correction across
   the pairs, separately for each statistic. Exceedance counts (how many
   draws are >= and <= the observed value) are reported so that a result
   beyond every draw is visible as such.

The older per-pair direct sampler (``run_pair_tests``) is kept for the
unit tests and for small checks with the internal backend.

Viewshed backends
-----------------
``--backend grass`` (default)
    GRASS GIS ``r.viewshed``. Viewsheds are computed in batches inside
    persistent GRASS sessions (``--workers`` in parallel), which is what
    makes a pool of thousands of viewsheds feasible. Requires the GRASS
    launcher (``--grass-binary``, e.g. ``grass`` or ``grass84.bat``).

``--backend internal``
    A self-contained NumPy R2 (radial sweep) viewshed. It requires no
    external software and is what the unit tests and the demo exercise.
    It is an *approximation* of ``r.viewshed`` and will not reproduce the
    published numbers cell-for-cell.

Observed and null viewsheds always come from the same backend in one run,
so every comparison is like-for-like.

Author      : Idil Malgil
Affiliation : HSE University, Moscow
Contact     : idilmalgil@gmail.com
License     : MIT (see LICENSE)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Geospatial libraries (osgeo.gdal) are imported lazily inside the I/O
# functions so that the pure-maths layer of this module (aggregation,
# statistics, matched sampling, the internal viewshed, Holm correction)
# can be imported and tested with just numpy installed — the same
# convention as walk_lcp.py.

__version__ = "2.0.0"
_LOG = logging.getLogger("viewshed_null")

M2_PER_HA = 10_000.0


# =============================================================================
# Raster I/O
# =============================================================================

@dataclass
class Raster:
    """A raster band together with the georeferencing needed to use it."""
    array: np.ndarray
    gt: tuple[float, ...]
    projection: str
    nodata: float | None

    @property
    def pixel_size(self) -> float:
        """Pixel edge length in map units. Square pixels are assumed."""
        return abs(self.gt[1])

    @property
    def cell_area_ha(self) -> float:
        return (abs(self.gt[1]) * abs(self.gt[5])) / M2_PER_HA


def read_raster(path: Path, band: int = 1) -> Raster:
    """Read a single band of a raster into a :class:`Raster`."""
    from osgeo import gdal  # lazy import

    ds = gdal.Open(str(path))
    if ds is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    b = ds.GetRasterBand(band)
    return Raster(
        array=b.ReadAsArray().astype(np.float32),
        gt=tuple(ds.GetGeoTransform()),
        projection=ds.GetProjection(),
        nodata=b.GetNoDataValue(),
    )


def write_raster(path: Path, array: np.ndarray, like: Raster, nodata: float = 0.0) -> None:
    """Write ``array`` as a GeoTIFF using the georeferencing of ``like``."""
    from osgeo import gdal  # lazy import

    driver = gdal.GetDriverByName("GTiff")
    rows, cols = array.shape
    dtype = gdal.GDT_Byte if array.dtype == np.uint8 else gdal.GDT_Float32
    out = driver.Create(
        str(path), cols, rows, 1, dtype,
        options=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"],
    )
    out.SetGeoTransform(like.gt)
    out.SetProjection(like.projection)
    band = out.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(nodata)
    out.FlushCache()


def world_to_pixel(gt: tuple[float, ...], x: float, y: float) -> tuple[int, int]:
    """Convert (x, y) map coordinates to (row, col) pixel indices."""
    origin_x, px_w, _, origin_y, _, px_h = gt
    return int((origin_y - y) / abs(px_h)), int((x - origin_x) / px_w)


def pixel_to_world(gt: tuple[float, ...], row: int, col: int) -> tuple[float, float]:
    """Convert (row, col) pixel indices to the map coordinates of the cell centre."""
    origin_x, px_w, _, origin_y, _, px_h = gt
    return origin_x + (col + 0.5) * px_w, origin_y - (row + 0.5) * abs(px_h)


# =============================================================================
# Aggregation to the analysis grid
# =============================================================================

def _trim_to_multiple(arr: np.ndarray, factor: int) -> np.ndarray:
    """Crop the bottom/right edge so both dimensions divide by ``factor``."""
    rows = (arr.shape[0] // factor) * factor
    cols = (arr.shape[1] // factor) * factor
    return arr[:rows, :cols]


def aggregate_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean aggregation, used for the DEM.

    NaNs propagate as NaN so that nodata does not silently become terrain.
    """
    if factor == 1:
        return arr
    a = _trim_to_multiple(arr, factor)
    rows, cols = a.shape
    return a.reshape(rows // factor, factor, cols // factor, factor).mean(axis=(1, 3))


def aggregate_binary(mask: np.ndarray, factor: int, rule: str = "majority") -> np.ndarray:
    """Block aggregation of a boolean mask.

    Parameters
    ----------
    mask : np.ndarray of bool
    factor : int
        Aggregation factor (e.g. 5 to go from 0.4 m to 2 m).
    rule : {"majority", "any", "all"}
        ``majority`` marks the coarse cell visible when more than half of
        its fine cells are visible. This is the rule used for the
        published results; ``any`` and ``all`` bracket it and are provided
        for sensitivity checks.
    """
    if factor == 1:
        return mask.astype(bool)
    a = _trim_to_multiple(mask.astype(np.float32), factor)
    rows, cols = a.shape
    blocks = a.reshape(rows // factor, factor, cols // factor, factor)
    frac = blocks.mean(axis=(1, 3))
    if rule == "majority":
        return frac > 0.5
    if rule == "any":
        return frac > 0.0
    if rule == "all":
        return frac >= 1.0
    raise ValueError(f"Unknown aggregation rule: {rule!r}")


# =============================================================================
# Pairwise statistics
# =============================================================================

def union_ha(a: np.ndarray, b: np.ndarray, cell_ha: float) -> float:
    """Combined visible area of two viewsheds, in hectares."""
    return float(np.count_nonzero(a | b) * cell_ha)


def intersection_ha(a: np.ndarray, b: np.ndarray, cell_ha: float) -> float:
    """Shared visible area of two viewsheds, in hectares."""
    return float(np.count_nonzero(a & b) * cell_ha)


def pct_of_smaller(a: np.ndarray, b: np.ndarray) -> float:
    """Shared visible ground as a percentage of the smaller of the two viewsheds.

    This is the test statistic reported in the manuscript's overlap figure.
    Returns 0.0 when either viewshed is empty.
    """
    na = int(np.count_nonzero(a))
    nb = int(np.count_nonzero(b))
    smaller = min(na, nb)
    if smaller == 0:
        return 0.0
    return 100.0 * float(np.count_nonzero(a & b)) / smaller


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric overlap: |A & B| / |A | B|. Returns 0.0 when both are empty."""
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b)) / union


STATISTICS = {
    "union_ha": "Combined (union) visible area, hectares",
    "overlap_pct_smaller": "Shared visible ground, % of the smaller viewshed",
}

# Statistics reported by the pool model (version 2).
POOL_STATISTICS = {
    "overlap_ha": "Shared visible land, hectares",
    "overlap_pct_smaller": "Shared visible land, % of the smaller viewshed",
    "jaccard": "Jaccard index of the two viewsheds (intersection / union)",
    "union_ha": "Combined (union) visible land, hectares",
}


# =============================================================================
# Viewshed backends
# =============================================================================

class ViewshedBackend:
    """Interface for anything that can turn an observer cell into a viewshed."""

    name = "abstract"

    def compute(self, row: int, col: int) -> np.ndarray:
        raise NotImplementedError


def _perimeter_cells(nrows: int, ncols: int) -> np.ndarray:
    """(row, col) of every cell on the border of an array, as float."""
    rows = np.arange(nrows)
    cols = np.arange(ncols)
    top = np.stack([np.zeros_like(cols), cols], axis=1)
    bottom = np.stack([np.full_like(cols, nrows - 1), cols], axis=1)
    left = np.stack([rows, np.zeros_like(rows)], axis=1)
    right = np.stack([rows, np.full_like(rows, ncols - 1)], axis=1)
    return np.unique(np.concatenate([top, bottom, left, right]), axis=0).astype(np.float64)


class InternalViewshed(ViewshedBackend):
    """Self-contained R2 (radial sweep) binary viewshed in NumPy.

    Rays are cast from the observer to every cell on the perimeter of the
    analysis window and sampled at one-cell intervals. A sample is visible
    when its vertical angle from the observer equals or exceeds the running
    maximum angle along that ray. This is the classic R2 algorithm: fast,
    deterministic and adequate for testing, but an approximation — cells
    that no ray happens to pass through inherit their neighbour's verdict.

    It is *not* a drop-in substitute for GRASS ``r.viewshed`` and will not
    reproduce the published figures cell-for-cell.
    """

    name = "internal"

    def __init__(
        self,
        dem: np.ndarray,
        pixel_size: float,
        observer_height: float = 1.7,
        target_height: float = 0.0,
        max_distance: float = 3000.0,
        ray_chunk: int = 512,
    ) -> None:
        self.dem = dem
        self.px = pixel_size
        self.observer_height = observer_height
        self.target_height = target_height
        self.max_distance = max_distance
        self.ray_chunk = ray_chunk

    def compute(self, row: int, col: int) -> np.ndarray:
        dem = self.dem
        nrows, ncols = dem.shape
        radius = int(round(self.max_distance / self.px))

        r0, r1 = max(0, row - radius), min(nrows, row + radius + 1)
        c0, c1 = max(0, col - radius), min(ncols, col + radius + 1)
        sub = dem[r0:r1, c0:c1]
        orow, ocol = row - r0, col - c0
        snr, snc = sub.shape

        out = np.zeros((snr, snc), dtype=bool)
        z_obs = float(sub[orow, ocol])
        if not np.isfinite(z_obs):
            raise ValueError(f"Observer cell ({row}, {col}) has no elevation.")
        z_obs += self.observer_height

        perimeter = _perimeter_cells(snr, snc)
        steps = np.arange(1, radius + 1, dtype=np.float32)

        for lo in range(0, len(perimeter), self.ray_chunk):
            block = perimeter[lo:lo + self.ray_chunk]
            dr = block[:, 0] - orow
            dc = block[:, 1] - ocol
            length = np.maximum(np.hypot(dr, dc), 1e-9)
            ur = (dr / length).astype(np.float32)
            uc = (dc / length).astype(np.float32)

            rr = orow + ur[:, None] * steps[None, :]
            cc = ocol + uc[:, None] * steps[None, :]

            inside = (
                (rr >= 0) & (rr <= snr - 1)
                & (cc >= 0) & (cc <= snc - 1)
                & (steps[None, :] <= length[:, None].astype(np.float32))
            )
            ri = np.clip(np.rint(rr), 0, snr - 1).astype(np.int32)
            ci = np.clip(np.rint(cc), 0, snc - 1).astype(np.int32)

            z = sub[ri, ci]
            distance = steps[None, :] * self.px
            angle = (z + self.target_height - z_obs) / distance
            angle = np.where(inside & np.isfinite(angle), angle, -np.inf).astype(np.float32)

            running = np.maximum.accumulate(angle, axis=1)
            previous = np.concatenate(
                [np.full((angle.shape[0], 1), -np.inf, dtype=np.float32), running[:, :-1]],
                axis=1,
            )
            visible = inside & (angle >= previous) & np.isfinite(z)
            out[ri[visible], ci[visible]] = True

        out[orow, ocol] = True

        # Clip to a true circle — the window is square.
        rows_idx = np.arange(snr)[:, None] - orow
        cols_idx = np.arange(snc)[None, :] - ocol
        out &= (np.hypot(rows_idx, cols_idx) * self.px) <= self.max_distance

        full = np.zeros((nrows, ncols), dtype=bool)
        full[r0:r1, c0:c1] = out
        return full


_GRASS_SCRIPT = """
import os, sys
import grass.script as gs

dem = sys.argv[1]
out_tif = sys.argv[2]
east, north = float(sys.argv[3]), float(sys.argv[4])
obs_h, tgt_h, max_d = float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])

gs.run_command("r.in.gdal", input=dem, output="dem", flags="o", overwrite=True)
gs.run_command("g.region", raster="dem")
gs.run_command(
    "r.viewshed", input="dem", output="vs",
    coordinates=(east, north),
    observer_elevation=obs_h, target_elevation=tgt_h,
    max_distance=max_d, memory=2000, flags="b", overwrite=True,
)
gs.run_command("r.out.gdal", input="vs", output=out_tif,
               format="GTiff", type="Byte", createopt="COMPRESS=DEFLATE",
               overwrite=True)
"""


class GrassViewshed(ViewshedBackend):
    """Binary viewshed via GRASS GIS ``r.viewshed`` (``-b`` boolean mode).

    This is the backend that produced the published results. Each call
    spawns ``grass --tmp-location`` on the DEM, imports it, runs
    ``r.viewshed`` and exports the boolean result. That is a heavy way to
    compute one viewshed; for a 999-draw null model prefer running this on
    a machine with GRASS installed and enough time, or use
    ``--backend internal`` for a fast approximate check.
    """

    name = "grass"

    def __init__(
        self,
        dem_path: Path,
        gt: tuple[float, ...],
        shape: tuple[int, int],
        observer_height: float = 1.7,
        target_height: float = 0.0,
        max_distance: float = 3000.0,
        grass_binary: str = "grass",
        grass_python: str = "python",
    ) -> None:
        self.grass_python = grass_python
        self.dem_path = Path(dem_path)
        self.gt = gt
        self.shape = shape
        self.observer_height = observer_height
        self.target_height = target_height
        self.max_distance = max_distance
        self.grass_binary = grass_binary
        if shutil.which(grass_binary) is None:
            raise RuntimeError(
                f"GRASS binary {grass_binary!r} not found on PATH. Install GRASS GIS, "
                f"or pass --backend internal for the approximate NumPy viewshed."
            )

    def compute(self, row: int, col: int) -> np.ndarray:
        east, north = pixel_to_world(self.gt, row, col)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script = tmp_path / "_vs.py"
            script.write_text(_GRASS_SCRIPT, encoding="utf-8")
            out_tif = tmp_path / "vs.tif"
            cmd = [
                self.grass_binary, "--tmp-project", str(self.dem_path),
                "--exec", self.grass_python, str(script),
                str(self.dem_path), str(out_tif),
                f"{east:.6f}", f"{north:.6f}",
                str(self.observer_height), str(self.target_height),
                str(self.max_distance),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  shell=self.grass_binary.lower().endswith(".bat"))
            if proc.returncode != 0 or not out_tif.exists():
                raise RuntimeError(
                    "GRASS r.viewshed failed.\n"
                    f"command: {' '.join(cmd)}\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
                )
            vs = read_raster(out_tif)
        arr = vs.array
        mask = np.isfinite(arr) & (arr > 0)
        if mask.shape != self.shape:
            raise RuntimeError(
                f"GRASS viewshed shape {mask.shape} does not match DEM shape {self.shape}."
            )
        return mask


# =============================================================================
# Matched sampling
# =============================================================================

@dataclass
class Focus:
    """One observation point."""
    name: str
    x: float
    y: float
    row: int = -1
    col: int = -1
    elevation: float = float("nan")


@dataclass
class PairResult:
    """Everything computed for one observed pair."""
    pair: str
    statistic: str
    observed: float
    null_median: float
    null_max: float
    null_p05: float
    null_p95: float
    n_null: int
    p_value: float
    p_holm: float = float("nan")
    null_values: list[float] = field(default_factory=list, repr=False)


def candidate_cells(
    dem: np.ndarray,
    valid: np.ndarray,
    elevation: float,
    tol: float,
) -> np.ndarray:
    """Flat indices of cells within ``tol`` metres of ``elevation``.

    Only cells inside ``valid`` (the analysis extent) are eligible.
    """
    ok = valid & np.isfinite(dem) & (np.abs(dem - elevation) <= tol)
    return np.flatnonzero(ok.ravel())


def sample_matched_pairs(
    dem: np.ndarray,
    valid: np.ndarray,
    pixel_size: float,
    focus_a: Focus,
    focus_b: Focus,
    n_pairs: int,
    elev_tol: float,
    dist_tol: float,
    rng: np.random.Generator,
    max_attempts_per_pair: int = 2000,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Draw ``n_pairs`` random cell pairs matched to the observed pair.

    A drawn pair (a', b') is accepted when a' matches ``focus_a`` in
    elevation, b' matches ``focus_b`` in elevation, and the separation of
    (a', b') is within ``dist_tol`` of the observed separation.

    Raises
    ------
    RuntimeError
        If the constraints are too tight to fill the requested number of
        draws — better to fail loudly than to silently return a smaller,
        differently-shaped null distribution.
    """
    ncols = dem.shape[1]
    cand_a = candidate_cells(dem, valid, focus_a.elevation, elev_tol)
    cand_b = candidate_cells(dem, valid, focus_b.elevation, elev_tol)
    if cand_a.size == 0 or cand_b.size == 0:
        raise RuntimeError(
            f"No candidate cells within +/-{elev_tol} m for "
            f"{focus_a.name!r} ({cand_a.size}) / {focus_b.name!r} ({cand_b.size})."
        )

    observed_d = math.hypot(
        (focus_a.row - focus_b.row) * pixel_size,
        (focus_a.col - focus_b.col) * pixel_size,
    )
    lo, hi = observed_d * (1.0 - dist_tol), observed_d * (1.0 + dist_tol)

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    attempts = 0
    budget = n_pairs * max_attempts_per_pair
    while len(pairs) < n_pairs and attempts < budget:
        # Draw in blocks: rejection sampling one at a time is needlessly slow.
        block = max(1024, n_pairs)
        ia = rng.choice(cand_a, size=block)
        ib = rng.choice(cand_b, size=block)
        ra, ca = np.divmod(ia, ncols)
        rb, cb = np.divmod(ib, ncols)
        d = np.hypot((ra - rb) * pixel_size, (ca - cb) * pixel_size)
        keep = np.flatnonzero((d >= lo) & (d <= hi))
        for k in keep:
            pairs.append(((int(ra[k]), int(ca[k])), (int(rb[k]), int(cb[k]))))
            if len(pairs) == n_pairs:
                break
        attempts += block

    if len(pairs) < n_pairs:
        raise RuntimeError(
            f"Only {len(pairs)}/{n_pairs} matched pairs found for "
            f"{focus_a.name} x {focus_b.name} after {attempts} draws. "
            f"Relax --elev-tol / --dist-tol, or widen the analysis extent."
        )
    return pairs


# =============================================================================
# Inference
# =============================================================================

def empirical_p(observed: float, null_values: np.ndarray) -> float:
    """One-sided upper-tail p-value, ``p = (r + 1) / (n + 1)``.

    ``r`` counts null draws greater than or equal to the observed value.
    The +1 in both numerator and denominator counts the observed value
    itself as one realisation, so the smallest attainable p-value with
    999 draws is 0.001 and p can never be zero.
    """
    null_values = np.asarray(null_values, dtype=float)
    r = int(np.count_nonzero(null_values >= observed))
    return (r + 1) / (null_values.size + 1)


def empirical_p_lower(observed: float, null_values: np.ndarray) -> float:
    """One-sided lower-tail p-value, ``(r_le + 1) / (n + 1)``."""
    null_values = np.asarray(null_values, dtype=float)
    r = int(np.count_nonzero(null_values <= observed))
    return (r + 1) / (null_values.size + 1)


def empirical_p_two_sided(observed: float, null_values: np.ndarray) -> float:
    """Two-sided p-value: twice the smaller one-sided p-value, capped at 1."""
    return min(1.0, 2.0 * min(empirical_p(observed, null_values),
                              empirical_p_lower(observed, null_values)))


def holm(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values.

    Returns adjusted values in the original input order, monotonically
    non-decreasing in the sorted order and capped at 1.0.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)      # enforce monotonicity
        adjusted[idx] = min(running, 1.0)
    return adjusted


# =============================================================================
# Driver
# =============================================================================

def resolve_foci(foci: list[Focus], dem: np.ndarray, gt: tuple[float, ...]) -> None:
    """Fill in row, col and elevation for every focus, in place."""
    nrows, ncols = dem.shape
    for f in foci:
        f.row, f.col = world_to_pixel(gt, f.x, f.y)
        if not (0 <= f.row < nrows and 0 <= f.col < ncols):
            raise RuntimeError(
                f"Focus {f.name!r} at ({f.x}, {f.y}) falls outside the DEM."
            )
        f.elevation = float(dem[f.row, f.col])
        if not np.isfinite(f.elevation):
            raise RuntimeError(f"Focus {f.name!r} sits on a nodata cell.")
        _LOG.info("  %-32s row=%5d col=%5d z=%7.2f m", f.name, f.row, f.col, f.elevation)


def run_pair_tests(
    dem: np.ndarray,
    valid: np.ndarray,
    pixel_size: float,
    cell_ha: float,
    foci: dict[str, Focus],
    pairs: list[tuple[str, str]],
    backend: ViewshedBackend,
    observed_viewsheds: dict[str, np.ndarray] | None,
    n_null: int,
    elev_tol: float,
    dist_tol: float,
    seed: int,
) -> list[PairResult]:
    """Run both statistics for every pair and Holm-correct within each family."""
    rng = np.random.default_rng(seed)
    cache: dict[str, np.ndarray] = dict(observed_viewsheds or {})

    def viewshed_of(name: str) -> np.ndarray:
        if name not in cache:
            f = foci[name]
            _LOG.info("  computing observed viewshed (%s): %s", backend.name, name)
            cache[name] = backend.compute(f.row, f.col) & valid
        return cache[name]

    results: list[PairResult] = []

    for a_name, b_name in pairs:
        a, b = foci[a_name], foci[b_name]
        label = f"{a_name} x {b_name}"
        _LOG.info("Pair: %s", label)

        vs_a, vs_b = viewshed_of(a_name), viewshed_of(b_name)
        observed = {
            "union_ha": union_ha(vs_a, vs_b, cell_ha),
            "overlap_pct_smaller": pct_of_smaller(vs_a, vs_b),
        }
        _LOG.info("  observed union=%.2f ha  overlap=%.1f %% of smaller",
                  observed["union_ha"], observed["overlap_pct_smaller"])

        _LOG.info("  drawing %d matched null pairs (elev +/-%.1f m, dist +/-%.0f %%)",
                  n_null, elev_tol, dist_tol * 100)
        null_pairs = sample_matched_pairs(
            dem, valid, pixel_size, a, b, n_null, elev_tol, dist_tol, rng,
        )

        null = {key: np.empty(n_null, dtype=float) for key in observed}
        for i, (rc_a, rc_b) in enumerate(null_pairs):
            if i % 100 == 0:
                _LOG.info("    null draw %d/%d", i, n_null)
            na = backend.compute(*rc_a) & valid
            nb = backend.compute(*rc_b) & valid
            null["union_ha"][i] = union_ha(na, nb, cell_ha)
            null["overlap_pct_smaller"][i] = pct_of_smaller(na, nb)

        for key in observed:
            values = null[key]
            results.append(PairResult(
                pair=label,
                statistic=key,
                observed=observed[key],
                null_median=float(np.median(values)),
                null_max=float(values.max()),
                null_p05=float(np.percentile(values, 5)),
                null_p95=float(np.percentile(values, 95)),
                n_null=n_null,
                p_value=empirical_p(observed[key], values),
                null_values=values.tolist(),
            ))

    # Holm correction is applied within each statistic's family of tests,
    # not across the two families jointly.
    for key in STATISTICS:
        family = [r for r in results if r.statistic == key]
        if not family:
            continue
        adjusted = holm([r.p_value for r in family])
        for r, p_adj in zip(family, adjusted):
            r.p_holm = float(p_adj)

    return results


def write_results(results: list[PairResult], out_csv: Path, out_json: Path | None) -> None:
    """Write the results table (CSV) and, optionally, the full null draws (JSON)."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair", "statistic", "observed", "null_median", "null_p05",
            "null_p95", "null_max", "n_null", "p", "p_holm",
        ])
        for r in results:
            writer.writerow([
                r.pair, r.statistic, f"{r.observed:.4f}", f"{r.null_median:.4f}",
                f"{r.null_p05:.4f}", f"{r.null_p95:.4f}", f"{r.null_max:.4f}",
                r.n_null, f"{r.p_value:.4f}", f"{r.p_holm:.4f}",
            ])
    _LOG.info("Wrote results table to %s", out_csv)

    if out_json:
        payload = {
            "script": "viewshed_null.py",
            "script_version": __version__,
            "statistics": STATISTICS,
            "results": [
                {
                    "pair": r.pair,
                    "statistic": r.statistic,
                    "observed": r.observed,
                    "null_median": r.null_median,
                    "null_p05": r.null_p05,
                    "null_p95": r.null_p95,
                    "null_max": r.null_max,
                    "n_null": r.n_null,
                    "p": r.p_value,
                    "p_holm": r.p_holm,
                    "null_values": r.null_values,
                }
                for r in results
            ],
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        _LOG.info("Wrote full null distributions to %s", out_json)


def load_config(path: Path) -> tuple[list[Focus], list[tuple[str, str]]]:
    """Load foci and pairs from the JSON configuration file."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    foci = []
    for d in cfg["foci"]:
        for axis in ("x", "y"):
            if not isinstance(d[axis], (int, float)):
                raise RuntimeError(
                    f"Focus {d['name']!r} still has a placeholder {axis}-coordinate "
                    f"({d[axis]!r}). Fill in the real map coordinates, in the same "
                    f"projected CRS as the DEM, before running."
                )
        foci.append(Focus(name=d["name"], x=float(d["x"]), y=float(d["y"])))
    names = {f.name for f in foci}
    pairs: list[tuple[str, str]] = []
    for p in cfg["pairs"]:
        a, b = p["a"], p["b"]
        missing = {a, b} - names
        if missing:
            raise RuntimeError(f"Pair references unknown focus/foci: {sorted(missing)}")
        pairs.append((a, b))
    return foci, pairs


# =============================================================================
# Version 2: land-masked pool model
# =============================================================================

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount(packed: np.ndarray) -> int:
    """Number of set bits in a ``np.packbits`` array."""
    if hasattr(np, "bitwise_count"):                     # numpy >= 2.0
        return int(np.bitwise_count(packed).sum(dtype=np.int64))
    return int(_POPCOUNT[packed].sum(dtype=np.int64))


def packed_stats(pa: np.ndarray, pb: np.ndarray, cell_ha: float,
                 na: int | None = None, nb: int | None = None) -> dict[str, float]:
    """The four pool statistics from two bit-packed viewsheds."""
    na = popcount(pa) if na is None else na
    nb = popcount(pb) if nb is None else nb
    inter = popcount(pa & pb)
    union = na + nb - inter
    smaller = min(na, nb)
    return {
        "overlap_ha": inter * cell_ha,
        "overlap_pct_smaller": 100.0 * inter / smaller if smaller else 0.0,
        "jaccard": inter / union if union else 0.0,
        "union_ha": union * cell_ha,
    }


@dataclass
class AnalysisGrid:
    """The DEM on the analysis grid together with the analysis region."""
    dem: np.ndarray              # float64, NaN = nodata
    valid: np.ndarray            # DEM has data
    land: np.ndarray             # valid and not water  (= the analysis region)
    gt: tuple[float, ...]
    projection: str
    factor: int

    @property
    def pixel_size(self) -> float:
        return (abs(self.gt[1]) + abs(self.gt[5])) / 2.0

    @property
    def cell_ha(self) -> float:
        return abs(self.gt[1]) * abs(self.gt[5]) / M2_PER_HA


def build_analysis_grid(dem_path: Path, water_mask_path: Path | None, factor: int) -> AnalysisGrid:
    """Aggregate the DEM (block mean) and the water mask (majority) to the analysis grid.

    The water mask must be on exactly the DEM grid (same size and geotransform).
    Without a mask the analysis region is every valid DEM cell (the
    sea-included model of the original submission).
    """
    dem_raster = read_raster(dem_path)
    dem_fine = dem_raster.array.astype(np.float64)
    if dem_raster.nodata is not None:
        dem_fine = np.where(dem_fine == dem_raster.nodata, np.nan, dem_fine)
    dem = aggregate_mean(dem_fine, factor)
    valid = np.isfinite(dem)

    if water_mask_path is not None:
        wm = read_raster(water_mask_path)
        if wm.array.shape != dem_fine.shape or not np.allclose(wm.gt, dem_raster.gt):
            raise RuntimeError(
                f"Water mask {water_mask_path} is not on the DEM grid "
                f"(shape {wm.array.shape} vs {dem_fine.shape}, gt {wm.gt} vs {dem_raster.gt})."
            )
        water = aggregate_binary(wm.array > 0, factor, "majority")
        land = valid & ~water
    else:
        land = valid.copy()

    gt = list(dem_raster.gt)
    gt[1] *= factor
    gt[5] *= factor
    return AnalysisGrid(dem=dem, valid=valid, land=land, gt=tuple(gt),
                        projection=dem_raster.projection, factor=factor)


def write_grid_dem(grid: AnalysisGrid, path: Path, nodata: float = -99999.0) -> None:
    """Write the analysis-grid DEM (input for GRASS)."""
    arr = np.where(grid.valid, grid.dem, nodata).astype(np.float32)
    like = Raster(arr, grid.gt, grid.projection, nodata)
    write_raster(path, arr, like, nodata=nodata)


_GRASS_BATCH_SCRIPT = r'''
import json, sys
import numpy as np
import grass.script as gs
from grass.script import array as garray

dem_tif, job_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
job = json.load(open(job_path, encoding="utf-8"))
gs.run_command("r.in.gdal", input=dem_tif, output="dem", flags="o", overwrite=True, quiet=True)
gs.run_command("g.region", raster="dem")

if job["mode"] == "viewshed":
    count_mask = np.load(job["mask"])                 # cells that are counted (land)
    valid = np.load(job["valid"])
    packed, n_land, n_all = [], [], []
    for x, y in job["points"]:
        gs.run_command("r.viewshed", input="dem", output="vs", coordinates=(x, y),
                       observer_elevation=job["obs_h"], target_elevation=job["tgt_h"],
                       max_distance=job["max_d"], memory=job["memory"], flags="b",
                       overwrite=True, quiet=True)
        vs = garray.array("vs")
        vis = np.isfinite(vs) & (vs > 0)
        if vis.shape != count_mask.shape:
            raise RuntimeError("viewshed shape %s != grid %s" % (vis.shape, count_mask.shape))
        n_all.append(int(np.count_nonzero(vis & valid)))
        vis &= count_mask
        n_land.append(int(np.count_nonzero(vis)))
        packed.append(np.packbits(vis.ravel()))
    np.savez(out_path, packed=np.array(packed), n_land=np.array(n_land), n_all=np.array(n_all))

elif job["mode"] == "los":
    out = []
    for q in job["queries"]:
        gs.run_command("r.viewshed", input="dem", output="vs", coordinates=tuple(q["obs"]),
                       observer_elevation=q["obs_h"], target_elevation=q["tgt_h"],
                       max_distance=q["max_d"], memory=job["memory"], flags="b",
                       overwrite=True, quiet=True)
        res = gs.read_command("r.what", map="vs", coordinates=tuple(q["tgt"]), separator="|")
        val = res.strip().split("|")[-1]
        out.append(None if val in ("*", "") else float(val))
    json.dump(out, open(out_path, "w"))
'''


class GrassBatch:
    """Many ``r.viewshed`` runs in a few persistent GRASS sessions, in parallel."""

    def __init__(self, grid: AnalysisGrid, workdir: Path, grass_binary: str,
                 grass_python: str = "python", workers: int = 8,
                 observer_height: float = 1.7, target_height: float = 0.0,
                 max_distance: float = 3000.0, memory: int = 4000) -> None:
        self.grid = grid
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.grass_binary = grass_binary
        self.grass_python = grass_python
        self.workers = max(1, workers)
        self.obs_h, self.tgt_h, self.max_d, self.memory = (
            observer_height, target_height, max_distance, memory)
        self.dem_tif = self.workdir / f"_dem_f{grid.factor}.tif"
        write_grid_dem(grid, self.dem_tif)
        self.script = self.workdir / "_grass_batch.py"
        self.script.write_text(_GRASS_BATCH_SCRIPT, encoding="utf-8")
        np.save(self.workdir / "_land.npy", grid.land)
        np.save(self.workdir / "_valid.npy", grid.valid)

    def _run(self, tag: str, job: dict, out_suffix: str) -> Path:
        job_path = self.workdir / f"_job_{tag}.json"
        out_path = self.workdir / f"_out_{tag}{out_suffix}"
        text = json.dumps(job)
        # Resume: an identical job already finished in this work directory.
        if (out_path.exists() and job_path.exists()
                and job_path.read_text(encoding="utf-8") == text):
            _LOG.info("  reusing cached GRASS output %s", out_path.name)
            return out_path
        job_path.write_text(text, encoding="utf-8")
        cmd = [self.grass_binary, "--tmp-project", str(self.dem_tif), "--exec",
               self.grass_python, str(self.script), str(self.dem_tif), str(job_path), str(out_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              shell=self.grass_binary.lower().endswith(".bat"))
        if proc.returncode != 0 or not out_path.exists():
            raise RuntimeError(f"GRASS batch {tag} failed.\nstdout: {proc.stdout}\nstderr: {proc.stderr}")
        return out_path

    def viewsheds(self, points_xy: list[tuple[float, float]], tag: str
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bit-packed land viewsheds, land cell counts and all-valid cell counts."""
        from concurrent.futures import ThreadPoolExecutor

        chunks = [c for c in np.array_split(np.arange(len(points_xy)), self.workers) if c.size]
        common = {"mode": "viewshed", "obs_h": self.obs_h, "tgt_h": self.tgt_h,
                  "max_d": self.max_d, "memory": self.memory,
                  "mask": str(self.workdir / "_land.npy"), "valid": str(self.workdir / "_valid.npy")}

        def job_for(idx):
            return dict(common, points=[list(map(float, points_xy[i])) for i in idx])

        def cached(k, idx):
            out = self.workdir / f"_out_{tag}_{k}.npz"
            jp = self.workdir / f"_job_{tag}_{k}.json"
            return (out.exists() and out.stat().st_size > 0 and jp.exists()
                    and jp.read_text(encoding="utf-8") == json.dumps(job_for(idx)))

        # Chunks finished in an earlier (interrupted) run are reused; the rest
        # are split further so that all workers stay busy.
        missing = [k for k, idx in enumerate(chunks) if not cached(k, idx)]
        split = max(1, self.workers // max(1, len(missing)))
        tasks = []                                  # (chunk k, sub-tag, indices)
        for k, idx in enumerate(chunks):
            if k in missing and split > 1:
                for s, sub in enumerate(np.array_split(idx, split)):
                    if sub.size:
                        tasks.append((k, f"{tag}_{k}_s{s}of{split}", sub))
            else:
                tasks.append((k, f"{tag}_{k}", idx))

        def run_task(task):
            _, sub_tag, idx = task
            try:
                return np.load(self._run(sub_tag, job_for(idx), ".npz"))
            except RuntimeError:                    # one retry: transient GRASS I/O errors
                _LOG.warning("  GRASS batch %s failed once, retrying", sub_tag)
                return np.load(self._run(sub_tag, job_for(idx), ".npz"))

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            parts = list(ex.map(run_task, tasks))
        return (np.concatenate([p["packed"] for p in parts]),
                np.concatenate([p["n_land"] for p in parts]),
                np.concatenate([p["n_all"] for p in parts]))

    def los(self, queries: list[dict], tag: str = "los") -> list[float | None]:
        job = {"mode": "los", "memory": self.memory, "queries": queries}
        return json.load(open(self._run(tag, job, ".json"), encoding="utf-8"))


def internal_viewsheds(grid: AnalysisGrid, points_xy, observer_height=1.7,
                       target_height=0.0, max_distance=3000.0):
    """Same output as :meth:`GrassBatch.viewsheds`, with the internal backend."""
    eng = InternalViewshed(grid.dem, grid.pixel_size, observer_height, target_height, max_distance)
    packed, n_land, n_all = [], [], []
    for x, y in points_xy:
        r, c = world_to_pixel(grid.gt, x, y)
        vis = eng.compute(r, c) & grid.valid
        n_all.append(int(np.count_nonzero(vis)))
        vis &= grid.land
        n_land.append(int(np.count_nonzero(vis)))
        packed.append(np.packbits(vis.ravel()))
    return np.array(packed), np.array(n_land), np.array(n_all)


def draw_pool(grid: AnalysisGrid, size: int, rng: np.random.Generator) -> np.ndarray:
    """Flat indices of ``size`` distinct random cells from the analysis region (land)."""
    cells = np.flatnonzero(grid.land.ravel())
    if size > cells.size:
        raise RuntimeError(f"Pool size {size} exceeds the {cells.size} land cells.")
    return np.sort(rng.choice(cells, size=size, replace=False))


def matched_pool_pairs(pool_z: np.ndarray, pool_xy: np.ndarray, z_a: float, z_b: float,
                       separation: float, elev_tol: float, dist_tol: float) -> np.ndarray:
    """All (i, j) pool pairs matching (z_a, z_b, separation); shape (k, 2)."""
    ia = np.flatnonzero(np.abs(pool_z - z_a) <= elev_tol)
    ib = np.flatnonzero(np.abs(pool_z - z_b) <= elev_tol)
    if ia.size == 0 or ib.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    d = np.hypot(pool_xy[ia, 0][:, None] - pool_xy[ib, 0][None, :],
                 pool_xy[ia, 1][:, None] - pool_xy[ib, 1][None, :])
    ok = (np.abs(d - separation) <= dist_tol * separation) & (ia[:, None] != ib[None, :])
    r, c = np.nonzero(ok)
    return np.stack([ia[r], ib[c]], axis=1)


def summarise_null(observed: float, values: np.ndarray) -> dict:
    """Null summary, exceedance counts and the three p-values for one statistic."""
    n = values.size
    r_ge = int(np.count_nonzero(values >= observed))
    r_le = int(np.count_nonzero(values <= observed))
    p_up, p_lo = (r_ge + 1) / (n + 1), (r_le + 1) / (n + 1)
    return {
        "observed": float(observed),
        "null_median": float(np.median(values)),
        "null_q05": float(np.percentile(values, 5)),
        "null_q95": float(np.percentile(values, 95)),
        "null_min": float(values.min()),
        "null_max": float(values.max()),
        "n_null": int(n),
        "n_null_ge_observed": r_ge,
        "n_null_le_observed": r_le,
        "p_upper": p_up,
        "p_lower": p_lo,
        "p_two_sided": min(1.0, 2.0 * min(p_up, p_lo)),
        "beyond_all_draws": ("above" if r_ge == 0 else "below" if r_le == 0 else ""),
        "percentile": 100.0 * float(np.count_nonzero(values < observed)) / n,
    }


def line_of_sight_profile(dem: np.ndarray, gt: tuple[float, ...], a_xy, b_xy,
                          h_a: float, h_b: float) -> dict:
    """Exact sight-line test along the A->B profile (bilinear DEM, step = cell/4).

    The sight line runs from z(A) + h_a to z(B) + h_b. ``clearance_m`` is the
    smallest height of the sight line above the terrain between the two end
    cells (negative = blocked); ``blocked_at_m`` is the distance from A of
    that tightest point.
    """
    px = abs(gt[1])
    (xa, ya), (xb, yb) = a_xy, b_xy
    length = math.hypot(xb - xa, yb - ya)
    n = max(2, int(length / (px / 4.0)))
    t = np.linspace(0.0, 1.0, n + 1)
    x, y = xa + t * (xb - xa), ya + t * (yb - ya)
    fc = (x - gt[0]) / gt[1] - 0.5
    fr = (y - gt[3]) / gt[5] - 0.5
    c0 = np.clip(np.floor(fc).astype(int), 0, dem.shape[1] - 2)
    r0 = np.clip(np.floor(fr).astype(int), 0, dem.shape[0] - 2)
    dc, dr = np.clip(fc - c0, 0, 1), np.clip(fr - r0, 0, 1)
    z = (dem[r0, c0] * (1 - dr) * (1 - dc) + dem[r0, c0 + 1] * (1 - dr) * dc
         + dem[r0 + 1, c0] * dr * (1 - dc) + dem[r0 + 1, c0 + 1] * dr * dc)
    line = (z[0] + h_a) + t * ((z[-1] + h_b) - (z[0] + h_a))
    inner = (t * length > px) & ((1 - t) * length > px)     # skip the two end cells
    margin = np.where(inner, line - z, np.inf)
    k = int(np.nanargmin(margin))
    return {"visible": bool(margin[k] >= 0), "clearance_m": float(margin[k]),
            "blocked_at_m": float(t[k] * length), "length_m": float(length)}


def truncation_summary(grid: AnalysisGrid, x: float, y: float, vis_packed: np.ndarray,
                       max_distance: float) -> dict:
    """How far the survey extent, rather than the radius, limits one viewshed."""
    from scipy import ndimage  # lazy import, only needed here

    rows, cols = grid.dem.shape
    r, c = world_to_pixel(grid.gt, x, y)
    yy = (np.arange(rows)[:, None] - r) * abs(grid.gt[5])
    xx = (np.arange(cols)[None, :] - c) * abs(grid.gt[1])
    dist = np.hypot(xx, yy)
    disk_area_ha = math.pi * max_distance ** 2 / M2_PER_HA
    in_disk_valid = grid.valid & (dist <= max_distance)
    vis = np.unpackbits(vis_packed, count=rows * cols).reshape(rows, cols).astype(bool)
    # visible cells touching the edge of the DEM (nodata or raster border)
    edge = ~ndimage.binary_erosion(np.pad(grid.valid, 1), np.ones((3, 3)))[1:-1, 1:-1] & grid.valid
    return {
        "radius_m": max_distance,
        "disk_area_ha": disk_area_ha,
        "dem_valid_in_disk_ha": float(np.count_nonzero(in_disk_valid) * grid.cell_ha),
        "pct_of_disk_covered_by_dem": 100.0 * np.count_nonzero(in_disk_valid) * grid.cell_ha / disk_area_ha,
        "farthest_valid_cell_m": float(dist[grid.valid].max()),
        "farthest_visible_cell_m": float(dist[vis].max()) if vis.any() else 0.0,
        "visible_cells_on_dem_edge": int(np.count_nonzero(vis & edge)),
        "truncated_by_survey_extent": bool(dist[grid.valid].max() < max_distance or
                                           np.count_nonzero(vis & edge) > 0),
    }


def run_pool_model(grid: AnalysisGrid, foci: list[Focus], pairs: list[tuple[str, str]],
                   compute_viewsheds, pool_size: int, n_null: int, elev_tol: float,
                   dist_tol: float, seed: int, pool_index: np.ndarray | None = None) -> dict:
    """Run the land-masked pool null model. Returns everything needed for output.

    ``compute_viewsheds(points_xy, tag)`` must return (packed, n_land, n_all).
    ``pool_index`` reuses a given set of pool cells (flat indices on this grid)
    instead of drawing a new pool — used for the resolution check.
    """
    rng = np.random.default_rng(seed)
    rows, cols = grid.dem.shape
    if pool_index is None:
        pool_index = draw_pool(grid, pool_size, rng)
    pr, pc = np.divmod(pool_index, cols)
    pool_xy = np.array([pixel_to_world(grid.gt, int(a), int(b)) for a, b in zip(pr, pc)])
    pool_z = grid.dem[pr, pc]

    resolve_foci(foci, grid.dem, grid.gt)
    for f in foci:
        if not grid.land[f.row, f.col]:
            _LOG.warning("Focus %s falls on a water/no-data cell of the analysis grid.", f.name)
    fmap = {f.name: f for f in foci}

    _LOG.info("Computing %d focus viewsheds", len(foci))
    f_packed, f_land, f_all = compute_viewsheds([(f.x, f.y) for f in foci], "foci")
    fidx = {f.name: i for i, f in enumerate(foci)}
    _LOG.info("Computing %d pool viewsheds", len(pool_xy))
    p_packed, p_land, p_all = compute_viewsheds([tuple(p) for p in pool_xy], "pool")

    results, draws = [], {}
    for a_name, b_name in pairs:
        A, B = fmap[a_name], fmap[b_name]
        ia, ib = fidx[a_name], fidx[b_name]
        sep = math.hypot(A.x - B.x, A.y - B.y)
        obs = packed_stats(f_packed[ia], f_packed[ib], grid.cell_ha, f_land[ia], f_land[ib])
        avail = matched_pool_pairs(pool_z, pool_xy, A.elevation, B.elevation, sep, elev_tol, dist_tol)
        if avail.shape[0] == 0:
            raise RuntimeError(f"No matched pool pairs for {a_name} x {b_name}.")
        replace = avail.shape[0] < n_null
        if replace:
            _LOG.warning("%s x %s: only %d matched pairs, drawing with replacement",
                         a_name, b_name, avail.shape[0])
        pick = avail[rng.choice(avail.shape[0], size=n_null, replace=replace)]
        vals = {k: np.empty(n_null) for k in POOL_STATISTICS}
        for i, (u, v) in enumerate(pick):
            s = packed_stats(p_packed[u], p_packed[v], grid.cell_ha, p_land[u], p_land[v])
            for k in POOL_STATISTICS:
                vals[k][i] = s[k]
        draws[f"{a_name}_x_{b_name}"] = {"pool_pairs": pick, **vals}
        for k in POOL_STATISTICS:
            row = {"pair": f"{a_name} x {b_name}", "a": a_name, "b": b_name,
                   "separation_m": sep, "z_a": A.elevation, "z_b": B.elevation,
                   "statistic": k, "n_matched_available": int(avail.shape[0]),
                   "drawn_with_replacement": bool(replace)}
            row.update(summarise_null(obs[k], vals[k]))
            results.append(row)
        _LOG.info("  %-36s overlap=%.2f ha (%.1f %%), J=%.3f, union=%.2f ha, avail=%d",
                  f"{a_name} x {b_name}", obs["overlap_ha"], obs["overlap_pct_smaller"],
                  obs["jaccard"], obs["union_ha"], avail.shape[0])

    for k in POOL_STATISTICS:
        fam = [r for r in results if r["statistic"] == k]
        for r, p in zip(fam, holm([r["p_two_sided"] for r in fam])):
            r["p_holm_two_sided"] = float(p)

    return {
        "results": results, "draws": draws,
        "pool": {"index": pool_index, "xy": pool_xy, "z": pool_z,
                 "land_ha": p_land * grid.cell_ha, "all_ha": p_all * grid.cell_ha},
        "foci": [{"name": f.name, "x": f.x, "y": f.y, "row": f.row, "col": f.col,
                  "z": f.elevation, "on_land": bool(grid.land[f.row, f.col]),
                  "visible_land_ha": float(f_land[i] * grid.cell_ha),
                  "visible_all_ha": float(f_all[i] * grid.cell_ha)}
                 for i, f in enumerate(foci)],
        "foci_packed": f_packed,
        "region": {"cell_size_m": grid.pixel_size, "rows": rows, "cols": cols,
                   "valid_ha": float(np.count_nonzero(grid.valid) * grid.cell_ha),
                   "land_ha": float(np.count_nonzero(grid.land) * grid.cell_ha),
                   "water_or_excluded_ha": float(np.count_nonzero(grid.valid & ~grid.land) * grid.cell_ha)},
    }


def write_pool_outputs(out: dict, outdir: Path, meta: dict) -> None:
    """Write results CSV/JSON, null draws (.npz), pool and foci tables."""
    outdir.mkdir(parents=True, exist_ok=True)
    keys = ["pair", "a", "b", "separation_m", "z_a", "z_b", "statistic", "observed",
            "null_median", "null_q05", "null_q95", "null_min", "null_max", "n_null",
            "n_matched_available", "drawn_with_replacement", "n_null_ge_observed",
            "n_null_le_observed", "beyond_all_draws", "percentile", "p_upper", "p_lower",
            "p_two_sided", "p_holm_two_sided"]
    with open(outdir / "results_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in out["results"]:
            w.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in keys})
    np.savez_compressed(outdir / "null_draws.npz", **{
        f"{pair}__{k}": v for pair, d in out["draws"].items() for k, v in d.items()})
    with open(outdir / "pool_points.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pool_id", "x", "y", "z", "visible_land_ha", "visible_all_ha"])
        p = out["pool"]
        for i in range(len(p["z"])):
            w.writerow([i, f"{p['xy'][i, 0]:.3f}", f"{p['xy'][i, 1]:.3f}", f"{p['z'][i]:.3f}",
                        f"{p['land_ha'][i]:.4f}", f"{p['all_ha'][i]:.4f}"])
    payload = {"script": "viewshed_null.py", "script_version": __version__, **meta,
               "statistics": POOL_STATISTICS, "region": out["region"], "foci": out["foci"],
               "results": out["results"]}
    with open(outdir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="viewshed_null",
        description="Pairwise viewshed overlap and combined-area tests against "
                    "a matched null model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dem", required=True, type=Path,
                   help="Low-pass filtered DEM (GeoTIFF), in a projected CRS in metres.")
    p.add_argument("--config", required=True, type=Path,
                   help="JSON file listing the foci and the pairs to test.")
    p.add_argument("--mode", choices=["pool", "direct"], default="pool",
                   help="'pool' = land-masked pool model (Table 1 / S5, version 2); "
                        "'direct' = the version-1 per-pair sampler.")
    p.add_argument("--water-mask", type=Path, default=None,
                   help="0/1 raster on the DEM grid, 1 = sea/harbour. Water cells are "
                        "excluded from the counted viewsheds and from random sampling.")
    p.add_argument("--outdir", type=Path, default=None,
                   help="pool mode: output directory.")
    p.add_argument("--pool-size", type=int, default=3000,
                   help="pool mode: number of random land points.")
    p.add_argument("--workers", type=int, default=8,
                   help="pool mode, GRASS backend: parallel GRASS sessions.")
    p.add_argument("--grass-python", default="python",
                   help="Python interpreter to run inside the GRASS session.")
    p.add_argument("--output", type=Path, default=None,
                   help="direct mode: path for the results CSV.")
    p.add_argument("--output-json", type=Path, default=None,
                   help="direct mode: also write the full null distributions to this JSON path.")

    p.add_argument("--observed-dir", type=Path, default=None,
                   help="Directory of pre-computed observed viewshed GeoTIFFs, "
                        "one per focus, named <focus name>.tif. These are the "
                        "GRASS outputs archived with the dataset.")
    p.add_argument("--recompute-observed", action="store_true",
                   help="Ignore --observed-dir and recompute the observed "
                        "viewsheds with the selected backend. Required for a "
                        "like-for-like comparison when --backend internal is used.")

    p.add_argument("--backend", choices=["grass", "internal"], default="grass",
                   help="Viewshed engine for the null draws.")
    p.add_argument("--grass-binary", default="grass",
                   help="Name or path of the GRASS launcher executable.")

    p.add_argument("--aggregate", type=int, default=5,
                   help="Block factor applied to the DEM and viewsheds "
                        "(5 takes a 0.4 m grid to the 2 m analysis grid).")
    p.add_argument("--aggregate-rule", choices=["majority", "any", "all"],
                   default="majority",
                   help="Rule for aggregating binary viewsheds.")
    p.add_argument("--observer-height", type=float, default=1.7,
                   help="Observer height above ground, metres.")
    p.add_argument("--target-height", type=float, default=0.0,
                   help="Target height above ground, metres.")
    p.add_argument("--max-distance", type=float, default=3000.0,
                   help="Maximum viewshed radius, metres.")

    p.add_argument("--n-null", type=int, default=999,
                   help="Number of matched random pairs per test.")
    p.add_argument("--elev-tol", type=float, default=5.0,
                   help="Elevation matching tolerance, metres.")
    p.add_argument("--dist-tol", type=float, default=0.10,
                   help="Separation matching tolerance, as a fraction.")
    p.add_argument("--seed", type=int, default=20260814,
                   help="Random seed, for an exactly reproducible null model. "
                        "The default is the seed used for the published results.")

    p.add_argument("--pool-from", type=Path, default=None,
                   help="pool mode: reuse the pool points of an earlier run (its "
                        "pool_points.csv) instead of drawing new ones.")
    p.add_argument("--pool-subset-bands", action="store_true",
                   help="With --pool-from: keep only pool points inside the elevation "
                        "band of a tested focus (saves time at 0.4 m).")
    p.add_argument("--only-pairs", default=None,
                   help="Comma-separated A:B names; test only these pairs.")

    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Console log level.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _pool_from_csv(path: Path, grid: AnalysisGrid) -> np.ndarray:
    """Flat indices (on ``grid``) of the pool points listed in a pool_points.csv."""
    idx = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r, c = world_to_pixel(grid.gt, float(row["x"]), float(row["y"]))
            idx.append(r * grid.dem.shape[1] + c)
    return np.array(idx, dtype=np.int64)


def main_pool(args) -> int:
    """Land-masked pool model + line of sight + truncation check."""
    from datetime import datetime, timezone

    if args.outdir is None:
        raise SystemExit("--outdir is required in pool mode")
    grid = build_analysis_grid(args.dem, args.water_mask, args.aggregate)
    _LOG.info("Analysis grid %d x %d, %.4f m; valid %.2f ha, land %.2f ha",
              grid.dem.shape[0], grid.dem.shape[1], grid.pixel_size,
              np.count_nonzero(grid.valid) * grid.cell_ha, np.count_nonzero(grid.land) * grid.cell_ha)

    foci, pairs = load_config(args.config)
    if args.only_pairs:
        wanted = [tuple(s.split(":")) for s in args.only_pairs.split(",")]
        pairs = [p for p in pairs if p in wanted]
    used = {n for p in pairs for n in p}
    foci = [f for f in foci if f.name in used or not args.only_pairs]

    if args.backend == "grass":
        gb = GrassBatch(grid, args.outdir / "_grass", args.grass_binary, args.grass_python,
                        args.workers, args.observer_height, args.target_height, args.max_distance)
        compute = gb.viewsheds
    else:
        gb = None
        compute = lambda pts, tag: internal_viewsheds(          # noqa: E731
            grid, pts, args.observer_height, args.target_height, args.max_distance)

    pool_index = None
    if args.pool_from:
        pool_index = _pool_from_csv(args.pool_from, grid)
        if args.pool_subset_bands:
            resolve_foci(foci, grid.dem, grid.gt)
            z = grid.dem.ravel()[pool_index]
            keep = np.zeros(z.size, bool)
            for f in foci:
                keep |= np.abs(z - f.elevation) <= args.elev_tol
            _LOG.info("Pool subset: %d of %d points lie in a tested elevation band",
                      keep.sum(), z.size)
            pool_index = pool_index[keep]

    out = run_pool_model(grid, foci, pairs, compute, args.pool_size, args.n_null,
                         args.elev_tol, args.dist_tol, args.seed, pool_index)

    # Direct line of sight, both directions, target 1.7 m (and 0 m for reference).
    fmap = {f.name: f for f in foci}
    los_rows, queries = [], []
    for a_name, b_name in pairs:
        for src, dst in ((a_name, b_name), (b_name, a_name)):
            for tgt_h in (1.7, 0.0):
                S, T = fmap[src], fmap[dst]
                prof = line_of_sight_profile(grid.dem, grid.gt, (S.x, S.y), (T.x, T.y),
                                             args.observer_height, tgt_h)
                los_rows.append({"pair": f"{a_name} x {b_name}", "from": src, "to": dst,
                                 "observer_h_m": args.observer_height, "target_h_m": tgt_h,
                                 **{f"profile_{k}": v for k, v in prof.items()}})
                queries.append({"obs": [S.x, S.y], "tgt": [T.x, T.y],
                                "obs_h": args.observer_height, "tgt_h": tgt_h,
                                "max_d": prof["length_m"] + 3 * grid.pixel_size})
    if gb is not None:
        for row, v in zip(los_rows, gb.los(queries)):
            row["grass_r_viewshed_visible"] = bool(v) if v is not None else False
    with open(args.outdir / "line_of_sight.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(los_rows[0].keys()))
        w.writeheader()
        w.writerows(los_rows)

    # Truncation: survey extent versus the 3,000 m radius.
    trunc = []
    for i, f in enumerate(foci):
        t = truncation_summary(grid, f.x, f.y, out["foci_packed"][i], args.max_distance)
        trunc.append({"focus": f.name, **t})
    with open(args.outdir / "viewshed_truncation.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(trunc[0].keys()))
        w.writeheader()
        w.writerows(trunc)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "dem": str(args.dem), "water_mask": str(args.water_mask) if args.water_mask else None,
            "aggregate_factor": args.aggregate, "cell_size_m": grid.pixel_size,
            "backend": args.backend, "observer_height_m": args.observer_height,
            "target_height_m": args.target_height, "max_distance_m": args.max_distance,
            "pool_size": int(len(out["pool"]["z"])), "pool_from": str(args.pool_from) if args.pool_from else None,
            "n_null": args.n_null, "elev_tol_m": args.elev_tol, "dist_tol": args.dist_tol,
            "seed": args.seed, "p_value": "two-sided, min(1, 2*min(upper, lower)); upper=(r_ge+1)/(n+1)",
            "multiple_comparisons": "Holm across the tested pairs, per statistic",
        },
        "line_of_sight": los_rows, "truncation": trunc,
    }
    write_pool_outputs(out, args.outdir, meta)
    _LOG.info("Wrote pool-model outputs to %s", args.outdir)
    for r in out["results"]:
        if r["statistic"] in ("overlap_pct_smaller", "jaccard", "union_ha"):
            _LOG.info("%-34s %-20s obs=%9.3f med=%9.3f p2=%.3f holm=%.3f %s",
                      r["pair"], r["statistic"], r["observed"], r["null_median"],
                      r["p_two_sided"], r["p_holm_two_sided"], r["beyond_all_draws"])
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.mode == "pool":
        return main_pool(args)
    if args.output is None:
        raise SystemExit("--output is required in direct mode")

    if args.backend == "internal" and args.observed_dir and not args.recompute_observed:
        _LOG.error(
            "Refusing to compare GRASS observed viewsheds against an internally "
            "computed null. Either use --backend grass, or add "
            "--recompute-observed to compute both sides with the internal backend."
        )
        return 2

    _LOG.info("Loading DEM: %s", args.dem)
    dem_raster = read_raster(args.dem)
    dem_fine = dem_raster.array.astype(np.float64)
    if dem_raster.nodata is not None:
        dem_fine = np.where(dem_fine == dem_raster.nodata, np.nan, dem_fine)

    _LOG.info("Aggregating by %d (%.2f m -> %.2f m)",
              args.aggregate, dem_raster.pixel_size,
              dem_raster.pixel_size * args.aggregate)
    dem = aggregate_mean(dem_fine, args.aggregate)
    valid = np.isfinite(dem)

    gt = list(dem_raster.gt)
    gt[1] *= args.aggregate
    gt[5] *= args.aggregate
    gt = tuple(gt)
    pixel_size = abs(gt[1])
    cell_ha = (abs(gt[1]) * abs(gt[5])) / M2_PER_HA

    _LOG.info("Analysis grid: %d x %d cells, %.2f m, valid extent %.3f km2",
              dem.shape[0], dem.shape[1], pixel_size,
              np.count_nonzero(valid) * pixel_size ** 2 / 1e6)

    foci_list, pairs = load_config(args.config)
    _LOG.info("Resolving %d foci:", len(foci_list))
    resolve_foci(foci_list, dem, gt)
    foci = {f.name: f for f in foci_list}

    observed_viewsheds: dict[str, np.ndarray] = {}
    if args.observed_dir and not args.recompute_observed:
        for name in {n for pair in pairs for n in pair}:
            tif = args.observed_dir / f"{name}.tif"
            if not tif.exists():
                raise RuntimeError(f"Missing observed viewshed for {name!r}: {tif}")
            raw = read_raster(tif)
            mask = np.isfinite(raw.array) & (raw.array > 0)
            observed_viewsheds[name] = aggregate_binary(
                mask, args.aggregate, args.aggregate_rule
            ) & valid
            _LOG.info("  loaded observed viewshed: %s", tif.name)

    if args.backend == "grass":
        # GRASS must see the DEM on the analysis grid, otherwise its output
        # (full resolution) cannot match the aggregated grid (fixed in 2.0.0).
        agg_dem = Path(tempfile.mkdtemp(prefix="vsnull_")) / f"dem_f{args.aggregate}.tif"
        write_raster(agg_dem, np.where(valid, dem, -99999.0).astype(np.float32),
                     Raster(dem, gt, dem_raster.projection, -99999.0), nodata=-99999.0)
        backend: ViewshedBackend = GrassViewshed(
            dem_path=agg_dem, gt=gt, shape=dem.shape,
            observer_height=args.observer_height,
            target_height=args.target_height,
            max_distance=args.max_distance,
            grass_binary=args.grass_binary,
            grass_python=args.grass_python,
        )
    else:
        backend = InternalViewshed(
            dem=dem, pixel_size=pixel_size,
            observer_height=args.observer_height,
            target_height=args.target_height,
            max_distance=args.max_distance,
        )
    _LOG.info("Viewshed backend: %s", backend.name)

    results = run_pair_tests(
        dem=dem, valid=valid, pixel_size=pixel_size, cell_ha=cell_ha,
        foci=foci, pairs=pairs, backend=backend,
        observed_viewsheds=observed_viewsheds or None,
        n_null=args.n_null, elev_tol=args.elev_tol,
        dist_tol=args.dist_tol, seed=args.seed,
    )

    write_results(results, args.output, args.output_json)

    _LOG.info("")
    _LOG.info("%-46s %-22s %10s %10s %8s %8s", "pair", "statistic",
              "observed", "null med", "p", "p Holm")
    for r in results:
        _LOG.info("%-46s %-22s %10.3f %10.3f %8.3f %8.3f",
                  r.pair, r.statistic, r.observed, r.null_median,
                  r.p_value, r.p_holm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
