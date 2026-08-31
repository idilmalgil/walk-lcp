#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viewshed_null.py — Pairwise viewshed overlap against a matched null model.

Companion to ``walk_lcp.py`` in the reproducibility package accompanying the
manuscript. Where ``walk_lcp.py`` produces the least-cost routes, this module
produces the two viewshed inference results reported in the manuscript:

* the **union test** — the combined (union) visible area of a pair of foci,
  in hectares, compared against a matched null distribution;
* the **overlap test** — the shared (intersection) visible ground expressed
  as a percentage of the smaller of the two viewsheds, compared against the
  same matched null distribution.

Method
------
For each observed pair of foci (A, B):

1. Load (or compute) the binary viewshed of A and of B.
2. Aggregate both viewsheds and the DEM to the analysis grid (2 m by
   default — the effective resolution of the low-pass filtered surface;
   the nominal 0.4 m grid is oversampled for this purpose).
3. Compute the two observed statistics over the common analysis extent
   (the valid-data mask, *not* the raster bounding box).
4. Draw ``n`` random pairs (a', b') from the same DEM, matched to the
   observed pair on:
     - elevation: ``|z(a') - z(A)| <= elev_tol`` and
       ``|z(b') - z(B)| <= elev_tol``   (default 5 m)
     - separation: ``|d(a', b') - d(A, B)| <= dist_tol * d(A, B)``
       (default 10 %)
5. Compute the same two statistics for every null pair.
6. Report the one-sided upper-tail p-value ``p = (r + 1) / (n + 1)``,
   where ``r`` is the number of null values greater than or equal to the
   observed value, and apply the Holm correction across the family of
   tests supplied in one run.

All test directions are fixed a priori (upper tail): the hypothesis under
test is that the observed pair is *more* intervisible / *more* extensive in
combined coverage than terrain alone predicts at comparable elevation and
separation.

Viewshed backends
-----------------
``--backend grass`` (default)
    Shells out to GRASS GIS ``r.viewshed``. This is the backend that
    produced the published results and is the one to use for
    reproduction. Requires ``grass`` on PATH.

``--backend internal``
    A self-contained NumPy R2 (radial sweep) viewshed. It requires no
    external software and is what the unit tests and the demo exercise.
    It is an *approximation* of ``r.viewshed`` and will not reproduce the
    published numbers cell-for-cell.

The two backends must not be mixed within a run: if the observed
viewsheds were produced by GRASS, the null draws must be too, or the
comparison is not like-for-like. The CLI enforces this — see
``--observed-dir`` and ``--recompute-observed``.

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

__version__ = "1.2.0"
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


STATISTICS = {
    "union_ha": "Combined (union) visible area, hectares",
    "overlap_pct_smaller": "Shared visible ground, % of the smaller viewshed",
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
    ) -> None:
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
                self.grass_binary, "--tmp-location", str(self.dem_path),
                "--exec", sys.executable, str(script),
                str(self.dem_path), str(out_tif),
                f"{east:.6f}", f"{north:.6f}",
                str(self.observer_height), str(self.target_height),
                str(self.max_distance),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
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
    p.add_argument("--output", required=True, type=Path,
                   help="Path for the results CSV.")
    p.add_argument("--output-json", type=Path, default=None,
                   help="Also write the full null distributions to this JSON path.")

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
    p.add_argument("--seed", type=int, default=20260525,
                   help="Random seed, for an exactly reproducible null model.")

    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Console log level.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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
        backend: ViewshedBackend = GrassViewshed(
            dem_path=args.dem, gt=gt, shape=dem.shape,
            observer_height=args.observer_height,
            target_height=args.target_height,
            max_distance=args.max_distance,
            grass_binary=args.grass_binary,
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
