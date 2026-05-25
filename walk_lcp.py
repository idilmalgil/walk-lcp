#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
walk_lcp.py — Least-cost walking paths over a DEM with Tobler's hiking function.

Computes least-cost paths (LCPs) and travel times on a digital elevation
model using Dijkstra's algorithm with edge costs derived from Tobler's
hiking function (Tobler 1993). For every (start, stop) pair the algorithm
returns the time-optimal walking route across the surface together with
its length and travel time.

This is the standalone (non-QGIS) version intended for the reproducibility
package accompanying the manuscript. For a version that runs inside QGIS
see ``walk_lcp_qgis.py``.

Algorithm
---------
1. Read the DEM (GeoTIFF) into a NumPy array.
2. For each start point, run 8-connected Dijkstra on the grid where the
   edge cost between adjacent cells (i, j) and (i', j') is
   distance / Tobler_speed(slope), in seconds.
3. For each stop point, reconstruct the optimal path back to the start
   using the predecessor array.
4. Write all routes to a single polyline layer (GeoPackage by default,
   or ESRI Shapefile if requested) with route_id, start_id, stop_id,
   dist_m, time_s, time_min, time_hr attributes.
5. Optionally export per-start cumulative time rasters (GeoTIFF).
6. Write a JSON manifest of the run (parameters, software versions,
   input/output file hashes, timestamp) to support exact verification
   by reviewers.

Tobler's hiking function
------------------------
Walking speed (km/h) as a function of slope S (rise/run):

    v(S) = 6 * exp(-3.5 * |S + 0.05|)

Peak speed (~6 km/h) occurs at S = -0.05 (5 % downhill).

References
----------
- Tobler, W. (1993). Three Presentations on Geographical Analysis and
  Modeling. NCGIA Technical Report 93-1, University of California,
  Santa Barbara.
- Dijkstra, E. W. (1959). A note on two problems in connexion with
  graphs. Numerische Mathematik 1: 269-271.

Author      : [FILL: your name]
Affiliation : [FILL: your institution]
Contact     : [FILL: your email]
License     : MIT (see LICENSE)
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import logging
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

# Geospatial libraries (geopandas, shapely, osgeo.gdal) are imported lazily
# inside the I/O functions so that the pure-math layer of this module
# (tobler_speed, world_to_pixel, pixel_to_world, dijkstra_from_start,
# reconstruct_path) can be imported and tested with just numpy installed.

__version__ = "1.0.0"
_LOG = logging.getLogger("walk_lcp")


# =============================================================================
# Tobler's hiking function
# =============================================================================

def tobler_speed(gradient: float) -> float:
    """Walking speed (m/s) from Tobler's hiking function.

    Parameters
    ----------
    gradient : float
        Slope as rise/run (dimensionless). Positive = uphill,
        negative = downhill.

    Returns
    -------
    float
        Speed in metres per second; clamped to a minimum of 0.01 m/s
        so that effectively infinite costs do not arise on cliffs.

    References
    ----------
    Tobler (1993). NCGIA Technical Report 93-1.
    """
    v_kmh = 6.0 * math.exp(-3.5 * abs(gradient + 0.05))
    return max(v_kmh / 3.6, 0.01)


# =============================================================================
# Pixel / world coordinate helpers
# =============================================================================

def world_to_pixel(gt: Sequence[float], x: float, y: float) -> tuple[int, int]:
    """Convert (x, y) map coordinates to (row, col) DEM pixel indices."""
    origin_x, px_w, _, origin_y, _, px_h = gt
    col = int((x - origin_x) / px_w)
    row = int((origin_y - y) / abs(px_h))
    return row, col


def pixel_to_world(gt: Sequence[float], row: int, col: int) -> tuple[float, float]:
    """Convert (row, col) pixel indices to map coordinates of the cell centre."""
    origin_x, px_w, _, origin_y, _, px_h = gt
    x = origin_x + (col + 0.5) * px_w
    y = origin_y - (row + 0.5) * abs(px_h)
    return x, y


# =============================================================================
# Dijkstra least-cost search
# =============================================================================

def dijkstra_from_start(
    dem: np.ndarray,
    gt: Sequence[float],
    start_rc: tuple[int, int],
    nodata: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-source Dijkstra on the DEM grid using Tobler edge costs.

    Parameters
    ----------
    dem : np.ndarray
        2D elevation array (float).
    gt : sequence of 6 floats
        GDAL-style affine GeoTransform.
    start_rc : (int, int)
        (row, col) of the start cell.
    nodata : float or None
        NoData value to exclude (cells equal to this are not traversed).

    Returns
    -------
    cost : np.ndarray
        2D array of cumulative travel time (seconds) from start.
        Unreachable cells contain np.inf.
    prev : np.ndarray, shape (rows, cols, 2), int32
        Predecessor (row, col) for each cell, or (-1, -1) if unset.
    """
    nrows, ncols = dem.shape
    sr, sc = start_rc

    cost = np.full((nrows, ncols), np.inf, dtype=np.float64)
    prev = np.full((nrows, ncols, 2), -1, dtype=np.int32)
    pq: list[tuple[float, int, int]] = []

    cost[sr, sc] = 0.0
    heapq.heappush(pq, (0.0, sr, sc))

    px_w = gt[1]
    px_h = abs(gt[5])

    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1))

    while pq:
        c, r, c0 = heapq.heappop(pq)
        if c > cost[r, c0]:
            continue

        z_here = dem[r, c0]
        if nodata is not None and z_here == nodata:
            continue

        for dr, dc in neighbors:
            rr, cc = r + dr, c0 + dc
            if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
                continue

            z_there = dem[rr, cc]
            if nodata is not None and z_there == nodata:
                continue

            dx = dc * px_w
            dy = dr * px_h
            dist = math.hypot(dx, dy)
            if dist == 0:
                continue

            slope = (z_there - z_here) / dist
            v = tobler_speed(slope)
            step_cost = dist / v

            newcost = c + step_cost
            if newcost < cost[rr, cc]:
                cost[rr, cc] = newcost
                prev[rr, cc] = [r, c0]
                heapq.heappush(pq, (newcost, rr, cc))

    return cost, prev


def reconstruct_path(
    prev: np.ndarray,
    gt: Sequence[float],
    start_rc: tuple[int, int],
    stop_rc: tuple[int, int],
) -> tuple[list[tuple[int, int]] | None, float | None]:
    """Walk the predecessor array from stop back to start.

    Returns
    -------
    path_rc : list of (row, col), or None if unreachable.
    total_distance_m : float, or None if unreachable.
    """
    sr, sc = start_rc
    tr, tc = stop_rc
    r, c0 = tr, tc

    if prev[r, c0, 0] == -1 and (r, c0) != (sr, sc):
        return None, None

    px_w = gt[1]
    px_h = abs(gt[5])

    path: list[tuple[int, int]] = []
    total_dist = 0.0

    while True:
        path.append((r, c0))
        if (r, c0) == (sr, sc):
            break
        pr, pc = prev[r, c0]
        dx = (c0 - pc) * px_w
        dy = (r - pr) * px_h
        total_dist += math.hypot(dx, dy)
        r, c0 = int(pr), int(pc)

    path.reverse()
    return path, total_dist


# =============================================================================
# I/O
# =============================================================================

@dataclass
class PointRecord:
    fid: int
    x: float
    y: float
    row: int
    col: int


def read_points(shp_path: Path, gt: Sequence[float]) -> list[PointRecord]:
    """Read a point layer (shapefile / GeoPackage) and resolve pixel indices."""
    import geopandas as gpd  # lazy import

    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise RuntimeError(f"Point layer is empty: {shp_path}")
    if not all(geom.geom_type == "Point" for geom in gdf.geometry):
        raise RuntimeError(f"Layer is not a point layer: {shp_path}")
    pts = []
    for idx, row in gdf.iterrows():
        x, y = float(row.geometry.x), float(row.geometry.y)
        r, c = world_to_pixel(gt, x, y)
        pts.append(PointRecord(fid=int(idx), x=x, y=y, row=r, col=c))
    return pts


def write_routes(
    routes: list[dict],
    gt: Sequence[float],
    out_path: Path,
    crs,
) -> None:
    """Write all routes to a single polyline layer (GeoPackage or Shapefile)."""
    import geopandas as gpd  # lazy import
    from shapely.geometry import LineString  # lazy import

    records = []
    for r in routes:
        if not r["path_rc"]:
            continue
        coords = [pixel_to_world(gt, rr, cc) for (rr, cc) in r["path_rc"]]
        records.append({
            "route_id": int(r["route_id"]),
            "start_id": int(r["start_id"]),
            "stop_id":  int(r["stop_id"]),
            "dist_m":   float(r["dist_m"]),
            "time_s":   float(r["time_s"]),
            "time_min": float(r["time_s"] / 60.0),
            "time_hr":  float(r["time_s"] / 3600.0),
            "geometry": LineString(coords),
        })
    gdf = gpd.GeoDataFrame(records, crs=crs)
    driver = "GPKG" if out_path.suffix.lower() == ".gpkg" else "ESRI Shapefile"
    gdf.to_file(out_path, driver=driver)
    _LOG.info("Wrote %d routes to %s", len(records), out_path)


def write_cost_raster(
    cost: np.ndarray,
    dem_ds,
    out_path: Path,
) -> None:
    """Write a cumulative-cost (travel time, seconds) raster as GeoTIFF."""
    from osgeo import gdal  # lazy import

    driver = gdal.GetDriverByName("GTiff")
    r, c = cost.shape
    out = driver.Create(
        str(out_path), c, r, 1, gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"],
    )
    out.SetGeoTransform(dem_ds.GetGeoTransform())
    out.SetProjection(dem_ds.GetProjection())
    band = out.GetRasterBand(1)
    band.WriteArray(np.where(np.isfinite(cost), cost, -9999.0).astype(np.float32))
    band.SetNoDataValue(-9999.0)
    out.FlushCache()


# =============================================================================
# Run manifest (for reproducibility)
# =============================================================================

def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(
    manifest_path: Path,
    inputs: dict[str, Path],
    outputs: list[Path],
    parameters: dict,
) -> None:
    """Write a JSON manifest recording everything needed to reproduce the run."""
    # Lazy imports so the math layer doesn't need geopandas/gdal installed.
    import geopandas as gpd
    from osgeo import gdal

    manifest = {
        "script": "walk_lcp.py",
        "script_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "libraries": {
            "numpy": np.__version__,
            "gdal": gdal.__version__,
            "geopandas": gpd.__version__,
        },
        "parameters": parameters,
        "inputs": {
            name: {
                "path": str(p),
                "sha256": sha256_of(p) if p.is_file() else None,
                "bytes": p.stat().st_size if p.is_file() else None,
            }
            for name, p in inputs.items()
        },
        "outputs": [
            {
                "path": str(p),
                "sha256": sha256_of(p) if p.is_file() else None,
                "bytes": p.stat().st_size if p.is_file() else None,
            }
            for p in outputs
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _LOG.info("Wrote run manifest to %s", manifest_path)


# =============================================================================
# Driver
# =============================================================================

def compute_all_lcps(
    dem_path: Path,
    start_path: Path,
    stop_path: Path,
    output_path: Path,
    export_cumulative: bool = False,
    manifest_path: Path | None = None,
) -> int:
    """End-to-end run: read inputs, compute LCPs, write outputs and manifest.

    Returns the number of routes written.
    """
    # Lazy imports — keep the math layer pure-numpy.
    import geopandas as gpd
    from osgeo import gdal

    _LOG.info("Loading DEM: %s", dem_path)
    ds = gdal.Open(str(dem_path))
    if ds is None:
        raise RuntimeError(f"Cannot open DEM: {dem_path}")

    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    dem = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()
    crs = gpd.read_file(start_path).crs   # use start-layer CRS

    _LOG.info("Reading start points: %s", start_path)
    starts = read_points(start_path, gt)
    _LOG.info("  %d start points", len(starts))

    _LOG.info("Reading stop points: %s", stop_path)
    stops = read_points(stop_path, gt)
    _LOG.info("  %d stop points", len(stops))

    routes: list[dict] = []
    extra_outputs: list[Path] = []
    route_id = 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for s_idx, s in enumerate(starts, start=1):
        _LOG.info("Dijkstra from start fid=%d (%d/%d)", s.fid, s_idx, len(starts))
        cost, prev = dijkstra_from_start(dem, gt, (s.row, s.col), nodata)

        if export_cumulative:
            out_cost = output_path.parent / f"cum_walk_time_start_{s.fid}.tif"
            write_cost_raster(cost, ds, out_cost)
            extra_outputs.append(out_cost)
            _LOG.info("  wrote cumulative time raster: %s", out_cost)

        for t in stops:
            time_sec = float(cost[t.row, t.col])
            if not math.isfinite(time_sec):
                _LOG.warning("  stop fid=%d unreachable from start fid=%d", t.fid, s.fid)
                continue
            path_rc, dist_m = reconstruct_path(prev, gt, (s.row, s.col), (t.row, t.col))
            if path_rc is None:
                continue
            routes.append({
                "route_id": route_id,
                "start_id": s.fid,
                "stop_id":  t.fid,
                "path_rc":  path_rc,
                "dist_m":   dist_m,
                "time_s":   time_sec,
            })
            _LOG.info("  route start=%d stop=%d dist=%.1f m time=%.1f min",
                      s.fid, t.fid, dist_m, time_sec / 60.0)
            route_id += 1

    if not routes:
        _LOG.warning("No routes computed (no reachable start-stop pairs).")
        return 0

    write_routes(routes, gt, output_path, crs)

    if manifest_path:
        write_manifest(
            manifest_path,
            inputs={"dem": dem_path, "start": start_path, "stop": stop_path},
            outputs=[output_path, *extra_outputs],
            parameters={
                "export_cumulative": export_cumulative,
                "tobler_function": "v_kmh = 6 * exp(-3.5 * |S + 0.05|)",
                "connectivity": 8,
                "minimum_speed_m_per_s": 0.01,
            },
        )

    return len(routes)


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="walk_lcp",
        description="Least-cost walking paths over a DEM using Tobler's "
                    "hiking function.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dem", required=True, type=Path,
                   help="Path to the input DEM (GeoTIFF).")
    p.add_argument("--start", required=True, type=Path,
                   help="Path to the start points layer (shapefile or GeoPackage).")
    p.add_argument("--stop", required=True, type=Path,
                   help="Path to the stop points layer (shapefile or GeoPackage).")
    p.add_argument("--output", required=True, type=Path,
                   help="Path for the routes output. Use .gpkg (recommended) "
                        "or .shp.")
    p.add_argument("--export-cumulative", action="store_true",
                   help="Also write a cumulative travel-time raster (GeoTIFF) "
                        "per start point, next to --output.")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Write a JSON run manifest (versions, parameters, "
                        "file hashes) to this path.")
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
    n = compute_all_lcps(
        dem_path=args.dem,
        start_path=args.start,
        stop_path=args.stop,
        output_path=args.output,
        export_cumulative=args.export_cumulative,
        manifest_path=args.manifest,
    )
    _LOG.info("Done. %d route(s) written.", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
