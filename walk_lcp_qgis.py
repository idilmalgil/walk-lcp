# -*- coding: utf-8 -*-
"""
walk_lcp_qgis.py — Least-cost walking paths over a DEM with Tobler's hiking
function, runnable inside the QGIS Python console / Script Editor.

This is the QGIS-bound twin of ``walk_lcp.py``. The mathematics are identical
(Dijkstra on an 8-connected DEM grid, Tobler's hiking function for edge
cost); the only differences are the I/O backends (qgis.core for vector
layers) and the entry point (no CLI; the parameters are filled in at the
top of the file and the script is executed via the QGIS Python console).

Reproducibility note
--------------------
For journal verification we recommend running the standalone version
(``walk_lcp.py``) since it does not require a QGIS install. This file is
provided for users who already work in QGIS and want a one-click run.

Author      : [FILL: your name]
License     : MIT (see LICENSE)
"""

import os
import math
import heapq
import numpy as np

from osgeo import gdal
from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsVectorFileWriter,
    QgsFields,
    QgsField,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant


# =============================================================================
# USER PARAMETERS — EDIT THESE BEFORE RUNNING
# =============================================================================

DEM_PATH   = r"[FILL absolute path to DEM .tif]"
START_SHP  = r"[FILL absolute path to start points .shp or .gpkg]"
STOP_SHP   = r"[FILL absolute path to stop points .shp or .gpkg]"
OUTPUT_DIR = r"[FILL absolute output folder]"

# Optionally export a cumulative travel-time raster per start point.
# Can produce many files for surveys with many starts.
EXPORT_CUM_RASTERS = False


# =============================================================================
# Core algorithm (identical to walk_lcp.py)
# =============================================================================

def tobler_speed(gradient):
    """Walking speed (m/s) from Tobler's (1993) hiking function."""
    v_kmh = 6.0 * math.exp(-3.5 * abs(gradient + 0.05))
    return max(v_kmh / 3.6, 0.01)


def world_to_pixel(gt, x, y):
    origin_x, px_w, _, origin_y, _, px_h = gt
    col = int((x - origin_x) / px_w)
    row = int((origin_y - y) / abs(px_h))
    return row, col


def pixel_to_world(gt, row, col):
    origin_x, px_w, _, origin_y, _, px_h = gt
    x = origin_x + (col + 0.5) * px_w
    y = origin_y - (row + 0.5) * abs(px_h)
    return x, y


def dijkstra_from_start(dem, gt, start_rc, nodata):
    nrows, ncols = dem.shape
    sr, sc = start_rc

    cost = np.full((nrows, ncols), np.inf)
    prev = np.full((nrows, ncols, 2), -1, dtype=np.int32)
    pq = []

    cost[sr, sc] = 0.0
    heapq.heappush(pq, (0.0, sr, sc))

    px_w = gt[1]
    px_h = abs(gt[5])

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

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


def reconstruct_path(prev, gt, start_rc, stop_rc):
    sr, sc = start_rc
    tr, tc = stop_rc
    r, c0 = tr, tc

    if prev[r, c0, 0] == -1 and (r, c0) != (sr, sc):
        return None, None

    px_w = gt[1]
    px_h = abs(gt[5])

    path = []
    total_dist = 0.0
    while True:
        path.append((r, c0))
        if (r, c0) == (sr, sc):
            break
        pr, pc = prev[r, c0]
        dx = (c0 - pc) * px_w
        dy = (r - pr) * px_h
        total_dist += math.hypot(dx, dy)
        r, c0 = pr, pc
    path.reverse()
    return path, total_dist


# =============================================================================
# QGIS-specific I/O
# =============================================================================

def read_points_with_rc(shp_path, gt):
    vl = QgsVectorLayer(shp_path, "pts", "ogr")
    if not vl.isValid():
        raise RuntimeError("Cannot open layer: " + shp_path)
    pts = []
    for f in vl.getFeatures():
        p = f.geometry().asPoint()
        x, y = p.x(), p.y()
        r, c = world_to_pixel(gt, x, y)
        pts.append({"fid": f.id(), "x": x, "y": y, "row": r, "col": c})
    if not pts:
        raise RuntimeError("Layer has no points: " + shp_path)
    return pts


def write_all_routes_shp(routes, gt, out_path):
    fields = QgsFields()
    fields.append(QgsField("route_id", QVariant.Int))
    fields.append(QgsField("start_id", QVariant.Int))
    fields.append(QgsField("stop_id",  QVariant.Int))
    fields.append(QgsField("dist_m",   QVariant.Double))
    fields.append(QgsField("time_s",   QVariant.Double))
    fields.append(QgsField("time_min", QVariant.Double))
    fields.append(QgsField("time_hr",  QVariant.Double))

    # Remove pre-existing sidecars
    if os.path.exists(out_path):
        for ext in ["", ".dbf", ".prj", ".shx", ".cpg"]:
            f = out_path.replace(".shp", ext)
            if os.path.exists(f):
                os.remove(f)

    writer = QgsVectorFileWriter(
        out_path, "UTF-8", fields,
        QgsWkbTypes.LineString,
        QgsProject.instance().crs(),
        "ESRI Shapefile",
    )

    for rinfo in routes:
        path_rc = rinfo["path_rc"]
        if not path_rc:
            continue
        pts = [QgsPointXY(*pixel_to_world(gt, rr, cc)) for (rr, cc) in path_rc]
        feat = QgsFeature()
        feat.setFields(fields)
        feat["route_id"] = int(rinfo["route_id"])
        feat["start_id"] = int(rinfo["start_id"])
        feat["stop_id"]  = int(rinfo["stop_id"])
        feat["dist_m"]   = float(rinfo["dist_m"])
        feat["time_s"]   = float(rinfo["time_s"])
        feat["time_min"] = float(rinfo["time_s"] / 60.0)
        feat["time_hr"]  = float(rinfo["time_s"] / 3600.0)
        feat.setGeometry(QgsGeometry.fromPolylineXY(pts))
        writer.addFeature(feat)
    del writer

    vl = QgsVectorLayer(out_path, "LCP_all_routes", "ogr")
    if vl.isValid():
        QgsProject.instance().addMapLayer(vl)


def write_cost_raster(cost, dem_ds, out_path):
    driver = gdal.GetDriverByName("GTiff")
    r, c = cost.shape
    out = driver.Create(out_path, c, r, 1, gdal.GDT_Float32)
    out.SetGeoTransform(dem_ds.GetGeoTransform())
    out.SetProjection(dem_ds.GetProjection())
    band = out.GetRasterBand(1)
    band.WriteArray(cost)
    band.SetNoDataValue(-9999)
    out.FlushCache()


# =============================================================================
# Main
# =============================================================================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("[INFO] Loading DEM…")
    ds = gdal.Open(DEM_PATH)
    if ds is None:
        raise RuntimeError("Cannot open DEM: " + DEM_PATH)

    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    dem = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()

    print("[INFO] Reading start points…")
    starts = read_points_with_rc(START_SHP, gt)
    print("  Found {} start points".format(len(starts)))

    print("[INFO] Reading stop points…")
    stops = read_points_with_rc(STOP_SHP, gt)
    print("  Found {} stop points".format(len(stops)))

    routes = []
    route_id = 1

    for s_idx, s in enumerate(starts, start=1):
        start_rc = (s["row"], s["col"])
        print("\n[INFO] Dijkstra from start {} (#{}/{})".format(
            s["fid"], s_idx, len(starts)))
        cost, prev = dijkstra_from_start(dem, gt, start_rc, nodata)

        if EXPORT_CUM_RASTERS:
            out_cost = os.path.join(
                OUTPUT_DIR, "cum_walk_time_start_{}.tif".format(s["fid"]))
            print("  [INFO] Writing cumulative time raster -> " + out_cost)
            write_cost_raster(cost, ds, out_cost)

        for t in stops:
            stop_rc = (t["row"], t["col"])
            time_sec = float(cost[stop_rc[0], stop_rc[1]])
            if not math.isfinite(time_sec):
                print("  [WARN] Stop {} unreachable from start {} (skip)".format(
                    t["fid"], s["fid"]))
                continue
            path_rc, dist_m = reconstruct_path(prev, gt, start_rc, stop_rc)
            if path_rc is None:
                continue
            print("  [ROUTE] start={} stop={} dist={:.2f} m time={:.2f} min".format(
                s["fid"], t["fid"], dist_m, time_sec / 60.0))
            routes.append({
                "route_id": route_id,
                "start_id": s["fid"],
                "stop_id":  t["fid"],
                "path_rc":  path_rc,
                "dist_m":   dist_m,
                "time_s":   time_sec,
            })
            route_id += 1

    if not routes:
        print("[WARN] No routes computed.")
        return

    out_lcp = os.path.join(OUTPUT_DIR, "lcp_all_routes.shp")
    print("\n[INFO] Writing all routes -> " + out_lcp)
    write_all_routes_shp(routes, gt, out_lcp)

    print("\n========== SUMMARY ==========")
    print("Computed {} routes for {} start(s) x {} stop(s).".format(
        len(routes), len(starts), len(stops)))
    print("Shapefile:", out_lcp)
    print("================================\n")


main()
