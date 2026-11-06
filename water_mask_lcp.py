"""
Water mask on the LCP grid  (ortho RGB + LCP DEM + DSM)  ->  mask / impassable DEM / polygons
Runs in the QGIS Python Console (Show Editor > Run) or standalone:
    "C:\\Program Files\\QGIS 3.44.9\\bin\\python-qgis-ltr.bat" water_mask_lcp.py

Every raster output is snapped cell-for-cell to the LCP cost surface
("dem 04 Low Pass Filter.tif": 5106 x 4656 cells, 0.3999804 x 0.4000215 m,
EPSG:5253, NoData -99999), so it can be used directly with walk_lcp.py.

Decimation onto that grid:
    ortho  3.38 cm -> 0.40 m  (~12x)  average     clean colour, no glint/ripples
    DSM    6.76 cm -> 0.40 m  (~6x)   q3          keeps roofs/trees, ignores spikes
    DTM    = the LCP DEM itself (no resampling), so the mask matches the cost surface

A cell is water when ALL of these are true (DEM rules = manuscript criterion,
04_results/_superseded/LCP_v1_sea_included/README_LCP.txt section 4):
    1. Height : LCP DEM < ELEV_MAX                       (4 m)
    2. Relief : max - min of LCP DEM over RELIEF_WIN_M < RELIEF_MAX   (6 m, 0.5 m)
    3. nDSM   : DSM - DEM < NDSM_MAX                     (no roofs/trees/boats)
    4. Colour : (Blue - Red) / (Blue + Red) > BLUE_MIN   (optional, USE_COLOR),
                not required below SEA_LEVEL_MAX (shallow green/sandy water)
    5. Valid  : DEM has data and ortho alpha > 0
then majority smoothing -> sieve (keeps only large connected areas).

Outputs (OUT_DIR):
    water_mask_lcpgrid.tif        1 = water, 0 = land, LCP grid, Byte
                                  (deposited as 02_processed/water_mask_0.4m.tif)
    dem_lcp_water_nodata.tif      LCP DEM with water set to -99999 (water impassable)
                                  (deposited as 02_processed/dem 04 Low Pass Filter_water_nodata.tif)
    water_polygons.gpkg / water_outlines.gpkg
    diag_*.tif                    blueness / relief / ndsm for threshold tuning
"""

import os
import numpy as np
from osgeo import gdal, ogr
from scipy import ndimage

# ======================= SETTINGS (edit these) =======================
# Paths, relative to the folder that holds W25_data_archive/ and walk-lcp-main/: the
# full-resolution DJI Terra orthophoto (3.38 cm) and DSM (6.76 cm) are in 01_raw/.
ARCHIVE    = r"W25_data_archive"
LCP_DEM    = os.path.join(ARCHIVE, "02_processed", "dem 04 Low Pass Filter.tif")
NODES      = [os.path.join(ARCHIVE, "03_gis_inputs", "nodes_start.gpkg"),
              os.path.join(ARCHIVE, "03_gis_inputs", "nodes_stop.gpkg")]
ORTHO_PATH = os.path.join(ARCHIVE, "01_raw", "dom.tif")
DSM_PATH   = os.path.join(ARCHIVE, "01_raw", "dsm.tif")
OUT_DIR    = r"water_out"

ELEV_MAX       = 4.0    # m, manuscript criterion
RELIEF_WIN_M   = 6.0    # m, manuscript criterion (window for max - min)
RELIEF_MAX     = 0.5    # m, manuscript criterion
NDSM_MAX       = 0.8    # m above ground; roofs/trees/boats are higher
USE_COLOR      = True   # also require blue-ish ortho colour
BLUE_MIN       = 0.0    # raise if beach/fields are detected, lower if water is missed
SEA_LEVEL_MAX  = 0.3    # m; flat cells below this are water whatever their colour
SMOOTH_WIN_M   = 2.0    # majority filter window in metres (0 = off)
MIN_AREA_M2    = 500    # water areas and land holes smaller than this are removed
SIMPLIFY_TOL_M = 0.4    # outline simplification (= 1 cell)
LOAD_DIAGNOSTICS = True
# =====================================================================

gdal.UseExceptions()
os.makedirs(OUT_DIR, exist_ok=True)
P = lambda name: os.path.join(OUT_DIR, name)


def odd_px(metres, cell):
    """Window size in pixels for a distance in metres, forced to an odd number."""
    return max(1, int(round(metres / cell)) | 1)


# ---------- 1. Read the LCP grid ----------
print("1/7 Reading LCP grid...")
d = gdal.Open(LCP_DEM)
geo, proj = d.GetGeoTransform(), d.GetProjection()
W, H = d.RasterXSize, d.RasterYSize
band = d.GetRasterBand(1)
DEM_NODATA = band.GetNoDataValue()
dem = band.ReadAsArray().astype(np.float32)
d = None
dem_void = (dem == DEM_NODATA) | np.isnan(dem)
dem[dem_void] = np.nan
cell = (abs(geo[1]) + abs(geo[5])) / 2.0
box = (geo[0], geo[3] + geo[5] * H, geo[0] + geo[1] * W, geo[3])   # minx, miny, maxx, maxy
print(f"    {W} x {H} cells, {geo[1]:.7f} x {abs(geo[5]):.7f} m, NoData {DEM_NODATA}")

# ---------- 2. Decimate ortho and DSM onto exactly that grid ----------
print("2/7 Decimating ortho (average) and DSM (q3) onto the LCP grid...")
ortho_w, dsm_w = P("_ortho_lcpgrid.tif"), P("_dsm_lcpgrid.tif")
grid = dict(format="GTiff", outputBounds=box, width=W, height=H, dstSRS=proj,
            multithread=True, warpMemoryLimit=2048,
            creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"])
if not os.path.exists(ortho_w):
    gdal.Warp(ortho_w, ORTHO_PATH, options=gdal.WarpOptions(
        resampleAlg="average", dstAlpha=True, **grid))
if not os.path.exists(dsm_w):
    gdal.Warp(dsm_w, DSM_PATH, options=gdal.WarpOptions(
        resampleAlg="q3", dstNodata=-9999, **grid))

o = gdal.Open(ortho_w)
R = o.GetRasterBand(1).ReadAsArray().astype(np.float32)
Bl = o.GetRasterBand(3).ReadAsArray().astype(np.float32)
valid = o.GetRasterBand(o.RasterCount).ReadAsArray() > 0
o = None

s = gdal.Open(dsm_w)
dsm = s.GetRasterBand(1).ReadAsArray().astype(np.float32)
s = None
dsm[dsm == -9999] = np.nan

# ---------- 3. Cues ----------
print("3/7 Computing relief, nDSM, blueness...")
k = odd_px(RELIEF_WIN_M, cell)
hi = ndimage.maximum_filter(np.where(dem_void, -np.inf, dem), size=k, mode="nearest")
lo = ndimage.minimum_filter(np.where(dem_void, np.inf, dem), size=k, mode="nearest")
relief = np.where(dem_void, np.nan, hi - lo).astype(np.float32)
del hi, lo
ndsm = dsm - dem
blueness = (Bl - R) / (Bl + R + 1.0)
del R, Bl

with np.errstate(invalid="ignore"):
    water = (~dem_void) & valid & (dem < ELEV_MAX) & (relief < RELIEF_MAX)
    water &= (ndsm < NDSM_MAX) | np.isnan(ndsm)          # DSM void over water passes
    if USE_COLOR:
        # shallow water over sand is green/brown, so colour is only required above sea level
        water &= (blueness > BLUE_MIN) | (dem < SEA_LEVEL_MAX)

# ---------- 4. Majority smoothing ----------
print("4/7 Smoothing...")
m = odd_px(SMOOTH_WIN_M, cell) if SMOOTH_WIN_M > 0 else 1
if m > 1:
    water = ndimage.uniform_filter(water.astype(np.float32), size=m, mode="nearest") > 0.5
water &= ~dem_void


def write(arr, path, dtype, nodata=None):
    ds = gdal.GetDriverByName("GTiff").Create(
        path, W, H, 1, dtype, options=["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform(geo)
    ds.SetProjection(proj)
    b = ds.GetRasterBand(1)
    if nodata is not None:
        b.SetNoDataValue(nodata)
    b.WriteArray(arr)
    ds.FlushCache()
    ds = None


# ---------- 5. Sieve: keep only large connected water, fill small holes ----------
print("5/7 Sieving...")
mask_path = P("water_mask_lcpgrid.tif")
write(water.astype(np.uint8), mask_path, gdal.GDT_Byte)
sieve_px = max(1, int(MIN_AREA_M2 / (abs(geo[1]) * abs(geo[5]))))
ds = gdal.Open(mask_path, gdal.GA_Update)
b = ds.GetRasterBand(1)
gdal.SieveFilter(b, None, b, sieve_px, 8)
water = (b.ReadAsArray() == 1) & ~dem_void
b.WriteArray(water.astype(np.uint8))
ds.FlushCache()
ds = None

n_valid = int((~dem_void).sum())
print(f"    Water: {int(water.sum()):,} cells = {100.0 * water.sum() / n_valid:.1f}% of valid DEM "
      f"(manuscript criterion: 1,628,056 cells, 11.0%)")

# Cost surface with water impassable (same grid, dtype and NoData as the LCP DEM)
dem_out = np.where(dem_void | water, DEM_NODATA, dem).astype(np.float32)
dem_out_path = P("dem_lcp_water_nodata.tif")
write(dem_out, dem_out_path, gdal.GDT_Float32, DEM_NODATA)
del dem_out

if LOAD_DIAGNOSTICS:
    write(blueness.astype(np.float32), P("diag_blueness.tif"), gdal.GDT_Float32)
    write(np.nan_to_num(relief, nan=-9999), P("diag_relief.tif"), gdal.GDT_Float32, -9999)
    write(np.nan_to_num(ndsm, nan=-9999).astype(np.float32), P("diag_ndsm.tif"),
          gdal.GDT_Float32, -9999)

# ---------- 6. Check LCP nodes against the mask ----------
print("6/7 Checking LCP nodes...")
walk = ~(dem_void | water)
labels, _ = ndimage.label(walk, structure=np.ones((3, 3)))     # 8-connected, as walk_lcp
comps = set()
for path in NODES:
    src = ogr.Open(path)
    lyr = src.GetLayer(0)
    for f in lyr:
        g = f.GetGeometryRef()
        col = int((g.GetX() - geo[0]) / geo[1])
        row = int((g.GetY() - geo[3]) / geo[5])
        name = f.GetField("location") if f.GetFieldIndex("location") >= 0 else f.GetFID()
        if not (0 <= row < H and 0 <= col < W) or not walk[row, col]:
            print(f"    WARNING: node '{name}' ({os.path.basename(path)}) is in water/NoData")
        else:
            comps.add(labels[row, col])
    src = None
print("    All nodes on one walkable component." if len(comps) == 1 else
      f"    WARNING: nodes fall on {len(comps)} separate walkable components "
      "- some routes will be unreachable with water impassable.")
del labels

# ---------- 7. Polygons + outlines ----------
print("7/7 Polygonizing...")
poly_raw = P("_water_polygons_raw.gpkg")
drv = ogr.GetDriverByName("GPKG")
for f in (poly_raw,):
    if os.path.exists(f):
        drv.DeleteDataSource(f)
ds = gdal.Open(mask_path)
b = ds.GetRasterBand(1)
vds = drv.CreateDataSource(poly_raw)
srs = ds.GetSpatialRef()
lyr = vds.CreateLayer("water", srs=srs, geom_type=ogr.wkbPolygon)
lyr.CreateField(ogr.FieldDefn("DN", ogr.OFTInteger))
gdal.Polygonize(b, b, lyr, 0, ["8CONNECTED=8"])   # band as its own mask -> only DN=1
vds = None
ds = None

poly_path, line_path = P("water_polygons.gpkg"), P("water_outlines.gpkg")
try:
    import processing
    from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer
    in_qgis = True
except ImportError:
    in_qgis = False

if in_qgis:
    fixed = processing.run("native:fixgeometries", {
        "INPUT": poly_raw, "OUTPUT": "TEMPORARY_OUTPUT"})["OUTPUT"]
    simp = processing.run("native:simplifygeometries", {
        "INPUT": fixed, "METHOD": 0, "TOLERANCE": SIMPLIFY_TOL_M,
        "OUTPUT": "TEMPORARY_OUTPUT"})["OUTPUT"]
    processing.run("native:smoothgeometry", {
        "INPUT": simp, "ITERATIONS": 3, "OFFSET": 0.25, "MAX_ANGLE": 180,
        "OUTPUT": poly_path})
    processing.run("native:polygonstolines", {"INPUT": poly_path, "OUTPUT": line_path})

    prj = QgsProject.instance()
    if LOAD_DIAGNOSTICS:
        for name in ("diag_blueness", "diag_relief", "diag_ndsm"):
            prj.addMapLayer(QgsRasterLayer(P(name + ".tif"), name))
    prj.addMapLayer(QgsRasterLayer(mask_path, "water_mask_lcpgrid"))
    prj.addMapLayer(QgsRasterLayer(dem_out_path, "dem_lcp_water_nodata"))
    prj.addMapLayer(QgsVectorLayer(poly_path, "water_polygons", "ogr"))
    prj.addMapLayer(QgsVectorLayer(line_path, "water_outlines", "ogr"))
else:
    # Standalone: simplify only (smoothing needs QGIS processing)
    for f in (poly_path, line_path):
        if os.path.exists(f):
            drv.DeleteDataSource(f)
    src = ogr.Open(poly_raw)
    out = drv.CreateDataSource(poly_path)
    lin = drv.CreateDataSource(line_path)
    pl = out.CreateLayer("water_polygons", srs=src.GetLayer(0).GetSpatialRef(),
                         geom_type=ogr.wkbMultiPolygon)
    ll = lin.CreateLayer("water_outlines", srs=src.GetLayer(0).GetSpatialRef(),
                         geom_type=ogr.wkbMultiLineString)
    for f in src.GetLayer(0):
        g = f.GetGeometryRef().MakeValid().SimplifyPreserveTopology(SIMPLIFY_TOL_M)
        nf = ogr.Feature(pl.GetLayerDefn())
        nf.SetGeometry(ogr.ForceToMultiPolygon(g))
        pl.CreateFeature(nf)
        lf = ogr.Feature(ll.GetLayerDefn())
        lf.SetGeometry(ogr.ForceToMultiLineString(g.GetBoundary()))
        ll.CreateFeature(lf)
    out = lin = src = None

print(f"Done -> {OUT_DIR}")
