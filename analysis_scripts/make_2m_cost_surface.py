"""2 m version of the water-masked LCP cost surface, for the scale check.
Block mean over 5 x 5 cells of the 0.4 m surface; a 2 m cell is walkable only
when the majority of its 25 fine cells are walkable (land with data)."""
# Run from the folder that holds W25_data_archive/ and walk-lcp-main/.
# Results are written to --outdir (default analysis_out/), so the deposit is not overwritten.
import argparse
from pathlib import Path
_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--outdir", type=Path, default=Path("analysis_out"))
OUTDIR = _p.parse_args().outdir
OUTDIR.mkdir(parents=True, exist_ok=True)
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
src = "W25_data_archive/02_processed/dem 04 Low Pass Filter_water_nodata.tif"
dst = str(OUTDIR / "dem_water_nodata_2m.tif")
ds = gdal.Open(src); b = ds.GetRasterBand(1); nd = b.GetNoDataValue()
z = b.ReadAsArray().astype(np.float64); ok = z != nd
f = 5; R, C = (z.shape[0] // f) * f, (z.shape[1] // f) * f
zz = np.where(ok, z, 0.0)[:R, :C].reshape(R // f, f, C // f, f).sum(axis=(1, 3))
nn = ok[:R, :C].reshape(R // f, f, C // f, f).sum(axis=(1, 3))
out = np.where(nn > f * f / 2, zz / np.maximum(nn, 1), nd).astype(np.float32)
gt = list(ds.GetGeoTransform()); gt[1] *= f; gt[5] *= f
o = gdal.GetDriverByName("GTiff").Create(dst, out.shape[1], out.shape[0], 1, gdal.GDT_Float32, ["COMPRESS=DEFLATE"])
o.SetGeoTransform(gt); o.SetProjection(ds.GetProjection()); o.GetRasterBand(1).SetNoDataValue(nd)
o.GetRasterBand(1).WriteArray(out); o = None
print(out.shape, gt[1], gt[5], "walkable cells", int((out != nd).sum()))
