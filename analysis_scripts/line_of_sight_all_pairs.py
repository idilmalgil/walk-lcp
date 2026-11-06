"""Direct line of sight for all seven pairs on the 0.4 m DEM (exact profile test)."""
# Run from the folder that holds W25_data_archive/ and walk-lcp-main/.
# Results are written to --outdir (default analysis_out/), so the deposit is not overwritten.
import argparse
from pathlib import Path
_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--outdir", type=Path, default=Path("analysis_out"))
OUTDIR = _p.parse_args().outdir
OUTDIR.mkdir(parents=True, exist_ok=True)
import csv
import json
import sys

import numpy as np

sys.path.insert(0, "walk-lcp-main")
from viewshed_null import line_of_sight_profile, read_raster  # noqa: E402

dem_r = read_raster("W25_data_archive/02_processed/dem 04 Low Pass Filter.tif")
dem = np.where(dem_r.array == dem_r.nodata, np.nan, dem_r.array).astype(np.float64)
cfg = json.load(open("walk-lcp-main/viewshed_foci_parion.json", encoding="utf-8"))
xy = {c["name"]: (c["x"], c["y"]) for c in cfg["foci"]}
out = []
for p in cfg["pairs"]:
    a, b = p["a"], p["b"]
    for src, dst in ((a, b), (b, a)):
        for th in (1.7, 0.0):
            r = line_of_sight_profile(dem, dem_r.gt, xy[src], xy[dst], 1.7, th)
            out.append({"pair": f"{a} x {b}", "from": src, "to": dst, "grid": "0.4 m",
                        "observer_h_m": 1.7, "target_h_m": th, **r})
            print(out[-1])
path = OUTDIR / "FINAL_line_of_sight_0.4m_all_pairs.csv"
with open(path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
