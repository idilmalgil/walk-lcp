"""Smallest extra height that gives a clear sight line, for every tested pair (0.4 m DEM)."""
# Run from the folder that holds W25_data_archive/ and walk-lcp-main/.
# Results are written to --outdir (default analysis_out/), so the deposit is not overwritten.
import argparse
from pathlib import Path
_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--outdir", type=Path, default=Path("analysis_out"))
OUTDIR = _p.parse_args().outdir
OUTDIR.mkdir(parents=True, exist_ok=True)
import json
import sys

import numpy as np

sys.path.insert(0, "walk-lcp-main")
from viewshed_null import line_of_sight_profile, read_raster  # noqa: E402

dem_r = read_raster("W25_data_archive/02_processed/dem 04 Low Pass Filter.tif")
dem = np.where(dem_r.array == dem_r.nodata, np.nan, dem_r.array).astype(np.float64)
cfg = json.load(open("walk-lcp-main/viewshed_foci_parion.json", encoding="utf-8"))
xy = {c["name"]: (c["x"], c["y"]) for c in cfg["foci"]}


def needed(a, b, which):
    """Bisection on the extra height at end `which` ('a' or 'b'), other end at 1.7 m."""
    lo, hi = 0.0, 200.0
    for _ in range(40):
        mid = (lo + hi) / 2
        ha, hb = (1.7 + mid, 1.7) if which == "a" else (1.7, 1.7 + mid)
        if line_of_sight_profile(dem, dem_r.gt, xy[a], xy[b], ha, hb)["visible"]:
            hi = mid
        else:
            lo = mid
    return hi


out = {}
for p in cfg["pairs"]:
    a, b = p["a"], p["b"]
    out[f"{a} x {b}"] = {f"extra_height_at_{a}_m": round(needed(a, b, "a"), 2),
                         f"extra_height_at_{b}_m": round(needed(a, b, "b"), 2)}
    print(f"{a} x {b}", out[f"{a} x {b}"])
json.dump(out, open(OUTDIR / "FINAL_line_of_sight_required_height.json", "w"), indent=1)
