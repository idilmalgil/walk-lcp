"""Compare the final routes with a no-mask baseline on the same nodes, the 2 m scale check and the
October 2026 routes; measure the water crossed by every route. Writes route_comparison.csv,
route_comparison_summary.json and scale_check_summary.json."""
# Run from the folder that holds W25_data_archive/ and walk-lcp-main/.
# Results are written to --outdir (default analysis_out/), so the deposit is not overwritten.
import argparse
from pathlib import Path
_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--outdir", type=Path, default=Path("analysis_out"))
OUTDIR = _p.parse_args().outdir
OUTDIR.mkdir(parents=True, exist_ok=True)
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import gdal

A = "W25_data_archive/"
L = A + "04_results/LCP_watermasked/"
new = gpd.read_file(L + "lcp_routes_watermasked.gpkg")
base = gpd.read_file(L + "lcp_routes_nomask_baseline.gpkg")
c2 = gpd.read_file(L + "scale_check_2m/lcp_routes_2m.gpkg")
octo = gpd.read_file(A + "04_results/_superseded/LCP_v2_watermasked_before_node_alignment/lcp_routes_watermasked.gpkg")
old_x = gpd.read_file(A + "04_results/_superseded/LCP_v1_sea_included/lcp_routes_excluded_sea.gpkg", on_invalid="ignore")
m = gdal.Open(A + "02_processed/water_mask_0.4m.tif")
gt = m.GetGeoTransform()
W = m.ReadAsArray() == 1


def water_run_m(geom):
    """Longest continuous stretch of the line over mapped water (sampled every 0.2 m)."""
    if geom.length == 0:
        return 0.0
    s = np.arange(0, geom.length, 0.2)
    xy = np.array([geom.interpolate(d).coords[0] for d in s])
    c = ((xy[:, 0] - gt[0]) / gt[1]).astype(int)
    r = ((xy[:, 1] - gt[3]) / gt[5]).astype(int)
    w = W[np.clip(r, 0, W.shape[0] - 1), np.clip(c, 0, W.shape[1] - 1)]
    best = cur = 0
    for v in w:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best * 0.2


new["water_m"] = new.geometry.apply(water_run_m)
base["water_m"] = base.geometry.apply(water_run_m)
n_i, b_i, c_i, o_i = (g.set_index("route_id") for g in (new, base, c2, octo))
rows = []
for rid, n in n_i.iterrows():
    b, t2, o = b_i.loc[rid], c_i.loc[rid], o_i.loc[rid]
    dm = dh = doct = float("nan")
    if n.geometry.length > 0 and t2.geometry.length > 0:
        pts = [t2.geometry.interpolate(d) for d in np.arange(0, t2.geometry.length, 1.0)]
        dm = float(np.mean([n.geometry.distance(p) for p in pts]))
        dh = float(n.geometry.hausdorff_distance(t2.geometry))
    if n.geometry.length > 0 and o.geometry.length > 0:
        pts = [n.geometry.interpolate(d) for d in np.arange(0, n.geometry.length, 1.0)]
        doct = float(np.mean([o.geometry.distance(p) for p in pts]))
    rows.append(dict(route_id=rid, start_id=n.start_id, stop_id=n.stop_id, start_name=n.start_name, stop_name=n.stop_name,
                     nomask_dist_m=b.dist_m, nomask_time_min=b.time_min, nomask_water_m=b.water_m,
                     masked_dist_m=n.dist_m, masked_time_min=n.time_min, masked_water_m=n.water_m,
                     same_geometry_as_nomask=bool(b.geometry.equals_exact(n.geometry, 1e-6)),
                     scale2m_dist_m=t2.dist_m, scale2m_time_min=t2.time_min, scale2m_mean_offset_m=dm, scale2m_hausdorff_m=dh,
                     october_dist_m=o.dist_m, october_time_min=o.time_min, october_mean_offset_m=doct,
                     crossed_water_october_baseline=bool(rid in set(old_x.route_id))))
d = pd.DataFrame(rows).sort_values("route_id")
d["crosses_water_without_mask"] = d.nomask_water_m > 0
d["d_time_mask_min"] = d.masked_time_min - d.nomask_time_min
d["d_dist_mask_m"] = d.masked_dist_m - d.nomask_dist_m
d["d_time_2m_vs_04m_pct"] = 100 * (d.scale2m_time_min - d.masked_time_min) / d.masked_time_min
d["d_dist_2m_vs_04m_pct"] = 100 * (d.scale2m_dist_m - d.masked_dist_m) / d.masked_dist_m
d["d_time_vs_october_min"] = d.masked_time_min - d.october_time_min
d["d_time_vs_october_pct"] = 100 * d.d_time_vs_october_min / d.october_time_min
d.to_csv(OUTDIR / "route_comparison.csv", index=False)
nd = d[d.masked_dist_m > 0]
x = nd[nd.nomask_water_m > 20]
o = nd[nd.nomask_water_m <= 20]
sc = d[(d.start_name != d.stop_name) & (d.masked_dist_m > 0)]
p = [.05, .25, .5, .75, .95]
oct_ = sc.d_time_vs_october_min
S = dict(n_routes=int(len(d)), n_degenerate_zero_length=int((d.masked_dist_m == 0).sum()), n_non_degenerate=int(len(nd)), max_water_crossed_m_masked=float(d.masked_water_m.max()), n_unreachable=0,
         would_cross_water_without_mask=dict(rule="more than 20 m of continuous mapped water, as in the first submission", n=int(len(x)), water_m_min=float(x.nomask_water_m.min()), water_m_max=float(x.nomask_water_m.max()), largest_other_route_water_m=float(o.nomask_water_m.max()),
                                             d_time_min=x.d_time_mask_min.describe().round(2).to_dict(), d_dist_m=x.d_dist_mask_m.describe().round(1).to_dict(), pct_time_increase_median=float((100 * x.d_time_mask_min / x.nomask_time_min).median())),
         other_routes=dict(n=int(len(o)), identical_geometry=int(o.same_geometry_as_nomask.sum()), max_abs_d_time_min=float(o.d_time_mask_min.abs().max()), max_abs_d_dist_m=float(o.d_dist_mask_m.abs().max())),
         change_from_october_routes=dict(n=int(len(sc)), d_time_min=oct_.describe(percentiles=p).round(2).to_dict(), median_abs_d_time_min=float(oct_.abs().median()), median_abs_d_time_pct=float(sc.d_time_vs_october_pct.abs().median()), identical_time=int((oct_.abs() < 1e-6).sum()), october_mean_offset_m=sc.october_mean_offset_m.describe(percentiles=p).round(2).to_dict()))
json.dump(S, open(OUTDIR / "route_comparison_summary.json", "w"), indent=1)
s2 = {k: sc[k].describe(percentiles=p).round(2).to_dict() for k in ("d_time_2m_vs_04m_pct", "d_dist_2m_vs_04m_pct", "scale2m_mean_offset_m", "scale2m_hausdorff_m")}
s2["n"] = int(len(sc))
s2["pct_routes_mean_offset_le_5m"] = float((sc.scale2m_mean_offset_m <= 5).mean() * 100)
s2["pct_routes_abs_dtime_le_3pct"] = float((sc.d_time_2m_vs_04m_pct.abs() <= 3).mean() * 100)
json.dump(s2, open(OUTDIR / "scale_check_summary.json", "w"), indent=1)
print(json.dumps(S, indent=1)[:600])
