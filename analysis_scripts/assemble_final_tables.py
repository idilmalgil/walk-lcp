"""Assemble the final, single results tables for Table 1, Table S5, Table S3 and the
resolution check from the land-masked runs."""
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
import os

A = "W25_data_archive/04_results"
N = A + "/Null-model"
R = {"land2": N + "/landmask_2m", "sea2": N + "/seaincluded_2m_reference",
     "land04": N + "/landmask_0.4m_fragile"}
OLD = A + "/_superseded/Null-model_v1_sea_included"


def rows(d):
    return list(csv.DictReader(open(os.path.join(d, "results_table.csv"), encoding="utf-8")))


def stat(rs, pair, s):
    return next(r for r in rs if r["pair"] == pair and r["statistic"] == s)


def los(d):
    return list(csv.DictReader(open(os.path.join(d, "line_of_sight.csv"), encoding="utf-8")))


def f(r, k):
    return float(r[k])


def fmt(x, n=3):
    return f"{x:.{n}f}"


pub = {r["A"] + " x " + r["B"]: r for r in
       csv.DictReader(open(OLD + "/a5_results_table.csv", encoding="utf-8"))}
pub_u = {r["A"] + " x " + r["B"]: r for r in
         json.load(open(OLD + "/Union test/union_test_results.json", encoding="utf-8"))}
L2, S2 = rows(R["land2"]), rows(R["sea2"])
J2 = json.load(open(os.path.join(R["land2"], "results.json"), encoding="utf-8"))
fa = {x["name"]: x for x in J2["foci"]}
LOS2 = los(R["land2"])
pairs = list(dict.fromkeys(r["pair"] for r in L2))


def los_pair(LS, p, a, b):
    ls = [x for x in LS if x["pair"] == p and float(x["target_h_m"]) == 1.7]
    return next(x for x in ls if x["from"] == a), next(x for x in ls if x["from"] == b)


# ---------------- Table 1 ----------------
t1 = []
for p in pairs:
    a, b = p.split(" x ")
    o, j, h, u = (stat(L2, p, s) for s in ("overlap_pct_smaller", "jaccard", "overlap_ha", "union_ha"))
    ab, ba = los_pair(LOS2, p, a, b)
    pp = pub.get(p, {})
    t1.append({
        "pair": p, "separation_m": fmt(f(o, "separation_m"), 1),
        "z_a_m": fmt(f(o, "z_a"), 2), "z_b_m": fmt(f(o, "z_b"), 2),
        "visible_land_a_ha": fmt(fa[a]["visible_land_ha"], 2),
        "visible_land_b_ha": fmt(fa[b]["visible_land_ha"], 2),
        "shared_land_ha": fmt(f(h, "observed"), 2), "union_land_ha": fmt(f(u, "observed"), 2),
        "overlap_pct_of_smaller": fmt(f(o, "observed"), 1), "jaccard": fmt(f(j, "observed"), 3),
        "LOS_a_to_b_1.7m": "yes" if ab["profile_visible"] == "True" else "no",
        "LOS_b_to_a_1.7m": "yes" if ba["profile_visible"] == "True" else "no",
        "LOS_min_clearance_m": fmt(min(f(ab, "profile_clearance_m"), f(ba, "profile_clearance_m")), 2),
        "LOS_grass_agrees": str(ab.get("grass_r_viewshed_visible") == ab["profile_visible"]
                                and ba.get("grass_r_viewshed_visible") == ba["profile_visible"]),
        "pct_null_median": fmt(f(o, "null_median"), 1), "pct_null_q05": fmt(f(o, "null_q05"), 1),
        "pct_null_q95": fmt(f(o, "null_q95"), 1),
        "pct_n_null_ge": o["n_null_ge_observed"], "pct_n_null_le": o["n_null_le_observed"],
        "pct_p_two_sided": fmt(f(o, "p_two_sided")), "pct_p_holm": fmt(f(o, "p_holm_two_sided")),
        "jaccard_null_median": fmt(f(j, "null_median")), "jaccard_null_q95": fmt(f(j, "null_q95")),
        "jaccard_n_null_ge": j["n_null_ge_observed"],
        "jaccard_p_two_sided": fmt(f(j, "p_two_sided")), "jaccard_p_holm": fmt(f(j, "p_holm_two_sided")),
        "shared_ha_n_null_ge": h["n_null_ge_observed"],
        "shared_ha_p_two_sided": fmt(f(h, "p_two_sided")), "shared_ha_p_holm": fmt(f(h, "p_holm_two_sided")),
        "n_matched_available": o["n_matched_available"], "n_draws": o["n_null"],
        "published_pct_0.4m_sea_included": pp.get("published_pct_04m", ""),
        "A5_pct_2m_sea_included": pp.get("obs_pct_of_smaller_2m", ""),
        "A5_p_one_sided": pp.get("p_value", ""), "A5_tail": pp.get("tail", ""),
        "v2_sea_included_pct": fmt(f(stat(S2, p, "overlap_pct_smaller"), "observed"), 1),
        "v2_sea_included_p_two_sided": fmt(f(stat(S2, p, "overlap_pct_smaller"), "p_two_sided")),
    })

# ---------------- Table S5 ----------------
s5 = []
for p in pairs:
    u = stat(L2, p, "union_ha")
    us = stat(S2, p, "union_ha")
    pu = pub_u.get(p, {})
    s5.append({
        "pair": p, "union_land_ha": fmt(f(u, "observed"), 2),
        "null_median_ha": fmt(f(u, "null_median"), 2), "null_q05_ha": fmt(f(u, "null_q05"), 2),
        "null_q95_ha": fmt(f(u, "null_q95"), 2), "null_max_ha": fmt(f(u, "null_max"), 2),
        "margin_above_null_max_ha": fmt(f(u, "observed") - f(u, "null_max"), 2),
        "n_null_ge": u["n_null_ge_observed"], "n_null_le": u["n_null_le_observed"],
        "beyond_all_draws": u["beyond_all_draws"],
        "p_upper": fmt(f(u, "p_upper")), "p_lower": fmt(f(u, "p_lower")),
        "p_two_sided": fmt(f(u, "p_two_sided")), "p_holm": fmt(f(u, "p_holm_two_sided")),
        "published_union_ha_2m_sea_included": pu.get("obs_union_ha", ""),
        "published_p_upper": pu.get("p_upper", ""),
        "v2_sea_included_union_ha": fmt(f(us, "observed"), 2),
        "v2_sea_included_p_two_sided": fmt(f(us, "p_two_sided")),
    })

# ---------------- Resolution check ----------------
res = []
if os.path.exists(os.path.join(R["land04"], "results_table.csv")):
    L4 = rows(R["land04"])
    J4 = json.load(open(os.path.join(R["land04"], "results.json"), encoding="utf-8"))
    fa4 = {x["name"]: x for x in J4["foci"]}
    LOS4 = los(R["land04"])
    for p in dict.fromkeys(r["pair"] for r in L4):
        a, b = p.split(" x ")
        for grid, LL, FF, LS, JJ in (("2 m", L2, fa, LOS2, J2), ("0.4 m", L4, fa4, LOS4, J4)):
            o, j, h, u = (stat(LL, p, s) for s in ("overlap_pct_smaller", "jaccard", "overlap_ha", "union_ha"))
            ab, ba = los_pair(LS, p, a, b)
            res.append({
                "pair": p, "grid": grid,
                "visible_land_a_ha": fmt(FF[a]["visible_land_ha"], 2),
                "visible_land_b_ha": fmt(FF[b]["visible_land_ha"], 2),
                "shared_ha": fmt(f(h, "observed"), 2), "union_ha": fmt(f(u, "observed"), 2),
                "pct_of_smaller": fmt(f(o, "observed"), 1), "jaccard": fmt(f(j, "observed"), 3),
                "pct_null_median": fmt(f(o, "null_median"), 1), "pct_null_q95": fmt(f(o, "null_q95"), 1),
                "pct_n_ge": o["n_null_ge_observed"], "pct_p2": fmt(f(o, "p_two_sided")),
                "jaccard_p2": fmt(f(j, "p_two_sided")),
                "shared_n_ge": h["n_null_ge_observed"], "shared_p2": fmt(f(h, "p_two_sided")),
                "union_null_median_ha": fmt(f(u, "null_median"), 2),
                "union_null_max_ha": fmt(f(u, "null_max"), 2),
                "union_margin_above_null_max_ha": fmt(f(u, "observed") - f(u, "null_max"), 2),
                "union_n_ge": u["n_null_ge_observed"], "union_p2": fmt(f(u, "p_two_sided")),
                "n_matched_available": o["n_matched_available"],
                "pool_points_used": JJ["parameters"]["pool_size"],
                "LOS_a_to_b/b_to_a": "/".join("yes" if x["profile_visible"] == "True" else "no" for x in (ab, ba)),
            })

# ---------------- Foci (Table S3): final 0.4 m viewsheds and the 2 m run ----------------
tr = {r["focus"]: r for r in csv.DictReader(
    open(os.path.join(R["land2"], "viewshed_truncation.csv"), encoding="utf-8"))}
cfg = {c["name"]: c for c in json.load(open("walk-lcp-main/viewshed_foci_parion.json", encoding="utf-8"))["foci"]}
land2 = J2["region"]["land_ha"]
s3 = []
for r in csv.DictReader(open(A + "/Viewshed_landmasked_0.4m_rerun/TableS3_foci_visible_land_0.4m_rerun.csv", encoding="utf-8")):
    n = r["focus"]
    t = tr[n]
    la, al = float(r["visible_land_ha"]), float(r["visible_valid_ha_incl_water"])
    s3.append({
        "focus": n, "label": r["label"], "x": r["x"], "y": r["y"], "z_0.4m": r["z_0.4m"],
        "z_2m": fmt(fa[n]["z"], 2), "on_land": fa[n]["on_land"],
        "visible_land_ha_0.4m": fmt(la, 2), "pct_of_land_211.00ha_0.4m": fmt(float(r["pct_of_land"]), 2),
        "visible_water_ha_0.4m": fmt(al - la, 2), "pct_of_viewshed_on_water": fmt(100 * (al - la) / al, 1),
        "visible_land_ha_2m": fmt(fa[n]["visible_land_ha"], 2),
        "pct_of_land_2m": fmt(100 * fa[n]["visible_land_ha"] / land2, 2),
        "farthest_visible_m_2m": fmt(float(t["farthest_visible_cell_m"]), 0),
        "farthest_dem_cell_m": fmt(float(t["farthest_valid_cell_m"]), 0),
        "pct_of_3000m_disc_inside_survey": fmt(float(t["pct_of_disk_covered_by_dem"]), 1),
        "visible_cells_on_survey_edge_2m": t["visible_cells_on_dem_edge"],
        "truncated_by_survey_boundary": t["truncated_by_survey_extent"],
    })


def write(name, data):
    with open(OUTDIR / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


write("FINAL_Table1_overlap_landmasked.csv", t1)
write("FINAL_TableS5_union_landmasked.csv", s5)
write("FINAL_TableS3_foci_visible_land.csv", s3)
if res:
    write("FINAL_resolution_check_0.4m_vs_2m.csv", res)
for t in (t1, s5, res, s3):
    for r in t:
        print(r)
    print()
