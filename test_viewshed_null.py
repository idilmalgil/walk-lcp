# -*- coding: utf-8 -*-
"""
test_viewshed_null.py — Unit tests for viewshed_null.py

Run with:
    python -m pytest test_viewshed_null.py -v

Like test_walk_lcp.py these tests build everything in memory and need no
input data and no GRASS installation — only numpy and pytest. They are
intended as a sanity check a journal reviewer can run in seconds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from viewshed_null import (
    Focus,
    InternalViewshed,
    aggregate_binary,
    aggregate_mean,
    candidate_cells,
    empirical_p,
    holm,
    intersection_ha,
    pct_of_smaller,
    pixel_to_world,
    sample_matched_pairs,
    union_ha,
    world_to_pixel,
)


# =============================================================================
# Aggregation
# =============================================================================

class TestAggregation:
    def test_mean_halves_a_ramp(self):
        arr = np.arange(16, dtype=float).reshape(4, 4)
        out = aggregate_mean(arr, 2)
        assert out.shape == (2, 2)
        # Top-left block is [[0,1],[4,5]] -> mean 2.5
        assert math.isclose(out[0, 0], 2.5)

    def test_mean_factor_one_is_identity(self):
        arr = np.random.default_rng(0).random((5, 5))
        assert np.array_equal(aggregate_mean(arr, 1), arr)

    def test_trims_ragged_edges(self):
        arr = np.ones((5, 7))
        assert aggregate_mean(arr, 2).shape == (2, 3)

    def test_binary_majority_rule(self):
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True                      # 1 of 4 = 25 %
        assert not aggregate_binary(mask, 2, "majority")[0, 0]
        mask[0, 1] = mask[1, 0] = True         # 3 of 4 = 75 %
        assert aggregate_binary(mask, 2, "majority")[0, 0]

    def test_binary_any_and_all(self):
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True
        assert aggregate_binary(mask, 2, "any")[0, 0]
        assert not aggregate_binary(mask, 2, "all")[0, 0]

    def test_binary_rejects_unknown_rule(self):
        with pytest.raises(ValueError):
            aggregate_binary(np.zeros((2, 2), dtype=bool), 2, "nonsense")


# =============================================================================
# Pairwise statistics
# =============================================================================

class TestStatistics:
    CELL_HA = 0.0004        # a 2 m cell is 4 m2

    def test_union_counts_each_cell_once(self):
        a = np.array([[1, 1], [0, 0]], dtype=bool)
        b = np.array([[1, 0], [1, 0]], dtype=bool)
        # union is 3 cells
        assert math.isclose(union_ha(a, b, self.CELL_HA), 3 * self.CELL_HA)

    def test_intersection(self):
        a = np.array([[1, 1], [0, 0]], dtype=bool)
        b = np.array([[1, 0], [1, 0]], dtype=bool)
        assert math.isclose(intersection_ha(a, b, self.CELL_HA), 1 * self.CELL_HA)

    def test_pct_of_smaller_uses_the_smaller_viewshed(self):
        a = np.array([[1, 1, 1, 1]], dtype=bool)     # 4 cells
        b = np.array([[1, 1, 0, 0]], dtype=bool)     # 2 cells, both shared
        # Shared = 2, smaller = 2 -> 100 %, not 50 %.
        assert math.isclose(pct_of_smaller(a, b), 100.0)

    def test_pct_is_symmetric(self):
        rng = np.random.default_rng(3)
        a = rng.random((10, 10)) > 0.5
        b = rng.random((10, 10)) > 0.5
        assert math.isclose(pct_of_smaller(a, b), pct_of_smaller(b, a))

    def test_empty_viewshed_gives_zero_not_nan(self):
        a = np.zeros((3, 3), dtype=bool)
        b = np.ones((3, 3), dtype=bool)
        assert pct_of_smaller(a, b) == 0.0


# =============================================================================
# Inference
# =============================================================================

class TestEmpiricalP:
    def test_floor_with_999_draws(self):
        """An observed value beating every draw gives p = 1/1000, never 0."""
        null = np.zeros(999)
        assert math.isclose(empirical_p(1.0, null), 0.001)

    def test_observed_below_every_draw_gives_p_one(self):
        null = np.ones(999)
        assert math.isclose(empirical_p(0.0, null), 1.0)

    def test_ties_count_towards_the_tail(self):
        """p = (r + 1)/(n + 1) counts null values >= observed."""
        null = np.array([5.0, 5.0, 1.0, 1.0])
        # two ties at 5.0 -> r = 2 -> p = 3/5
        assert math.isclose(empirical_p(5.0, null), 3 / 5)


class TestHolm:
    def test_single_test_is_unchanged(self):
        assert np.allclose(holm([0.02]), [0.02])

    def test_known_seven_test_case(self):
        """The smallest of seven p = 0.001 values becomes 0.007."""
        p = [0.001, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        adjusted = holm(p)
        assert math.isclose(adjusted[0], 0.007)

    def test_monotone_and_capped(self):
        p = [0.01, 0.02, 0.03, 0.04]
        adjusted = holm(p)
        ordered = adjusted[np.argsort(p)]
        assert np.all(np.diff(ordered) >= -1e-12)   # non-decreasing
        assert np.all(adjusted <= 1.0)

    def test_preserves_input_order(self):
        p = [0.9, 0.001, 0.5]
        adjusted = holm(p)
        # The smallest input p is at index 1, so the smallest output must be too.
        assert int(np.argmin(adjusted)) == 1


# =============================================================================
# Matched sampling
# =============================================================================

class TestMatchedSampling:
    @staticmethod
    def _ramp_dem(n=60):
        """A DEM rising 1 m per row, so elevation bands are whole rows."""
        return np.repeat(np.arange(n, dtype=float)[:, None], n, axis=1)

    def test_candidates_respect_elevation_tolerance(self):
        dem = self._ramp_dem()
        valid = np.ones_like(dem, dtype=bool)
        idx = candidate_cells(dem, valid, elevation=30.0, tol=1.0)
        rows = idx // dem.shape[1]
        assert set(np.unique(rows)) == {29, 30, 31}

    def test_candidates_respect_the_valid_mask(self):
        dem = self._ramp_dem()
        valid = np.ones_like(dem, dtype=bool)
        valid[:, :30] = False
        idx = candidate_cells(dem, valid, elevation=30.0, tol=1.0)
        cols = idx % dem.shape[1]
        assert cols.min() >= 30

    def test_nodata_cells_are_never_candidates(self):
        dem = self._ramp_dem()
        dem[10, 10] = np.nan
        valid = np.ones_like(dem, dtype=bool)
        idx = candidate_cells(dem, valid, elevation=10.0, tol=0.5)
        assert (10 * dem.shape[1] + 10) not in set(idx.tolist())

    def test_sampled_pairs_match_elevation_and_distance(self):
        dem = self._ramp_dem()
        valid = np.ones_like(dem, dtype=bool)
        a = Focus("a", 0, 0, row=10, col=10, elevation=10.0)
        b = Focus("b", 0, 0, row=40, col=10, elevation=40.0)
        rng = np.random.default_rng(42)
        pairs = sample_matched_pairs(
            dem, valid, pixel_size=2.0, focus_a=a, focus_b=b,
            n_pairs=50, elev_tol=2.0, dist_tol=0.10, rng=rng,
        )
        assert len(pairs) == 50
        observed_d = 30 * 2.0
        for (ra, ca), (rb, cb) in pairs:
            assert abs(dem[ra, ca] - a.elevation) <= 2.0
            assert abs(dem[rb, cb] - b.elevation) <= 2.0
            d = math.hypot((ra - rb) * 2.0, (ca - cb) * 2.0)
            assert 0.9 * observed_d <= d <= 1.1 * observed_d

    def test_same_seed_gives_the_same_draws(self):
        dem = self._ramp_dem()
        valid = np.ones_like(dem, dtype=bool)
        a = Focus("a", 0, 0, row=10, col=10, elevation=10.0)
        b = Focus("b", 0, 0, row=40, col=10, elevation=40.0)
        kwargs = dict(
            dem=dem, valid=valid, pixel_size=2.0, focus_a=a, focus_b=b,
            n_pairs=20, elev_tol=2.0, dist_tol=0.10,
        )
        first = sample_matched_pairs(rng=np.random.default_rng(7), **kwargs)
        second = sample_matched_pairs(rng=np.random.default_rng(7), **kwargs)
        assert first == second

    def test_impossible_constraints_raise(self):
        dem = self._ramp_dem()
        valid = np.ones_like(dem, dtype=bool)
        a = Focus("a", 0, 0, row=10, col=10, elevation=10.0)
        b = Focus("b", 0, 0, row=40, col=10, elevation=40.0)
        with pytest.raises(RuntimeError):
            # No cell anywhere sits at 5000 m.
            sample_matched_pairs(
                dem, valid, 2.0,
                Focus("a", 0, 0, row=10, col=10, elevation=5000.0), b,
                n_pairs=10, elev_tol=1.0, dist_tol=0.10,
                rng=np.random.default_rng(0),
            )


# =============================================================================
# Internal viewshed backend
# =============================================================================

class TestInternalViewshed:
    def test_flat_plain_is_fully_visible(self):
        dem = np.zeros((41, 41))
        vs = InternalViewshed(dem, pixel_size=1.0, observer_height=1.7,
                              max_distance=20.0).compute(20, 20)
        # Every cell inside the radius should be visible on a flat plain.
        rows = np.arange(41)[:, None] - 20
        cols = np.arange(41)[None, :] - 20
        inside = np.hypot(rows, cols) <= 20.0
        assert np.array_equal(vs & inside, inside)

    def test_observer_cell_is_always_visible(self):
        dem = np.random.default_rng(1).random((31, 31)) * 50
        vs = InternalViewshed(dem, 1.0, max_distance=15.0).compute(15, 15)
        assert vs[15, 15]

    def test_wall_casts_a_shadow(self):
        """A tall wall must hide the ground directly behind it."""
        dem = np.zeros((41, 41))
        dem[:, 25] = 100.0                       # wall along a column
        vs = InternalViewshed(dem, 1.0, observer_height=1.7,
                              max_distance=25.0).compute(20, 20)
        assert vs[20, 24]        # in front of the wall
        assert not vs[20, 30]    # behind the wall

    def test_nothing_beyond_max_distance(self):
        dem = np.zeros((81, 81))
        vs = InternalViewshed(dem, 1.0, max_distance=10.0).compute(40, 40)
        rows = np.arange(81)[:, None] - 40
        cols = np.arange(81)[None, :] - 40
        assert not vs[np.hypot(rows, cols) > 10.0].any()

    def test_higher_observer_sees_further_over_a_ridge(self):
        dem = np.zeros((61, 61))
        dem[:, 35] = 10.0
        low = InternalViewshed(dem, 1.0, observer_height=1.7,
                               max_distance=30.0).compute(30, 30)
        high = InternalViewshed(dem, 1.0, observer_height=50.0,
                                max_distance=30.0).compute(30, 30)
        assert np.count_nonzero(high) > np.count_nonzero(low)

    def test_nodata_observer_raises(self):
        dem = np.zeros((21, 21))
        dem[10, 10] = np.nan
        with pytest.raises(ValueError):
            InternalViewshed(dem, 1.0, max_distance=5.0).compute(10, 10)


# =============================================================================
# End-to-end driver
# =============================================================================

class TestRunPairTests:
    """Exercise the whole pipeline on a synthetic DEM.

    This uses the internal backend and in-memory arrays, so it needs
    neither GDAL nor GRASS. It is the fastest way to confirm the driver,
    the null model and the Holm correction wire together correctly.
    """

    @staticmethod
    def _hilly_dem(n=80, seed=11):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:n, 0:n]
        dem = 20.0 * np.exp(-((xx - 25) ** 2 + (yy - 25) ** 2) / 200.0)
        dem += 15.0 * np.exp(-((xx - 60) ** 2 + (yy - 55) ** 2) / 150.0)
        dem += rng.normal(0, 0.3, size=(n, n))
        return dem

    def _run(self, n_null=15):
        from viewshed_null import InternalViewshed, run_pair_tests

        dem = self._hilly_dem()
        valid = np.ones_like(dem, dtype=bool)
        foci_list = [
            Focus("hill_a", 0, 0, row=25, col=25, elevation=float(dem[25, 25])),
            Focus("hill_b", 0, 0, row=55, col=60, elevation=float(dem[55, 60])),
            Focus("flat_c", 0, 0, row=10, col=70, elevation=float(dem[10, 70])),
        ]
        foci = {f.name: f for f in foci_list}
        backend = InternalViewshed(dem, pixel_size=2.0, max_distance=60.0)
        return run_pair_tests(
            dem=dem, valid=valid, pixel_size=2.0, cell_ha=0.0004,
            foci=foci, pairs=[("hill_a", "hill_b"), ("hill_a", "flat_c")],
            backend=backend, observed_viewsheds=None,
            n_null=n_null, elev_tol=5.0, dist_tol=0.25, seed=99,
        )

    def test_returns_two_statistics_per_pair(self):
        results = self._run()
        assert len(results) == 4                       # 2 pairs x 2 statistics
        assert {r.statistic for r in results} == {"union_ha", "overlap_pct_smaller"}

    def test_p_values_are_in_range_and_holm_is_filled(self):
        for r in self._run():
            assert 0.0 < r.p_value <= 1.0
            assert 0.0 < r.p_holm <= 1.0
            assert r.p_holm >= r.p_value               # Holm never shrinks a p-value

    def test_null_summary_is_internally_consistent(self):
        for r in self._run():
            assert r.null_p05 <= r.null_median <= r.null_p95 <= r.null_max
            assert len(r.null_values) == r.n_null

    def test_union_is_never_smaller_than_either_viewshed(self):
        results = {r.statistic: r for r in self._run() if r.pair.startswith("hill_a x hill_b")}
        assert results["union_ha"].observed > 0.0
        assert 0.0 <= results["overlap_pct_smaller"].observed <= 100.0

    def test_run_is_deterministic_for_a_fixed_seed(self):
        first = [(r.pair, r.statistic, r.observed, r.p_value) for r in self._run()]
        second = [(r.pair, r.statistic, r.observed, r.p_value) for r in self._run()]
        assert first == second


# =============================================================================
# Configuration file
# =============================================================================

class TestLoadConfig:
    def test_round_trip(self, tmp_path):
        import json

        from viewshed_null import load_config

        cfg = {
            "foci": [
                {"name": "a", "x": 1.0, "y": 2.0},
                {"name": "b", "x": 3.0, "y": 4.0},
            ],
            "pairs": [{"a": "a", "b": "b"}],
        }
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        foci, pairs = load_config(path)
        assert [f.name for f in foci] == ["a", "b"]
        assert pairs == [("a", "b")]

    def test_placeholder_coordinates_raise(self, tmp_path):
        import json

        from viewshed_null import load_config

        cfg = {
            "foci": [{"name": "circular_tower", "x": "FILL", "y": "FILL"}],
            "pairs": [],
        }
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        with pytest.raises(RuntimeError, match="placeholder"):
            load_config(path)

    def test_shipped_parion_config_is_valid_json_with_seven_pairs(self):
        import json
        from pathlib import Path

        cfg = json.loads(
            Path("viewshed_foci_parion.json").read_text(encoding="utf-8")
        )
        assert len(cfg["pairs"]) == 7
        names = {f["name"] for f in cfg["foci"]}
        for pair in cfg["pairs"]:
            assert {pair["a"], pair["b"]} <= names

    def test_unknown_focus_in_a_pair_raises(self, tmp_path):
        import json

        from viewshed_null import load_config

        cfg = {
            "foci": [{"name": "a", "x": 1.0, "y": 2.0}],
            "pairs": [{"a": "a", "b": "ghost"}],
        }
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        with pytest.raises(RuntimeError, match="ghost"):
            load_config(path)


# =============================================================================
# Coordinate helpers (shared convention with walk_lcp.py)
# =============================================================================

class TestCoordinates:
    GT = (1000.0, 2.0, 0.0, 2000.0, 0.0, -2.0)

    def test_round_trip(self):
        for r in (0, 7, 50):
            for c in (0, 7, 50):
                x, y = pixel_to_world(self.GT, r, c)
                assert world_to_pixel(self.GT, x, y) == (r, c)


# =============================================================================
# Version 2: land-masked pool model
# =============================================================================

from viewshed_null import (  # noqa: E402
    AnalysisGrid,
    empirical_p_lower,
    empirical_p_two_sided,
    jaccard,
    line_of_sight_profile,
    matched_pool_pairs,
    packed_stats,
    run_pool_model,
    summarise_null,
)


class TestTwoSidedInference:
    def test_lower_tail_mirrors_upper(self):
        null = np.arange(1, 1000, dtype=float)          # 1 .. 999
        assert math.isclose(empirical_p_lower(0.0, null), 1 / 1000)
        assert math.isclose(empirical_p(1000.0, null), 1 / 1000)

    def test_two_sided_doubles_the_smaller_tail(self):
        null = np.arange(1, 1000, dtype=float)
        assert math.isclose(empirical_p_two_sided(1000.0, null), 2 / 1000)
        assert math.isclose(empirical_p_two_sided(0.0, null), 2 / 1000)

    def test_two_sided_is_capped_at_one(self):
        null = np.full(999, 5.0)
        assert empirical_p_two_sided(5.0, null) == 1.0

    def test_summary_reports_exceedance(self):
        null = np.linspace(0, 10, 999)
        s = summarise_null(11.0, null)
        assert s["n_null_ge_observed"] == 0 and s["beyond_all_draws"] == "above"
        assert math.isclose(s["p_upper"], 0.001) and math.isclose(s["p_two_sided"], 0.002)
        s = summarise_null(-1.0, null)
        assert s["beyond_all_draws"] == "below"


class TestJaccardAndPacked:
    def test_jaccard_basic(self):
        a = np.array([1, 1, 0, 0], dtype=bool)
        b = np.array([0, 1, 1, 0], dtype=bool)
        assert math.isclose(jaccard(a, b), 1 / 3)
        assert jaccard(np.zeros(4, bool), np.zeros(4, bool)) == 0.0

    def test_packed_stats_match_boolean_stats(self):
        rng = np.random.default_rng(5)
        a = rng.random(1003) > 0.6
        b = rng.random(1003) > 0.4
        s = packed_stats(np.packbits(a), np.packbits(b), cell_ha=0.0004)
        assert math.isclose(s["overlap_ha"], intersection_ha(a, b, 0.0004))
        assert math.isclose(s["union_ha"], union_ha(a, b, 0.0004))
        assert math.isclose(s["overlap_pct_smaller"], pct_of_smaller(a, b))
        assert math.isclose(s["jaccard"], jaccard(a, b))


class TestPoolMatching:
    def test_matched_pairs_respect_all_constraints(self):
        rng = np.random.default_rng(1)
        xy = rng.random((300, 2)) * 1000
        z = rng.random(300) * 50
        pairs = matched_pool_pairs(z, xy, z_a=10.0, z_b=40.0, separation=300.0,
                                   elev_tol=5.0, dist_tol=0.1)
        assert pairs.shape[0] > 0
        for i, j in pairs:
            assert i != j
            assert abs(z[i] - 10.0) <= 5.0 and abs(z[j] - 40.0) <= 5.0
            d = math.hypot(*(xy[i] - xy[j]))
            assert 270.0 <= d <= 330.0


class TestLineOfSight:
    GT = (0.0, 1.0, 0.0, 100.0, 0.0, -1.0)

    def test_flat_ground_is_visible(self):
        dem = np.zeros((100, 100))
        r = line_of_sight_profile(dem, self.GT, (10.5, 50.5), (90.5, 50.5), 1.7, 1.7)
        assert r["visible"] and math.isclose(r["clearance_m"], 1.7, abs_tol=1e-6)

    def test_wall_blocks_both_directions(self):
        dem = np.zeros((100, 100))
        dem[:, 50] = 10.0
        ab = line_of_sight_profile(dem, self.GT, (10.5, 50.5), (90.5, 50.5), 1.7, 1.7)
        ba = line_of_sight_profile(dem, self.GT, (90.5, 50.5), (10.5, 50.5), 1.7, 1.7)
        assert not ab["visible"] and not ba["visible"]
        # the wall (column 50, x = 50.5) lies 40 m from the observer at x = 10.5
        assert math.isclose(ab["blocked_at_m"], 40.0, abs_tol=0.5)
        assert math.isclose(ba["blocked_at_m"], 40.0, abs_tol=0.5)


class TestPoolModelEndToEnd:
    """The whole version-2 driver on a synthetic DEM with a 'sea' band."""

    @staticmethod
    def _grid(n=70):
        rng = np.random.default_rng(3)
        yy, xx = np.mgrid[0:n, 0:n]
        dem = 30.0 * np.exp(-((xx - 30) ** 2 + (yy - 35) ** 2) / 400.0) + rng.normal(0, 0.2, (n, n))
        dem[:, :8] = 0.0                                    # flat 'sea' strip
        valid = np.ones((n, n), bool)
        land = valid.copy()
        land[:, :8] = False
        return AnalysisGrid(dem=dem, valid=valid, land=land,
                            gt=(0.0, 2.0, 0.0, 2.0 * n, 0.0, -2.0), projection="", factor=1)

    def test_runs_and_never_counts_water(self):
        from viewshed_null import internal_viewsheds

        grid = self._grid()
        foci = [Focus("a", 60.5, 70.5), Focus("b", 100.5, 60.5)]
        compute = lambda pts, tag: internal_viewsheds(grid, pts, max_distance=200.0)  # noqa: E731
        out = run_pool_model(grid, foci, [("a", "b")], compute, pool_size=300,
                             n_null=50, elev_tol=8.0, dist_tol=0.3, seed=1)
        assert {r["statistic"] for r in out["results"]} == {
            "overlap_ha", "overlap_pct_smaller", "jaccard", "union_ha"}
        cols = grid.dem.shape[1]
        assert np.all(out["pool"]["index"] % cols >= 8)       # pool drawn from land only
        assert all(r["p_holm_two_sided"] >= r["p_two_sided"] for r in out["results"])
        # the counted viewsheds contain no water cell
        rows = grid.dem.shape[0]
        vis = np.unpackbits(out["foci_packed"][0], count=rows * cols).reshape(rows, cols)
        assert not vis[:, :8].any()
