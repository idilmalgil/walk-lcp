# -*- coding: utf-8 -*-
"""
test_walk_lcp.py — Unit tests for walk_lcp.py

Run with:
    python -m pytest test_walk_lcp.py -v

These tests are deliberately small and do not require any input data —
they build synthetic DEMs in memory and check the algorithm produces the
expected paths and travel times. They are intended as a sanity check that
a journal reviewer can run in seconds on any machine that has the
dependencies installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from walk_lcp import (
    tobler_speed,
    world_to_pixel,
    pixel_to_world,
    dijkstra_from_start,
    reconstruct_path,
)


# =============================================================================
# Tobler's hiking function
# =============================================================================

class TestToblerSpeed:
    def test_peak_at_minus_five_percent(self):
        """Tobler's function peaks at S = -0.05 (slight downhill)."""
        peak = tobler_speed(-0.05)
        # 6 km/h / 3.6 = 1.6667 m/s
        assert math.isclose(peak, 6.0 / 3.6, rel_tol=1e-9)

    def test_flat_terrain(self):
        """At S = 0 the speed should be 6 * exp(-3.5 * 0.05) / 3.6 m/s."""
        expected = 6.0 * math.exp(-3.5 * 0.05) / 3.6
        assert math.isclose(tobler_speed(0.0), expected, rel_tol=1e-9)

    def test_symmetry_around_minus_five_percent(self):
        """The function is symmetric about S = -0.05."""
        # S = -0.05 + delta and S = -0.05 - delta should give equal speeds.
        for delta in (0.05, 0.1, 0.25, 0.5):
            assert math.isclose(
                tobler_speed(-0.05 + delta),
                tobler_speed(-0.05 - delta),
                rel_tol=1e-12,
            )

    def test_minimum_speed_is_clamped(self):
        """Extreme slopes should not drive the speed below the floor."""
        # A near-vertical slope:
        assert tobler_speed(10.0) >= 0.01
        assert tobler_speed(-10.0) >= 0.01

    def test_uphill_slower_than_optimal_downhill(self):
        assert tobler_speed(0.10) < tobler_speed(-0.05)


# =============================================================================
# Pixel / world conversions
# =============================================================================

class TestPixelWorld:
    GT = (1000.0, 1.0, 0.0, 2000.0, 0.0, -1.0)  # origin (1000, 2000), 1 m px

    def test_round_trip(self):
        for r in (0, 5, 99):
            for c in (0, 5, 99):
                x, y = pixel_to_world(self.GT, r, c)
                rr, cc = world_to_pixel(self.GT, x, y)
                assert (rr, cc) == (r, c)

    def test_origin_pixel_centre(self):
        x, y = pixel_to_world(self.GT, 0, 0)
        # Cell-centre of (row=0, col=0): origin + half pixel
        assert math.isclose(x, 1000.5)
        assert math.isclose(y, 1999.5)


# =============================================================================
# Dijkstra on synthetic DEMs
# =============================================================================

class TestDijkstra:
    @staticmethod
    def _flat_dem(shape=(5, 5)):
        return np.zeros(shape, dtype=np.float32)

    @staticmethod
    def _ridge_dem(shape=(5, 5), ridge_col=2, ridge_height=100.0):
        dem = np.zeros(shape, dtype=np.float32)
        dem[:, ridge_col] = ridge_height
        return dem

    GT = (0.0, 1.0, 0.0, 5.0, 0.0, -1.0)  # 1 m pixels

    def test_flat_dem_self_cost_is_zero(self):
        dem = self._flat_dem()
        cost, _ = dijkstra_from_start(dem, self.GT, (0, 0), nodata=None)
        assert cost[0, 0] == 0.0

    def test_flat_dem_all_cells_reachable(self):
        dem = self._flat_dem()
        cost, _ = dijkstra_from_start(dem, self.GT, (2, 2), nodata=None)
        assert np.all(np.isfinite(cost))

    def test_flat_dem_diagonal_costs_more_than_axis(self):
        """On flat terrain, moving sqrt(2) m diagonally takes more time than 1 m axis."""
        dem = self._flat_dem()
        cost, _ = dijkstra_from_start(dem, self.GT, (2, 2), nodata=None)
        # (2,3) is 1 m east; (1,3) is sqrt(2) m NE.
        assert cost[1, 3] > cost[2, 3]
        # And the diagonal cost should be sqrt(2) times the axis cost:
        assert math.isclose(cost[1, 3] / cost[2, 3], math.sqrt(2.0), rel_tol=1e-9)

    def test_ridge_increases_cost(self):
        """A high ridge between start and stop should be slower to cross than flat."""
        flat = self._flat_dem()
        ridge = self._ridge_dem()
        cost_flat, _ = dijkstra_from_start(flat, self.GT, (2, 0), nodata=None)
        cost_ridge, _ = dijkstra_from_start(ridge, self.GT, (2, 0), nodata=None)
        assert cost_ridge[2, 4] > cost_flat[2, 4]

    def test_nodata_blocks_traversal(self):
        """Cells with the NoData value should not be traversed."""
        dem = self._flat_dem()
        dem[:, 2] = -9999.0  # wall of NoData blocking E side
        cost, _ = dijkstra_from_start(dem, self.GT, (2, 0), nodata=-9999.0)
        # East side of the wall must be unreachable.
        assert not math.isfinite(cost[2, 4])


# =============================================================================
# Path reconstruction
# =============================================================================

class TestReconstruct:
    GT = (0.0, 1.0, 0.0, 5.0, 0.0, -1.0)

    def test_round_trip_on_flat_dem(self):
        dem = np.zeros((5, 5), dtype=np.float32)
        start = (0, 0)
        stop = (4, 4)
        _, prev = dijkstra_from_start(dem, self.GT, start, nodata=None)
        path, dist = reconstruct_path(prev, self.GT, start, stop)
        assert path is not None
        assert path[0] == start
        assert path[-1] == stop
        # On flat terrain the shortest path is the diagonal: 4 * sqrt(2) m.
        assert math.isclose(dist, 4.0 * math.sqrt(2.0), rel_tol=1e-9)

    def test_unreachable_returns_none(self):
        dem = np.zeros((5, 5), dtype=np.float32)
        dem[:, 2] = -9999.0
        start = (2, 0)
        stop = (2, 4)
        _, prev = dijkstra_from_start(dem, self.GT, start, nodata=-9999.0)
        path, dist = reconstruct_path(prev, self.GT, start, stop)
        assert path is None
        assert dist is None
