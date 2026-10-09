"""Heat-context helpers: energy integrals, energy maps, crossing durations and the
selection of verified literature values."""

import json
from pathlib import Path

import numpy as np
import pytest

from plume_inv import heat_context as HC
from plume_inv import sources

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "heat_context.json"


class TestEnergy:
    def test_dyke_energy_matches_the_analytic_integral(self):
        length, height, tau = 5e3, 1e3, 15 * 3600.0
        b = sources.BASALT
        c = 2 * length * height * b.k_therm * b.delta_t / np.sqrt(np.pi * b.kappa)
        exact = 2 * c * (np.sqrt(tau) - np.sqrt(HC.T0_S))
        got = HC.energy_delivered(lambda t: sources.phi_dyke(t, length, height), tau)
        assert got == pytest.approx(exact, rel=2e-3)

    def test_constant_flux_integrates_exactly(self):
        got = HC.energy_delivered(lambda t: np.full_like(t, 3.0), 101.0, n=50)
        assert got == pytest.approx(3.0 * 100.0)

    def test_mean_power(self):
        assert HC.mean_power(2e16, 15 * 3600.0) == pytest.approx(2e16 / 54000.0)
        assert np.allclose(HC.mean_power([1.0, 4.0], [2.0, 2.0]), [0.5, 2.0])


class TestMaps:
    def test_lava_map_is_linear_in_area(self):
        dur = np.array([3600.0, 7200.0])
        areas = np.array([1e6, 7e6, 40e6])
        emap = HC.lava_energy_map(dur, areas, 1800.0, 15e6, n=2000)
        assert emap.shape == (3, 2)
        direct = HC.energy_delivered(
            lambda t: sources.phi_lava_cooling(t, 40e6, 1800.0), 7200.0, n=2000
        )
        assert emap[2, 1] == pytest.approx(direct, rel=1e-12)

    def test_lava_map_grows_with_duration(self):
        dur = np.geomspace(3600.0, 3.6e5, 5)
        curve = HC.lava_energy_curve(dur, 15e6, 1800.0, n=2000)
        assert np.all(np.diff(curve) > 0)

    def test_dyke_map_scales_with_face_area(self):
        lengths = np.array([1e3, 3e4])
        heights = np.array([1e2, 5e3])
        emap = HC.dyke_energy_map(lengths, heights, 54000.0, 3e4, 5e3, n=2000)
        assert emap.shape == (2, 2)
        assert emap[1, 1] / emap[0, 0] == pytest.approx(3e4 * 5e3 / (1e3 * 1e2))
        direct = HC.energy_delivered(lambda t: sources.phi_dyke(t, 1e3, 5e3), 54000.0, n=2000)
        assert emap[1, 0] == pytest.approx(direct, rel=1e-12)

    def test_crossing_duration_inverts_the_dyke_integral(self):
        length, height = 5e3, 1e3
        b = sources.BASALT
        c = 2 * length * height * b.k_therm * b.delta_t / np.sqrt(np.pi * b.kappa)
        target = 2e16
        exact = (target / (2 * c) + 1.0) ** 2
        phi = lambda t: sources.phi_dyke(t, length, height)  # noqa: E731
        got = HC.crossing_duration(phi, target, 3600.0, 1e7)
        # the root is exact for the quadrature it inverts ...
        assert HC.energy_delivered(phi, got) == pytest.approx(target, rel=1e-6)
        # ... and that quadrature (the source test's, 20000 steps from 1 s)
        # over-counts the t^-1/2 onset slightly, so the root sits a little early
        assert got == pytest.approx(exact, rel=2e-2)
        assert got < exact


class TestLiterature:
    CTX = {
        "entries": [
            {"id": "a", "verified": True, "low_W": 1.0, "high_W": 5.0, "value_W": None},
            {"id": "b", "verified": True, "low_W": None, "high_W": 9.0, "value_W": 7.0},
            {"id": "c", "verified": False, "low_W": 0.1, "high_W": 100.0},
            {"id": "d", "verified": True, "low_J": 2.0, "high_J": 3.0, "energy_J": None},
            {"id": "e", "verified": True, "energy_J": 8.0, "high_J": 8.0},
        ]
    }

    def test_only_verified_entries_are_returned(self):
        ids = [e["id"] for e in HC.verified_entries(self.CTX)]
        assert "c" not in ids and ids == ["a", "b", "d", "e"]

    def test_unverified_id_is_refused(self):
        with pytest.raises(KeyError):
            HC.verified_entries(self.CTX, ["a", "c"])

    def test_envelopes_skip_missing_values(self):
        rows = HC.verified_entries(self.CTX, ["a", "b"])
        assert HC.envelope_W(rows) == (1.0, 9.0)
        assert HC.envelope_J(HC.verified_entries(self.CTX, ["d", "e"])) == (2.0, 8.0)
        with pytest.raises(ValueError):
            HC.envelope_J(rows)


@pytest.mark.skipif(not RESULTS.exists(), reason="results/heat_context.json not built")
class TestResults:
    R = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}

    def test_ladder_uses_verified_entries_only(self):
        ctx = json.loads((ROOT / "results" / "heat_flux_context.json").read_text())
        ok = {e["id"] for e in ctx["entries"] if e.get("verified") is True}
        for row in self.R["ladder"]:
            assert set(row["entry_ids"]) <= ok
            assert row["low_W"] <= row["high_W"]

    def test_lava_map_reproduces_the_source_test(self):
        lm = self.R["lava_map"]
        assert lm["reproduces_source_test_rel_err"] < 1e-9
        assert lm["linearity_in_area_max_rel_err"] < 1e-9
        assert lm["crossing_duration_h_at_mapped_area"] == pytest.approx(
            lm["crossing_duration_h_power_law_source_test"], rel=0.03
        )
        assert lm["area_factor_needed_at_reference_duration"] == pytest.approx(
            lm["floor_J"] / lm["energy_at_mapped_area_reference_J"]
        )

    def test_dyke_map_reproduces_the_source_test(self):
        assert self.R["dyke_map"]["reproduces_source_test_rel_err"] < 1e-9

    def test_ratios_are_consistent(self):
        n, r = self.R["nesca"], self.R["ratios"]
        gh = r["_definitions"]["global_hydrothermal_central_W"]
        assert r["blocky"]["over_global_hydrothermal_central"] == pytest.approx(
            n["blocky"]["median_W"] / gh
        )
        lo, hi = r["blocky"]["hours_to_release_megaplume_range"]
        assert hi / lo == pytest.approx(10.0)
