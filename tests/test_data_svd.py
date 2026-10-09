"""The NESCA loader's provenance checks, and the identifiability machinery."""

import numpy as np
import pandas as pd
import pytest

from plume_inv import data as D
from plume_inv.svd import ObservationSet, analyse_spectrum, stacked_operator

needs_nesca = pytest.mark.nesca_data


class TestNescaLoader:
    @needs_nesca
    def test_loads_the_published_cores(self):
        df = D.load_nesca()
        assert len(df) == 132
        assert int(df["complete_four_fractions"].sum()) == 130
        assert set(df["dive"].dropna()) == {"T887", "T888", "T889", "T890", "T891"}

    @needs_nesca
    def test_recovers_both_core_areas_and_assigns_them_by_dive(self):
        """The footnote quotes one core diameter; the g/m2 column implies two.
        Dive T888 used a 7.08 cm corer, the rest 6.90 cm -- a 5.3 % step."""
        df = D.load_nesca()
        areas = sorted(df["core_area_m2"].dropna().unique())
        assert len(areas) == 2
        assert areas[1] / areas[0] == pytest.approx(1.053, abs=0.002)
        big = set(df.loc[df["core_area_m2"] == areas[1], "dive"])
        assert big == {"T888"}

    @needs_nesca
    def test_reproduces_the_published_mass_per_area(self):
        """Total glass divided by the per-dive area must give back the published
        g/m2 column.  This is the check that would catch a wrong core area."""
        df = D.load_nesca()
        got = df["published_total_glass_g"] / df["core_area_m2"]
        assert got.values == pytest.approx(df["published_g_per_m2"].values, rel=1e-9)

    @needs_nesca
    def test_qualitative_percentages_use_the_authors_convention(self):
        df = D.load_nesca()
        assert D.PCT_BELOW_ONE == 0.3
        assert D.PCT_WELL_BELOW_ONE == 0.1
        # 30 cores carry at least one qualitative percentage.
        assert int(df["any_qualitative_pct"].sum()) == 30
        # Where a percentage is qualitative it must still be a number.
        for name, *_ in [(f[0],) for f in D.FRACTIONS]:
            q = df[f"pct_glass_qualitative[{name}]"]
            assert np.all(np.isfinite(df.loc[q, f"pct_glass[{name}]"]))

    @needs_nesca
    def test_a_lost_fraction_is_missing_not_zero(self):
        """One core lost its 63-125 um fraction to a hole in the sieve.  A
        missing measurement is not a measurement of zero."""
        df = D.load_nesca()
        col = df["glass_g[63-125um]"]
        assert col.isna().sum() == 1
        assert not (col == 0).any()

    @needs_nesca
    def test_reconstruction_residual_is_carried_per_core(self):
        df = D.load_nesca()
        r = df["total_reconstruction_rel_error"]
        assert r.median() < 0.01
        assert (r > 0.05).sum() == 17  # flagged for down-weighting
        assert r.notna().all()

    @needs_nesca
    def test_local_xy_is_metric_and_centred(self):
        df = D.load_nesca()
        assert abs(df["x_m"].mean()) < 1.0
        assert abs(df["y_m"].mean()) < 1.0
        span = np.hypot(df["x_m"].max() - df["x_m"].min(), df["y_m"].max() - df["y_m"].min())
        assert 1e4 < span < 2e4  # a ~10-13 km survey

    def test_local_xy_matches_great_circle_distance(self):
        lat0, lon0 = 41.0, -127.5
        x, y = D.local_xy([41.05], [-127.45], lat0, lon0)
        # haversine for the same pair
        p1, p2 = np.radians([41.0, 41.05]), np.radians([-127.5, -127.45])
        dlat, dlon = p1[1] - p1[0], p2[1] - p2[0]
        a = np.sin(dlat / 2) ** 2 + np.cos(p1[0]) * np.cos(p1[1]) * np.sin(dlon / 2) ** 2
        great_circle = 2 * 6371000.0 * np.arcsin(np.sqrt(a))
        assert float(np.hypot(x, y)[0]) == pytest.approx(great_circle, rel=2e-3)

    def test_missing_raw_data_is_an_error_not_a_silent_empty_frame(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            D.load_nesca(raw_dir=tmp_path)

    @needs_nesca
    def test_an_unexpected_core_area_is_refused(self, tmp_path):
        """Corrupt the g/m2 column so it implies a third core geometry; the
        loader must refuse rather than silently mis-scale every deposit."""
        raw = D.RAW_DIR
        t = pd.ExcelFile(raw / "mmc3.xls").parse("Sheet1")
        t.loc[t["Sample"].notna() & t["g/m2"].notna(), "g/m2"] = (
            pd.to_numeric(t["g/m2"], errors="coerce") * 1.4
        )
        (tmp_path / "mmc1.xls").write_bytes((raw / "mmc1.xls").read_bytes())
        # An xlsx payload under the .xls name: the loader sniffs the content.
        t.to_excel(tmp_path / "mmc3.xls", engine="openpyxl", sheet_name="Sheet1", index=False)
        with pytest.raises(ValueError, match="core area"):
            D.load_nesca(raw_dir=tmp_path)


class TestSpectrum:
    def test_recovers_a_known_spectrum(self):
        s_true = np.array([4.0, 2.0, 1.0, 0.25])
        a = np.diag(s_true)
        sp = analyse_spectrum(a)
        assert sp.singular_values == pytest.approx(s_true)

    def test_fits_a_geometric_decay_rate(self):
        c = 0.7
        n = np.arange(12)
        a = np.diag(np.exp(-c * n))
        assert analyse_spectrum(a).decay_rate == pytest.approx(c, rel=1e-6)

    def test_relative_mode_count(self):
        sp = analyse_spectrum(np.diag([1.0, 0.1, 0.01, 0.001]))
        assert sp.n_resolvable_relative(0.05) == 2
        assert sp.n_resolvable_relative(0.005) == 3

    def test_snr_count_is_monotone_under_stacking_more_classes(self):
        """The mode-count criterion used to test polydispersity as a filter
        must not be able to fall when data are
        added: stacking rows can only increase A^T A."""
        r = np.linspace(500.0, 8000.0, 30)
        q = np.geomspace(1e4, 1e7, 40)
        sets = [
            ObservationSet(f"c{i}", w, r, np.full(r.size, 0.5))
            for i, w in enumerate((0.0037, 0.0128, 0.0367, 0.0817))
        ]
        counts = []
        for k in range(1, len(sets) + 1):
            sp = analyse_spectrum(stacked_operator(sets[:k], q, whiten=True))
            counts.append(sp.n_resolvable_snr(1e4))
        assert counts == sorted(counts)

    def test_whitening_divides_each_row_by_its_error(self):
        r = np.array([1e3, 2e3])
        q = np.geomspace(1e5, 1e6, 5)
        sigma = np.array([2.0, 4.0])
        plain = stacked_operator([ObservationSet("a", 0.03, r, sigma)], q, whiten=False)
        white = stacked_operator([ObservationSet("a", 0.03, r, sigma)], q, whiten=True)
        assert white == pytest.approx(plain / sigma[:, None])

    def test_picard_truncation_finds_the_turning_point(self):
        """Coefficients that fall and then rise must be truncated at the dip."""
        sp = analyse_spectrum(np.diag(np.exp(-0.8 * np.arange(10))))
        # data whose projection decays for 4 modes then flattens at the noise floor
        d = np.array([1.0, 0.3, 0.09, 0.03, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        k = sp.picard_truncation(d)
        assert 3 <= k <= 5
