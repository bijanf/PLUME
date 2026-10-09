"""Mode-structure helpers: resolution matrix, averaging-kernel width, data-space
patterns of the singular vectors."""

import numpy as np
import pytest

from plume_inv.kernel import design_matrix
from plume_inv.svd import (
    analyse_spectrum,
    averaging_kernel_row,
    captured_fraction,
    laplace_response,
    main_lobe_fwhm,
    orient_modes,
    resolution_matrix,
    sign_changes,
)

Q = np.geomspace(1e3, 1e8, 60)


@pytest.fixture(scope="module")
def spectrum():
    rng = np.random.default_rng(3)
    r = np.sort(rng.uniform(300.0, 7000.0, 80))
    a = np.vstack([design_matrix(r, Q, w) for w in (0.0034, 0.011, 0.030, 0.063)])
    return analyse_spectrum(a / a.max())


class TestResolution:
    def test_resolution_is_a_rank_k_projector(self, spectrum):
        for k in (1, 3, 6):
            r = resolution_matrix(spectrum.vt, k)
            assert np.allclose(r, r.T)
            assert np.allclose(r @ r, r, atol=1e-12)
            assert np.trace(r) == pytest.approx(k)

    def test_full_rank_resolution_is_the_identity(self):
        rng = np.random.default_rng(0)
        vt = np.linalg.qr(rng.normal(size=(8, 8)))[0].T
        assert np.allclose(resolution_matrix(vt, 8), np.eye(8))

    def test_averaging_row_interpolates_between_nodes(self, spectrum):
        r = resolution_matrix(spectrum.vt, 4)
        x = np.log10(Q)
        assert np.allclose(averaging_kernel_row(r, x, x[10]), r[10])
        mid = averaging_kernel_row(r, x, 0.5 * (x[10] + x[11]))
        assert np.allclose(mid, 0.5 * (r[10] + r[11]))


class TestWidths:
    def test_fwhm_of_a_gaussian(self):
        x = np.linspace(-5, 5, 4001)
        s = 0.7
        out = main_lobe_fwhm(x, np.exp(-0.5 * (x / s) ** 2))
        assert out["fwhm"] == pytest.approx(2 * np.sqrt(2 * np.log(2)) * s, rel=1e-4)
        assert not out["censored_low"] and not out["censored_high"]

    def test_fwhm_reports_a_censored_side(self):
        x = np.linspace(0, 1, 101)
        out = main_lobe_fwhm(x, np.exp(-x))
        assert out["censored_low"]
        assert out["x_low"] == 0.0


class TestModes:
    def test_orientation_keeps_the_spectrum_and_makes_the_peak_positive(self, spectrum):
        vt = orient_modes(spectrum.vt)
        assert np.allclose(np.abs(vt), np.abs(spectrum.vt))
        idx = np.argmax(np.abs(vt), axis=1)
        assert np.all(vt[np.arange(vt.shape[0]), idx] > 0)

    def test_captured_fraction_is_one_on_the_full_left_basis(self, spectrum):
        rng = np.random.default_rng(1)
        d = spectrum.u @ rng.normal(size=spectrum.u.shape[1])
        assert captured_fraction(spectrum.u, d, spectrum.u.shape[1]) == pytest.approx(1.0)
        f = [captured_fraction(spectrum.u, d, k) for k in (1, 2, 5)]
        assert 0.0 <= f[0] <= f[1] <= f[2] <= 1.0

    def test_laplace_response_matches_the_kernel_for_every_class(self, spectrum):
        """dOmega/w depends on (r, w) only through lambda = pi w r^2."""
        v = spectrum.vt[2]
        r = np.array([200.0, 1500.0, 6000.0])
        for w in (0.002, 0.03, 0.2):
            direct = design_matrix(r, Q, w) @ v / w
            assert np.allclose(laplace_response(Q, v, np.pi * w * r**2), direct, rtol=1e-12)

    def test_sign_changes_of_the_leading_modes_increase(self, spectrum):
        vt = orient_modes(spectrum.vt)
        assert sign_changes(vt[0]) == 0
        assert sign_changes(np.array([1.0, -1.0, 1e-9, -1e-9, 1.0])) == 2
