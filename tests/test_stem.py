"""Stem closures: internal consistency, and reproduction of the MTT coefficients."""

import numpy as np
import pytest

from plume_inv import stem
from plume_inv.constants import C_F_PLANAR, C_F_POINT, C_Q_POINT, K_HEAT

N = 1.0e-3
EPS = 0.1


def test_heat_flux_constant():
    """k = rho c / (alpha g) ~ 2.1e9 W s^3 m^-4 (Pegler & Ferguson 2021)."""
    assert stem.heat_flux_constant() == pytest.approx(2.1e9, rel=0.01)
    assert K_HEAT == pytest.approx(stem.heat_flux_constant())


def test_heat_flux_roundtrip():
    f0 = np.array([1.0, 1e2, 6e2, 1e4])
    assert stem.buoyancy_flux_from_heat(stem.heat_flux(f0)) == pytest.approx(f0)


def test_point_closure_is_self_inverse():
    """C_F_POINT must equal C_Q_POINT**(-4/3) for the two closures to invert."""
    assert C_F_POINT == pytest.approx(C_Q_POINT ** (-4.0 / 3.0), rel=2e-3)
    q = np.geomspace(1e4, 1e7, 25)
    f0 = stem.f0_point(q, N, EPS)
    assert stem.q_umb_point(f0, N, EPS) == pytest.approx(q, rel=3e-3)


def test_planar_closure_is_self_inverse():
    q = np.geomspace(1e4, 1e7, 25)
    for length in (1e3, 3e3, 1e4):
        f0 = stem.f0_planar(q, N, length, EPS)
        assert stem.q_umb_planar(f0, N, length, EPS) == pytest.approx(q)


def test_transition_length_formula():
    """l* = 3 (eps Q / N)^(1/3): pin the formula, including the epsilon.

    The previous version of this test asserted only that the two closures give
    buoyancy fluxes within a factor of two of each other at l = l*.  That band is
    analytically independent of the epsilon inside l*, so deleting epsilon --
    which changes l* by a factor 2.15 -- passed.  Assert the formula itself.
    """
    q = 7.6e5
    got = float(stem.transition_length(q, N, EPS))
    assert got == pytest.approx(3.0 * (EPS * q / N) ** (1.0 / 3.0))
    assert got == pytest.approx(1.27e3, rel=0.01)  # ~1.3 km at NESCA scales
    # Scaling: l* must be homogeneous of degree 1/3 in eps, Q and 1/N.
    assert float(stem.transition_length(8 * q, N, EPS)) == pytest.approx(2 * got)
    assert float(stem.transition_length(q, N, 8 * EPS)) == pytest.approx(2 * got)
    assert float(stem.transition_length(q, 8 * N, EPS)) == pytest.approx(got / 2)


def test_source_geometry_regime_at_nesca_scales():
    """The NESCA lava flow is kilometres across, so at the benchmark flux the
    source is in the PLANAR regime: mapped fissures exceed l* ~ 1.3 km.  This is
    the quantitative basis for preferring a line source."""
    q = 7.6e5
    l_star = float(stem.transition_length(q, N, EPS))
    assert 1.0e3 < l_star < 1.6e3
    # A 5 km fissure is well into the planar regime...
    assert 5.0e3 > 2 * l_star
    # ...and delivers a lower buoyancy flux for the same umbrella flux than a
    # point source would, because the same Q is fed by a weaker plume.
    assert float(stem.f0_planar(q, N, 5.0e3, EPS)) < float(stem.f0_point(q, N, EPS))


def test_gaussian_length_scale():
    assert stem.gaussian_length_scale(7.6e5, 0.03) == pytest.approx(5033.2, rel=1e-4)


class TestMTTIntegration:
    """Direct integration of the MTT equations fixes the closure coefficients."""

    def test_point_source_coefficient_matches_benchmark(self):
        """Q at the TOP OF RISE equals 3.52 (eps^2 F0^3 / N^5)^(1/4).

        This identifies which level the Pegler & Ferguson closure refers to:
        the top of rise, not the neutral level (where the coefficient is 2.49).
        """
        sol = stem.solve_stem_mtt(6e2, N, EPS)
        assert sol.c_q_max == pytest.approx(C_Q_POINT, rel=2e-3)
        assert sol.c_q_neutral == pytest.approx(2.4893, rel=1e-3)
        assert sol.c_q_neutral < sol.c_q_max

    def test_point_source_coefficient_is_universal(self):
        """The coefficient must not depend on F0, N, eps or the starting height."""
        ref = stem.solve_stem_mtt(6e2, N, EPS).c_q_max
        for f0 in (1e1, 1e3, 1e5):
            assert stem.solve_stem_mtt(f0, N, EPS).c_q_max == pytest.approx(ref, rel=1e-6)
        for n in (5e-4, 2e-3):
            assert stem.solve_stem_mtt(6e2, n, EPS).c_q_max == pytest.approx(ref, rel=1e-6)
        for eps in (0.05, 0.15):
            assert stem.solve_stem_mtt(6e2, N, eps).c_q_max == pytest.approx(ref, rel=1e-6)
        for z0f in (1e-4, 1e-7):
            assert stem.solve_stem_mtt(6e2, N, EPS, z0_frac=z0f).c_q_max == pytest.approx(
                ref, rel=1e-6
            )

    def test_fluxes_are_physical(self):
        sol = stem.solve_stem_mtt(6e2, N, EPS)
        assert 0 < sol.z_neutral < sol.z_max
        assert np.all(np.diff(sol.q) > 0)  # volume flux grows monotonically
        assert np.all(np.diff(sol.f) < 0)  # buoyancy flux decreases
        assert sol.f[0] == pytest.approx(6e2, rel=1e-6)

    def test_planar_coefficient_matches_benchmark(self):
        """Line-plume integration reproduces the planar closure coefficient 0.326."""
        sol = stem.solve_stem_mtt_planar(10.0, N, EPS)
        expected = C_F_PLANAR ** (-2.0 / 3.0)  # 2.1112
        assert sol.c_q_max == pytest.approx(expected, rel=2e-3)

    def test_planar_coefficient_is_universal(self):
        ref = stem.solve_stem_mtt_planar(10.0, N, EPS).c_q_max
        for f in (0.1, 1.0, 1e3):
            assert stem.solve_stem_mtt_planar(f, N, EPS).c_q_max == pytest.approx(ref, rel=1e-6)


class TestBenchmarkReproduction:
    """The Pegler & Ferguson (2021) numbers."""

    def test_length_scale(self):
        """L = sqrt(Q/w) ~ 5.0 km for Q = 7.6e5 m^3/s, w = 3 cm/s."""
        assert float(stem.gaussian_length_scale(7.6e5, 0.03)) == pytest.approx(5.0e3, rel=0.02)

    def test_buoyancy_flux(self):
        """F0 ~ 6e2 m^4 s^-3 for the published Q."""
        assert float(stem.f0_point(7.6e5, N, EPS)) == pytest.approx(6e2, rel=0.02)

    def test_heat_flux_in_published_range(self):
        """Phi = k F0 must land in the published 1.5 +/- 0.9 TW."""
        phi = float(stem.heat_flux(stem.f0_point(7.6e5, N, EPS)))
        assert 1.2e12 < phi < 1.6e12
        assert abs(phi - 1.5e12) < 0.9e12

    def test_point_source_rises_too_high(self):
        """A point source of this strength rises several km -- far above the
        ~1 km rise of observed megaplumes.  This is the quantitative basis for
        preferring a fissure source."""
        sol = stem.solve_stem_mtt(6e2, N, EPS)
        assert sol.z_max > 3.0e3
        assert sol.z_neutral > 2.0e3

    def test_fissure_source_lowers_the_rise_height(self):
        """Spreading the same buoyancy flux over a few-km fissure brings the
        neutral level down towards the observed ~1 km."""
        f0_total = 6e2
        point = stem.solve_stem_mtt(f0_total, N, EPS)
        for length in (2e3, 5e3):
            planar = stem.solve_stem_mtt_planar(f0_total / length, N, EPS)
            assert planar.z_max < point.z_max


class TestRiseHeightCoefficients:
    """The rise-height coefficients, and the epsilon dependence that the
    (f0/N^3)^(1/3) grouping hides for a line source."""

    def test_point_rise_height_helper_matches_integration(self):
        for f0 in (1e2, 6e2, 1e4):
            for n in (5e-4, 1e-3):
                for eps in (0.05, 0.1, 0.15):
                    sol = stem.solve_stem_mtt(f0, n, eps)
                    assert stem.point_rise_height(f0, n, eps, "max") == pytest.approx(
                        sol.z_max, rel=1e-4
                    )
                    assert stem.point_rise_height(f0, n, eps, "neutral") == pytest.approx(
                        sol.z_neutral, rel=1e-4
                    )

    def test_planar_rise_height_carries_an_explicit_epsilon(self):
        """z_max / (f0/N^3)^(1/3) is NOT a constant: it scales as eps^(-1/3).

        Quoting 3.5785 as a universal coefficient is correct only at eps = 0.1.
        The universal grouping is z_max = 1.6609899 (f0/(eps N^3))^(1/3).
        """
        f0, n = 10.0, 1.0e-3
        naive = {}
        for eps in (0.05, 0.1, 0.15):
            sol = stem.solve_stem_mtt_planar(f0, n, eps)
            naive[eps] = sol.z_max / (f0 / n**3) ** (1.0 / 3.0)
            assert stem.planar_rise_height(f0, n, 1.0, eps) == pytest.approx(sol.z_max, rel=1e-4)
        assert naive[0.05] > naive[0.10] > naive[0.15]
        assert naive[0.05] / naive[0.15] == pytest.approx((0.15 / 0.05) ** (1 / 3), rel=1e-3)
        assert naive[0.10] == pytest.approx(3.5785, rel=1e-3)

    def test_planar_volume_flux_coefficient_is_epsilon_independent(self):
        """Unlike the rise height, the volume-flux coefficient IS universal."""
        ref = stem.solve_stem_mtt_planar(10.0, 1e-3, 0.1).c_q_max
        for eps in (0.05, 0.15):
            assert stem.solve_stem_mtt_planar(10.0, 1e-3, eps).c_q_max == pytest.approx(
                ref, rel=1e-6
            )

    def test_fissure_length_inverts_the_rise_height(self):
        for z in (500.0, 1000.0, 2000.0):
            length = stem.fissure_length_for_rise(z, 6e2, 9.086e-4, 0.1)
            assert stem.planar_rise_height(6e2, 9.086e-4, length, 0.1) == pytest.approx(z)


class TestVariableStratification:
    """Integrating through a measured N(z) profile."""

    def test_reduces_to_the_uniform_case(self):
        n = 1.0e-3
        uniform = stem.solve_stem_mtt(6e2, n, 0.1)
        variable = stem.solve_stem_mtt_profile(6e2, lambda _z: n**2, 0.1)
        assert variable.z_max == pytest.approx(uniform.z_max, rel=1e-4)
        assert variable.z_neutral == pytest.approx(uniform.z_neutral, rel=1e-4)
        assert variable.q_max == pytest.approx(uniform.q_max, rel=1e-4)

    def test_increasing_stratification_aloft_lowers_the_rise_height(self):
        """This is why a single deep-layer N over-estimates the rise of a plume
        that leaves that layer."""
        n_deep = 9.086e-4

        def n2(z):
            return (n_deep * (1.0 + 3.0 * max(z - 1000.0, 0.0) / 2000.0)) ** 2

        uniform = stem.solve_stem_mtt(6e2, n_deep, 0.1)
        variable = stem.solve_stem_mtt_profile(6e2, n2, 0.1, n_scale=n_deep)
        assert variable.z_max < uniform.z_max
        assert variable.z_max > 1000.0

    def test_reports_when_the_plume_reaches_the_ceiling(self):
        weak = 1e-5

        def n2(_z):
            return weak**2

        sol = stem.solve_stem_mtt_profile(6e2, n2, 0.1, z_ceiling=500.0, n_scale=weak)
        assert sol.reached_ceiling is True
        assert sol.z_max == pytest.approx(500.0)
        deep = stem.solve_stem_mtt_profile(6e2, lambda _z: 1e-6, 0.1, z_ceiling=1e5, n_scale=1e-3)
        assert deep.reached_ceiling is False
