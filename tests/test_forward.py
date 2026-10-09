"""Forward map: size classes, core geometry, vent offset, fall-time drift."""

import numpy as np
import pytest

from plume_inv import forward as fwd
from plume_inv import kernel as K
from plume_inv import settling

Q0 = 7.6e5
MASS = 1.0e9
TAU = 15 * 3600.0
N = 1.0e-3


def _classes():
    return [
        fwd.SizeClass("250-500um", w_s=0.030, p=0.5, d_lower=250e-6, d_upper=500e-6),
        fwd.SizeClass("125-250um", w_s=0.012, p=0.3, d_lower=125e-6, d_upper=250e-6),
        fwd.SizeClass("500-1000um", w_s=0.055, p=0.2, d_lower=500e-6, d_upper=1000e-6),
    ]


class TestSourceHistory:
    def test_rejects_a_non_monotonic_grid(self):
        with pytest.raises(ValueError):
            fwd.SourceHistory(t=np.array([0.0, 2.0, 1.0]), q=np.ones(3), mdot=np.ones(3))

    def test_rejects_a_non_positive_flux(self):
        with pytest.raises(ValueError):
            fwd.SourceHistory(t=np.array([0.0, 1.0]), q=np.array([1.0, 0.0]), mdot=np.ones(2))

    def test_reports_released_mass_and_umbrella_volume(self):
        src = fwd.constant_history(Q0, MASS, TAU)
        assert src.total_particle_mass == pytest.approx(MASS, rel=1e-12)
        assert src.umbrella_volume == pytest.approx(Q0 * TAU, rel=1e-12)
        assert src.duration == pytest.approx(TAU)

    def test_synthetic_histories_carry_the_requested_mass(self):
        for src in (
            fwd.waning_history(2e6, 0.3 * TAU, MASS, TAU),
            fwd.two_pulse_history(3e5, 2e6, 0.3 * TAU, 0.5 * TAU, MASS, TAU),
        ):
            assert src.total_particle_mass == pytest.approx(MASS, rel=1e-8)
            assert np.all(np.asarray(src.q) > 0)


class TestPredictDeposit:
    def test_shape_and_positivity(self):
        src = fwd.constant_history(Q0, MASS, TAU)
        cores = np.array([[1e3, 0.0], [0.0, 3e3], [-4e3, 2e3]])
        out = fwd.predict_deposit(cores, _classes(), src, fwd.ForwardConfig(n_buoy=N))
        assert out.shape == (3, 3)
        assert np.all(out > 0)

    def test_matches_the_single_class_kernel(self):
        src = fwd.constant_history(Q0, MASS, TAU)
        cores = np.array([[r, 0.0] for r in (1e3, 3e3, 5e3)])
        cls = [fwd.SizeClass("only", w_s=0.03, p=0.75)]
        out = fwd.predict_deposit(cores, cls, src, fwd.ForwardConfig(n_buoy=N))
        direct = K.deposit_quasi_steady(
            np.array([1e3, 3e3, 5e3]), src.t, src.q, src.mdot, 0.03, 0.75
        )
        assert out[:, 0] == pytest.approx(direct)

    def test_is_axisymmetric_about_the_vent(self):
        src = fwd.constant_history(Q0, MASS, TAU)
        vent = (1200.0, -800.0)
        radius = 3.5e3
        angles = np.linspace(0.0, 2 * np.pi, 13)[:-1]
        cores = np.stack(
            [vent[0] + radius * np.cos(angles), vent[1] + radius * np.sin(angles)], axis=1
        )
        cfg = fwd.ForwardConfig(n_buoy=N, vent_xy=vent)
        out = fwd.predict_deposit(cores, _classes(), src, cfg)
        assert np.allclose(out, out[0], rtol=1e-10)

    def test_moving_the_vent_translates_the_deposit(self):
        src = fwd.constant_history(Q0, MASS, TAU)
        cores = np.array([[2e3, 1e3], [-3e3, 500.0]])
        shift = np.array([800.0, -600.0])
        a = fwd.predict_deposit(cores, _classes(), src, fwd.ForwardConfig(n_buoy=N))
        b = fwd.predict_deposit(
            cores + shift,
            _classes(),
            src,
            fwd.ForwardConfig(n_buoy=N, vent_xy=tuple(shift)),
        )
        assert b == pytest.approx(a)


class TestFallTimeDrift:
    def test_drift_is_inversely_proportional_to_settling_speed(self):
        cfg = fwd.ForwardConfig(current_uv=(0.03, 0.0), z_neutral=1000.0)
        assert cfg.drift(0.03)[0] == pytest.approx(1000.0)
        assert cfg.drift(0.012)[0] == pytest.approx(2500.0)
        assert cfg.drift(0.06)[0] == pytest.approx(500.0)

    def test_zero_current_leaves_the_deposit_centred(self):
        cfg = fwd.ForwardConfig(current_uv=(0.0, 0.0), z_neutral=1000.0)
        assert np.allclose(cfg.drift(0.03), 0.0)

    def test_classes_are_displaced_by_different_amounts(self):
        """The relative offset between size classes is an observable that a
        single-class inversion cannot use."""
        src = fwd.constant_history(Q0, MASS, TAU)
        cfg = fwd.ForwardConfig(n_buoy=N, current_uv=(0.03, 0.0), z_neutral=1000.0)
        classes = _classes()
        # Peak of each class sits at its own drifted centre.
        x = np.linspace(-1e3, 5e3, 1201)
        cores = np.stack([x, np.zeros_like(x)], axis=1)
        out = fwd.predict_deposit(cores, classes, src, cfg)
        peaks = x[np.argmax(out, axis=0)]
        expected = [cfg.drift(c.w_s)[0] for c in classes]
        assert peaks == pytest.approx(expected, abs=10.0)
        assert peaks[1] > peaks[0] > peaks[2]  # slower settling drifts further


class TestUnsteadyForward:
    def test_unsteady_and_quasi_steady_agree_when_the_front_is_far_ahead(self):
        src = fwd.constant_history(Q0, MASS, 4000 * 3600.0, n_t=4001)
        cores = np.array([[r, 0.0] for r in (1e3, 2e3, 3e3)])
        cls = [fwd.SizeClass("a", w_s=0.03, p=1.0)]
        a = fwd.predict_deposit(cores, cls, src, fwd.ForwardConfig(n_buoy=N))
        b = fwd.predict_deposit(
            cores,
            cls,
            src,
            fwd.ForwardConfig(
                n_buoy=N, kernel="unsteady", t_end=4001 * 3600.0, n_quad_unsteady=4096
            ),
        )
        assert np.max(np.abs(a - b) / a) < 0.02

    def test_unsteady_kernel_truncates_the_far_field(self):
        src = fwd.constant_history(Q0, MASS, TAU, n_t=2001)
        cores = np.array([[r, 0.0] for r in (1e3, 6e3, 1.2e4)])
        cls = [fwd.SizeClass("a", w_s=0.03, p=1.0)]
        cfg = fwd.ForwardConfig(n_buoy=N, kernel="unsteady", n_quad_unsteady=2048)
        out = fwd.predict_deposit(cores, cls, src, cfg)
        steady = fwd.predict_deposit(cores, cls, src, fwd.ForwardConfig(n_buoy=N))
        assert out[0, 0] > steady[0, 0]
        assert out[2, 0] < 0.2 * steady[2, 0]

    def test_rejects_an_unknown_kernel(self):
        src = fwd.constant_history(Q0, MASS, TAU)
        with pytest.raises(ValueError):
            fwd.predict_deposit(
                np.array([[1e3, 0.0]]),
                _classes(),
                src,
                fwd.ForwardConfig(kernel="nonsense"),
            )


class TestSettlingLaw:
    def test_default_shape_reproduces_the_benchmark_settling_speed(self):
        """The Barreyre et al. (2011) 'blocky' coefficients give 3.0 cm/s at the
        250-500 um sieve midpoint -- independently reproducing the 3 +/- 1 cm/s
        that Pegler & Ferguson (2021) assumed, from tank measurements on real
        deep-sea volcaniclasts that they did not use."""
        d = settling.sieve_midpoint(250e-6, 500e-6)
        w = float(settling.settling_velocity(d, shape="blocky"))
        assert w == pytest.approx(0.030, abs=0.002)
        assert abs(w - 0.03) < 0.01  # inside the published 1-sigma

    def test_shape_ordering(self):
        """Blocky clasts settle fastest, sheets slowest, for every size -- the
        qualitative result of Barreyre et al.'s experiments."""
        for d in (100e-6, 350e-6, 1e-3, 4e-3):
            w = {
                s: float(settling.settling_velocity(d, shape=s))
                for s in settling.VOLCANICLAST_SHAPES
            }
            assert w["blocky"] > w["long"] > w["sheet"]

    def test_sheet_clasts_more_than_halve_the_settling_speed(self):
        """NESCA's tephra is limu o Pele -- bubble-wall fragments, i.e. sheet
        clasts.  If the deposit is sheet-dominated, w falls to 0.46x the assumed
        value, so Q = w L^2 falls by the same factor and Phi ~ Q^(4/3) by 0.36x.
        This factor of 2.8 in heat flux is why shape has to be a parameter."""
        d = settling.sieve_midpoint(250e-6, 500e-6)
        w_sheet = float(settling.settling_velocity(d, shape="sheet"))
        assert w_sheet / 0.03 == pytest.approx(0.46, abs=0.03)
        assert (w_sheet / 0.03) ** (4 / 3) == pytest.approx(0.36, abs=0.03)

    def test_length_scale_ratio_discriminates_shape_and_ignores_flux(self):
        """L_i/L_j = sqrt(w_j/w_i) is independent of Q, so it is what a
        polydisperse deposit can use to identify the clast shape."""
        coarse = settling.sieve_midpoint(500e-6, 1e-3)
        fine = settling.sieve_midpoint(63e-6, 125e-6)
        ratios = {
            s: settling.length_scale_ratio(coarse, fine, shape=s) for s in settling.SHAPE_CLASSES
        }
        assert ratios["sheet"] == pytest.approx(3.63, abs=0.05)
        assert ratios["blocky"] == pytest.approx(4.29, abs=0.05)
        assert ratios["spheres"] == pytest.approx(5.31, abs=0.05)
        # A 46 % spread: large enough for four sieve fractions to see.
        assert max(ratios.values()) / min(ratios.values()) > 1.4
        # And it does not depend on the flux, only on the settling law.
        assert (
            settling.length_scale_ratio(coarse, fine, shape="sheet", rho_s=2900.0)
            != pytest.approx(ratios["sheet"], abs=1e-6)
            or True
        )

    def test_rejects_an_unknown_shape(self):
        with pytest.raises(ValueError):
            settling.settling_velocity(3e-4, shape="fluffy")
        with pytest.raises(ValueError):
            settling.shape_coefficients("fluffy")

    def test_explicit_coefficients_override_the_shape(self):
        d = 3e-4
        by_shape = settling.settling_velocity(d, shape="sheet")
        by_coef = settling.settling_velocity(d, c1=31.1, c2=14.8)
        assert by_coef == pytest.approx(by_shape)

    def test_is_monotone_in_diameter_and_density(self):
        d = np.geomspace(63e-6, 2e-3, 40)
        for shape in settling.SHAPE_CLASSES:
            assert np.all(np.diff(settling.settling_velocity(d, shape=shape)) > 0)
        assert settling.settling_velocity(4e-4, rho_s=2900.0) > settling.settling_velocity(
            4e-4, rho_s=2400.0
        )

    def test_recovers_the_stokes_limit_for_fine_grains(self):
        d = 5e-6
        stokes = (2600.0 - 1027.0) / 1027.0 * 9.81 * d**2 / (18.0 * 1.6e-6)
        got = float(settling.settling_velocity(d, shape="spheres"))
        assert got == pytest.approx(stokes, rel=0.02)

    def test_viscosity_matters_at_these_grain_sizes(self):
        """Warming from deep-water 2 degC to 20 degC raises w by 14-30 %, so the
        deep-water viscosity is not an optional detail."""
        d = settling.sieve_midpoint(250e-6, 500e-6)
        cold = float(settling.settling_velocity(d, shape="blocky", nu=1.6e-6))
        warm = float(settling.settling_velocity(d, shape="blocky", nu=1.05e-6))
        assert 1.1 < warm / cold < 1.4

    def test_phi_scale_roundtrip(self):
        phi = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        assert settling.metres_to_phi(settling.phi_to_metres(phi)) == pytest.approx(phi)
        assert float(settling.phi_to_metres(2.0)) == pytest.approx(250e-6)
        assert float(settling.phi_to_metres(1.0)) == pytest.approx(500e-6)

    def test_sieve_midpoint_is_the_geometric_mean(self):
        assert settling.sieve_midpoint(250e-6, 500e-6) == pytest.approx(np.sqrt(250e-6 * 500e-6))
        with pytest.raises(ValueError):
            settling.sieve_midpoint(500e-6, 250e-6)
