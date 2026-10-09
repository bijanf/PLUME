"""Deposition kernels: Gaussian limit, mass conservation, Theorem 1, nu(Q)."""

import numpy as np
import pytest

from plume_inv import forward as fwd
from plume_inv import kernel as K
from plume_inv import umbrella as U

Q0 = 7.6e5  # m^3 s^-1
W = 0.03  # m s^-1
MASS = 1.0e9  # kg
TAU = 15 * 3600.0  # s
N = 1.0e-3  # s^-1
LAM = 0.2
L = np.sqrt(Q0 / W)


class TestGaussianLimit:
    def test_steady_monodisperse_kernel_is_the_benchmark_gaussian(self):
        """Machine-precision agreement with Omega_0 exp[-pi (r/L)^2]."""
        r = np.linspace(0.0, 3 * L, 401)
        src = fwd.constant_history(Q0, MASS, TAU, n_t=4001)
        num = K.deposit_quasi_steady(r, src.t, src.q, src.mdot, W, 1.0)
        ana = MASS * (W / Q0) * np.exp(-np.pi * (r / L) ** 2)
        assert np.max(np.abs(num - ana)) / ana.max() < 1e-13

    def test_peak_value_and_length_scale(self):
        r = np.array([0.0, L])
        dep = K.gaussian_deposit(r, MASS, Q0, W)
        assert dep[0] == pytest.approx(MASS / L**2)
        assert dep[1] / dep[0] == pytest.approx(np.exp(-np.pi))

    def test_fraction_of_mass_inside_L(self):
        """Exactly 1 - exp(-pi) = 0.9568 of the mass falls within r < L.

        Pegler & Ferguson (2021) quote "~93 %"; the exact value for the stated
        kernel is 95.68 %.  Recorded here so that the right number is quoted.
        """
        r = np.linspace(0.0, 60 * L, 400001)
        dep = K.gaussian_deposit(r, MASS, Q0, W)
        cum = np.cumsum(np.gradient(r) * 2 * np.pi * r * dep)
        inside = np.interp(L, r, cum) / MASS
        assert inside == pytest.approx(1.0 - np.exp(-np.pi), rel=1e-4)
        assert inside == pytest.approx(0.9568, abs=1e-3)


class TestMassConservation:
    @staticmethod
    def _deposited_mass(t, q, mdot, w, n_s=3000):
        """int Omega 2 pi r dr, evaluated on s = r^2.

        On s the kernel is a pure exponential with decay length Q/(pi w), which
        spans the whole range of Q in the history.  A geometric s-grid resolves
        every decay length with the same relative spacing, so one modest grid is
        accurate for fast- and slow-decaying contributions alike -- and it keeps
        the (r, t) outer product small enough to be fast.
        """
        q = np.asarray(q, dtype=float)
        s_max = 40.0 * float(q.max()) / (np.pi * w)
        s_min = 1e-6 * float(q.min()) / (np.pi * w)
        s = np.concatenate(([0.0], np.geomspace(s_min, s_max, n_s)))
        dep = K.deposit_quasi_steady(np.sqrt(s), t, q, mdot, w, 1.0)
        return float(np.pi * np.trapezoid(dep, s))

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_arbitrary_histories_conserve_mass(self, seed):
        rng = np.random.default_rng(seed)
        t = np.linspace(0.0, TAU, 4001)
        q = (
            Q0
            * np.exp(rng.uniform(-1.0, 1.0) * np.sin(3 * np.pi * t / TAU))
            * (0.3 + 0.7 * np.exp(-t / (0.6 * TAU)))
        )
        mdot = 1e5 * (1.0 + 0.5 * np.cos(5 * np.pi * t / TAU) ** 2)
        released = np.trapezoid(mdot, t)
        assert self._deposited_mass(t, q, mdot, W) == pytest.approx(released, rel=1e-5)

    @pytest.mark.parametrize("w", [0.005, 0.03, 0.08])
    def test_conservation_holds_for_every_size_class(self, w):
        src = fwd.waning_history(2e6, 0.3 * TAU, MASS, TAU, n_t=4001)
        got = self._deposited_mass(src.t, src.q, src.mdot, w)
        assert got == pytest.approx(MASS, rel=1e-5)

    def test_mass_fraction_scales_the_deposit(self):
        src = fwd.constant_history(Q0, MASS, TAU)
        r = np.linspace(0.0, 2 * L, 101)
        full = K.deposit_quasi_steady(r, src.t, src.q, src.mdot, W, 1.0)
        third = K.deposit_quasi_steady(r, src.t, src.q, src.mdot, W, 1.0 / 3.0)
        assert third == pytest.approx(full / 3.0)


class TestTheorem1:
    """The deposit depends on (Q, Mdot_p) only through nu."""

    def test_time_reversal_gives_an_identical_deposit(self):
        t = np.linspace(0.0, TAU, 8001)
        q = Q0 * (0.5 + 1.0 * t / TAU)
        mdot = np.full_like(t, 1e5)
        r = np.linspace(0.0, 3 * L, 201)
        d_forward = K.deposit_quasi_steady(r, t, q, mdot, W, 1.0)
        d_reverse = K.deposit_quasi_steady(r, t, q[::-1].copy(), mdot, W, 1.0)
        assert np.max(np.abs(d_forward - d_reverse)) / d_forward.max() < 1e-13

    def test_two_pulses_in_either_order_give_an_identical_deposit(self):
        """A weak-then-strong and a strong-then-weak history with the same nu.

        The pulse is deliberately placed OFF-CENTRE so that the history is not
        time-symmetric: an earlier version centred it, which made the reversed
        history identical to the original and the test vacuous.  Here the two
        histories are genuinely different functions of time (they differ by 39 %
        of the peak, asserted below) that induce the same nu.
        """
        t = np.linspace(0.0, TAU, 8001)
        edge = 0.02 * TAU
        window = 0.5 * (np.tanh((t - 0.15 * TAU) / edge) - np.tanh((t - 0.45 * TAU) / edge))
        a = Q0 * (1.0 + 1.5 * window)
        b = a[::-1].copy()
        # Guard: the test is only meaningful if the two histories really differ.
        assert np.max(np.abs(a - b)) / a.max() > 0.3
        mdot = np.full_like(t, 1e5)
        r = np.linspace(0.0, 3 * L, 201)
        da = K.deposit_quasi_steady(r, t, a, mdot, W, 1.0)
        db = K.deposit_quasi_steady(r, t, b, mdot, W, 1.0)
        assert np.max(np.abs(da - db)) / da.max() < 1e-13

    def test_discontinuous_histories_agree_as_the_quadrature_refines(self):
        """For a discontinuous Q the invariance is exact in the continuum, but
        the trapezoidal rule straddles each jump and resolves it only to first
        order.  Refining the grid must drive the residual to zero -- which is
        what separates a quadrature artefact from a violation of the theorem.

        The two histories are a CYCLIC PERMUTATION of the same three blocks, not
        a time reversal.  Reversal is a symmetry of a uniform grid, so the two
        quadratures would be identical term by term and the residual would be
        pure round-off, measuring nothing; a cyclic shift puts the jumps at
        genuinely different positions within their cells.  Both histories spend
        the same time at each flux level with the same constant release rate, so
        they induce the same nu.
        """
        r = np.linspace(0.0, 3 * L, 201)
        levels = np.array([1.0, 2.5, 0.6]) * Q0
        widths = np.array([0.2, 0.3, 0.5])  # fractions of TAU

        def build(t, order):
            edges = np.concatenate(([0.0], np.cumsum(widths[list(order)]))) * TAU
            q = np.empty_like(t)
            for k, idx in enumerate(order):
                sel = (t >= edges[k]) & (t < edges[k + 1])
                q[sel] = levels[idx]
            q[t >= edges[-1]] = levels[order[-1]]
            return q

        errs = []
        for n in (1001, 4001, 16001):
            t = np.linspace(0.0, TAU, n)
            a = build(t, (0, 1, 2))
            b = build(t, (1, 2, 0))  # same nu, cyclically shifted
            mdot = np.full_like(t, 1e5)
            da = K.deposit_quasi_steady(r, t, a, mdot, W, 1.0)
            db = K.deposit_quasi_steady(r, t, b, mdot, W, 1.0)
            errs.append(np.max(np.abs(da - db)) / da.max())
        assert (
            np.max(
                np.abs(
                    build(np.linspace(0, TAU, 1001), (0, 1, 2))
                    - build(np.linspace(0, TAU, 1001), (1, 2, 0))
                )
            )
            > 0
        )
        # A real, resolvable quadrature error -- not round-off -- that converges.
        assert errs[0] > 1e-9, "nothing to converge: the two quadratures coincide"
        assert errs[-1] < errs[0] / 4.0
        assert errs[-1] < 5e-5

    def test_invariance_holds_for_every_size_class_simultaneously(self):
        t = np.linspace(0.0, TAU, 8001)
        q = Q0 * (0.5 + 1.5 * (t / TAU) ** 2)
        mdot = 1e5 * (1.0 + t / TAU)
        r = np.linspace(0.0, 3 * L, 101)
        # A relabelling that preserves nu: reverse BOTH Q and Mdot together.
        for w in (0.005, 0.012, 0.03, 0.08):
            d1 = K.deposit_quasi_steady(r, t, q, mdot, w, 1.0)
            d2 = K.deposit_quasi_steady(r, t, q[::-1].copy(), mdot[::-1].copy(), w, 1.0)
            assert np.max(np.abs(d1 - d2)) / d1.max() < 1e-12

    def test_different_nu_gives_a_different_deposit(self):
        """The invariance must not be vacuous: changing nu must change Omega."""
        t = np.linspace(0.0, TAU, 8001)
        r = np.linspace(0.0, 3 * L, 201)
        flat = K.deposit_quasi_steady(r, t, np.full_like(t, Q0), np.full_like(t, 1e5), W, 1.0)
        spread = K.deposit_quasi_steady(
            r, t, Q0 * np.where(t < TAU / 2, 0.4, 1.6), np.full_like(t, 1e5), W, 1.0
        )
        assert np.max(np.abs(flat - spread)) / flat.max() > 0.05


class TestNuRepresentation:
    def test_nu_binning_conserves_mass(self):
        t = np.linspace(0.0, TAU, 8001)
        q = Q0 * (0.5 + 1.0 * t / TAU)
        mdot = 1e5 * (1.0 + 0.3 * np.sin(2 * np.pi * t / TAU))
        edges = np.geomspace(q.min() * 0.999, q.max() * 1.001, 201)
        nu = K.nu_from_history(t, q, mdot, edges)
        assert nu.sum() == pytest.approx(np.trapezoid(mdot, t), rel=1e-6)

    def test_deposit_from_nu_matches_the_time_integral(self):
        t = np.linspace(0.0, TAU, 12001)
        q = Q0 * (0.5 + 1.0 * t / TAU)
        mdot = 1e5 * (1.0 + 0.3 * np.sin(2 * np.pi * t / TAU))
        edges = np.geomspace(q.min() * 0.999, q.max() * 1.001, 401)
        nodes = np.sqrt(edges[1:] * edges[:-1])
        nu = K.nu_from_history(t, q, mdot, edges)
        r = np.linspace(0.0, 3 * L, 201)
        direct = K.deposit_quasi_steady(r, t, q, mdot, W, 1.0)
        via_nu = K.deposit_from_nu(r, nodes, nu, W, 1.0)
        assert np.max(np.abs(direct - via_nu)) / direct.max() < 1e-5

    def test_design_matrix_reproduces_deposit_from_nu(self):
        nodes = np.geomspace(1e5, 3e6, 33)
        nu = np.exp(-((np.log(nodes / 7.6e5)) ** 2) / 0.5) * 1e8
        r = np.linspace(0.0, 3 * L, 51)
        a = K.design_matrix(r, nodes, W, 0.4)
        assert a.shape == (51, 33)
        assert a @ nu == pytest.approx(K.deposit_from_nu(r, nodes, nu, W, 0.4))

    def test_laplace_abscissa_is_pi_w_r_squared(self):
        """Pin the actual formula, not just its proportionality in w.

        The previous version only checked that doubling w doubles the abscissa,
        which any prefactor and any power of r would satisfy.
        """
        r = np.array([1e3, 5e3])
        assert K.laplace_abscissa(0.03, r) == pytest.approx(np.pi * 0.03 * r**2)
        assert float(K.laplace_abscissa(1.0 / np.pi, 1.0)) == pytest.approx(1.0)

    def test_the_kernel_really_is_that_laplace_transform(self):
        """Omega_i(r)/(p_i w_i) must equal L[x nu~](pi w_i r^2), i.e. the same
        transform of the same measure for every size class, sampled at a
        class-dependent abscissa.  Two classes evaluated at radii chosen so that
        their abscissae coincide must therefore give the same value."""
        nodes = np.geomspace(1e5, 3e6, 61)
        nu = np.exp(-((np.log(nodes / 7.6e5)) ** 2) / 0.5) * 1e8
        w_a, w_b = 0.012, 0.048
        r_a = np.array([2.0e3, 4.0e3, 6.0e3])
        r_b = r_a * np.sqrt(w_a / w_b)  # same pi w r^2
        assert K.laplace_abscissa(w_a, r_a) == pytest.approx(K.laplace_abscissa(w_b, r_b))
        da = K.deposit_from_nu(r_a, nodes, nu, w_a, 1.0) / w_a
        db = K.deposit_from_nu(r_b, nodes, nu, w_b, 1.0) / w_b
        assert da == pytest.approx(db, rel=1e-12)


class TestUnsteadyKernel:
    def test_front_law_matches_the_closed_form(self):
        t = np.linspace(0.0, 20 * 3600.0, 4001)
        umb = U.solve_umbrella(lambda _t: Q0, t, N, LAM)
        analytic = U.front_radius_constant_q(t[1:], Q0, N, LAM)
        assert np.max(np.abs(umb.front_radius[1:] - analytic) / analytic) < 1e-8

    def test_front_positions_are_of_the_expected_size(self):
        """R_f ~ 1 km at 1 h, ~4.5 km at 10 h, ~7 km at 20 h for the benchmark Q."""
        for hours, expected in ((1.0, 980.0), (10.0, 4550.0), (20.0, 7220.0)):
            got = float(U.front_radius_constant_q(hours * 3600.0, Q0, N, LAM))
            assert got == pytest.approx(expected, rel=0.02)

    def test_converges_to_the_gaussian_kernel_for_constant_q_and_long_times(self):
        """With Q constant and the record long enough for the front to outrun
        the deposit, the unsteady kernel must reproduce the Gaussian kernel."""
        tau_long = 5000 * 3600.0
        src = fwd.constant_history(Q0, MASS, tau_long, n_t=6001)
        umb = U.solve_umbrella(lambda _t: Q0, src.t, N, LAM)
        r = np.array([1e3, 2e3, 3e3, 4e3])
        unsteady = K.deposit_unsteady(r, umb, src.t, src.q, src.mdot, W, 1.0, n_quad=8192)
        steady = K.deposit_quasi_steady(r, src.t, src.q, src.mdot, W, 1.0)
        assert np.max(np.abs(unsteady - steady) / steady) < 0.01

    def test_conserves_mass_over_the_full_settling_window(self):
        src = fwd.constant_history(Q0, MASS, TAU, n_t=3001)
        t_end = K.settling_window(TAU, Q0, N, W, LAM)
        t, q, mdot = K.extend_history(src.t, src.q, src.mdot, t_end, n_extra=3000)
        umb = U.solve_umbrella(lambda tt: float(np.interp(tt, t, q)), t, N, LAM)
        r = np.linspace(1.0, umb.max_front_radius * 0.9999, 4000)
        dep = K.deposit_unsteady(r, umb, t, q, mdot, W, 1.0, n_quad=4096)
        deposited = np.trapezoid(2 * np.pi * r * dep, r)
        assert deposited / MASS == pytest.approx(1.0, abs=2e-3)

    def test_deposit_vanishes_beyond_the_front(self):
        src = fwd.constant_history(Q0, MASS, TAU, n_t=3001)
        umb = U.solve_umbrella(lambda _t: Q0, src.t, N, LAM)
        r_max = umb.max_front_radius
        r = np.array([r_max * 1.01, r_max * 1.5, r_max * 3.0])
        dep = K.deposit_unsteady(r, umb, src.t, src.q, src.mdot, W, 1.0)
        assert np.all(dep == 0.0)

    def test_box_model_thickness_is_defined_by_volume_and_front_radius(self):
        """V = pi R_f^2 h is the DEFINITION of h in the box model, so this only
        checks that the solver keeps its own invariant.  It is not evidence that
        the deposit extent measures umbrella volume: that apparent bound is
        vacuous, and the real bound is the weaker V >= pi R^2 h_min.
        """
        src = fwd.constant_history(Q0, MASS, TAU, n_t=3001)
        umb = U.solve_umbrella(lambda _t: Q0, src.t, N, LAM)
        assert np.pi * umb.front_radius**2 * umb.thickness == pytest.approx(umb.volume, rel=1e-10)

    def test_the_front_keeps_advancing_after_the_source_stops(self):
        """With V fixed the ODE gives R_f ~ t^(1/3) and h ~ t^(-2/3), so there
        is no final front radius -- which is why the deposit has no hard outer
        edge that could be read as a volume measurement."""
        src = fwd.constant_history(Q0, MASS, TAU, n_t=2001)
        t = np.linspace(0.0, 40 * TAU, 8001)
        umb = U.solve_umbrella(lambda tt: Q0 if tt <= TAU else 0.0, t, N, LAM)
        late = t > 10 * TAU
        # Fit a power law to the late-time front: R_f ~ t^p with p -> 1/3.
        p_fit = np.polyfit(np.log(t[late]), np.log(umb.front_radius[late]), 1)[0]
        assert p_fit == pytest.approx(1.0 / 3.0, abs=0.02)
        assert umb.front_radius[-1] > 2.0 * umb.front_radius[t.searchsorted(2 * TAU)]
        assert umb.volume[-1] == pytest.approx(Q0 * TAU, rel=1e-3)
        assert src.umbrella_volume == pytest.approx(Q0 * TAU, rel=1e-12)

    def test_quadrature_converges(self):
        src = fwd.constant_history(Q0, MASS, TAU, n_t=3001)
        umb = U.solve_umbrella(lambda _t: Q0, src.t, N, LAM)
        r = np.array([1e3, 2e3, 3e3, 4e3])
        ref = K.deposit_unsteady(r, umb, src.t, src.q, src.mdot, W, 1.0, n_quad=16384)
        errs = [
            np.max(
                np.abs(K.deposit_unsteady(r, umb, src.t, src.q, src.mdot, W, 1.0, n_quad=n) - ref)
                / ref
            )
            for n in (512, 2048)
        ]
        assert errs[1] < errs[0] / 3.0
        assert errs[1] < 1e-5

    def test_unsteady_kernel_is_steeper_than_the_gaussian_at_eruption_timescales(self):
        """The physical claim: when the front takes as long as
        the eruption to cross the deposit, the quasi-steady kernel over-predicts
        the far field and under-predicts the near field."""
        src = fwd.constant_history(Q0, MASS, TAU, n_t=3001)
        t_end = K.settling_window(TAU, Q0, N, W, LAM)
        t, q, mdot = K.extend_history(src.t, src.q, src.mdot, t_end, n_extra=2000)
        umb = U.solve_umbrella(lambda tt: float(np.interp(tt, t, q)), t, N, LAM)
        r = np.array([1e3, 5e3, 7e3])
        unsteady = K.deposit_unsteady(r, umb, t, q, mdot, W, 1.0, n_quad=8192)
        steady = K.deposit_quasi_steady(r, src.t, src.q, src.mdot, W, 1.0)
        ratio = unsteady / steady
        assert ratio[0] > 1.05  # near field enhanced
        assert ratio[2] < 0.7  # far field suppressed
        assert ratio[0] > ratio[1] > ratio[2]


class TestEnergyFromNu:
    """energy_from_nu is the only code path to the paper's headline energy."""

    PREF_ARGS = dict(n_buoy=N, epsilon=0.1, k_heat=2.0938e9, c_f=0.187)

    def _pref(self):
        a = self.PREF_ARGS
        return a["k_heat"] * a["c_f"] * (a["n_buoy"] ** 5 / a["epsilon"] ** 2) ** (1 / 3)

    def test_matches_the_closed_form_for_a_point_mass(self):
        """For nu = M delta_{Q0} and a constant release rate, E = pref * M * Q0^(4/3)."""
        nodes = np.array([Q0])
        nu = np.array([MASS])
        got = K.energy_from_nu(nodes, nu, **self.PREF_ARGS)
        assert got == pytest.approx(self._pref() * MASS * Q0 ** (4 / 3))

    def test_source_link_m_equals_four_thirds_erases_the_shape_of_nu(self):
        """With Mdot_p = c Q^(4/3) the exponent 4/3 - m is zero, so E depends on
        nu only through its total mass.  This degeneracy is a result, not a bug:
        with that link the deposit constrains total energy but not its time
        structure."""
        c = 3.0
        link = lambda q: c * q ** (4 / 3)  # noqa: E731
        concentrated = (np.array([Q0]), np.array([MASS]))
        spread = (np.array([0.25 * Q0, 4.0 * Q0]), np.array([0.5 * MASS, 0.5 * MASS]))
        e_a = K.energy_from_nu(*concentrated, mdot_over_q=link, **self.PREF_ARGS)
        e_b = K.energy_from_nu(*spread, mdot_over_q=link, **self.PREF_ARGS)
        assert e_a == pytest.approx(e_b, rel=1e-12)
        assert e_a == pytest.approx(self._pref() * MASS / c)

    def test_source_link_m_equals_one_makes_spreading_lower_the_energy(self):
        """With Mdot_p = c Q the exponent is 1/3 and Q -> Q^(1/3) is CONCAVE, so
        by Jensen a spread nu gives LESS energy than a concentrated one at the
        same mean."""
        c = 3.0
        link = lambda q: c * q  # noqa: E731
        mean_q = Q0
        concentrated = (np.array([mean_q]), np.array([MASS]))
        # Same nu-mean of Q, spread symmetrically.
        spread = (np.array([0.25 * mean_q, 1.75 * mean_q]), np.array([0.5 * MASS, 0.5 * MASS]))
        assert np.average(spread[0], weights=spread[1]) == pytest.approx(mean_q)
        e_a = K.energy_from_nu(*concentrated, mdot_over_q=link, **self.PREF_ARGS)
        e_b = K.energy_from_nu(*spread, mdot_over_q=link, **self.PREF_ARGS)
        assert e_b < e_a

    def test_rejects_a_non_positive_source_link(self):
        with pytest.raises(ValueError):
            K.energy_from_nu(
                np.array([Q0]), np.array([MASS]), mdot_over_q=lambda q: 0.0 * q, **self.PREF_ARGS
            )


class TestUnsteadyKernelScaling:
    """Properties of deposit_unsteady that the limit tests do not pin down."""

    @staticmethod
    def _setup(n_t=2001):
        src = fwd.constant_history(Q0, MASS, TAU, n_t=n_t)
        t_end = K.settling_window(TAU, Q0, N, W, LAM)
        t, q, mdot = K.extend_history(src.t, src.q, src.mdot, t_end, n_extra=1500)
        umb = U.solve_umbrella(lambda tt: float(np.interp(tt, t, q)), t, N, LAM)
        return umb, t, q, mdot

    def test_mass_fraction_scales_the_unsteady_deposit_linearly(self):
        """The p_i factor was previously exercised only at p_i = 1."""
        umb, t, q, mdot = self._setup()
        r = np.array([1e3, 3e3, 5e3])
        full = K.deposit_unsteady(r, umb, t, q, mdot, W, 1.0, n_quad=1024)
        part = K.deposit_unsteady(r, umb, t, q, mdot, W, 0.37, n_quad=1024)
        assert part == pytest.approx(0.37 * full, rel=1e-12)
        assert np.all(full > 0)

    def test_released_mass_scales_the_unsteady_deposit_linearly(self):
        umb, t, q, mdot = self._setup()
        r = np.array([1e3, 3e3, 5e3])
        a = K.deposit_unsteady(r, umb, t, q, mdot, W, 1.0, n_quad=1024)
        b = K.deposit_unsteady(r, umb, t, q, 2.5 * mdot, W, 1.0, n_quad=1024)
        assert b == pytest.approx(2.5 * a, rel=1e-12)

    def test_slower_classes_travel_further(self):
        umb, t, q, mdot = self._setup()
        r = np.array([6e3])
        slow = K.deposit_unsteady(r, umb, t, q, mdot, 0.012, 1.0, n_quad=2048)
        fast = K.deposit_unsteady(r, umb, t, q, mdot, 0.055, 1.0, n_quad=2048)
        assert slow[0] > fast[0]

    def test_rejects_release_where_the_flux_vanishes(self):
        umb, t, q, mdot = self._setup()
        bad = mdot.copy()
        bad[-1] = 1.0  # release after the source shut off
        with pytest.raises(ValueError):
            K.deposit_unsteady(np.array([1e3]), umb, t, q, bad, W, 1.0)

    def test_settling_window_grows_when_the_class_settles_more_slowly(self):
        """The residual criterion must actually bind."""
        fast = K.settling_window(TAU, Q0, N, 0.08, LAM)
        slow = K.settling_window(TAU, Q0, N, 0.004, LAM)
        assert slow > fast

    def test_extend_history_preserves_the_erupted_volume(self):
        src = fwd.constant_history(Q0, MASS, TAU, n_t=2001)
        t, q, mdot = K.extend_history(src.t, src.q, src.mdot, 4 * TAU, n_extra=1500)
        assert float(np.trapezoid(q, t)) == pytest.approx(Q0 * TAU, rel=1e-6)
        assert float(np.trapezoid(mdot, t)) == pytest.approx(MASS, rel=1e-6)
        with pytest.raises(ValueError):
            K.extend_history(src.t, src.q, src.mdot, 0.5 * TAU)
