"""Shape-space machinery: the length-scale test, its degeneracy, and the profile test.

The two properties the shape-space results rest on are checked here on
synthetic deposits with known answers: the length-scale chi-squared depends on
(C1, C2) through C1^2/C2 alone, and a spread of the umbrella flux biases the
length-scale test while the profile test, which fits the spread, recovers the
true settling law.
"""

import numpy as np
import pytest

from plume_inv import settling, shape
from plume_inv.data import FRACTIONS

MIDS = np.array([settling.sieve_midpoint(f[1], f[2]) for f in FRACTIONS])
RNG_R = np.random.default_rng(3).uniform(300.0, 8000.0, 150)


def gaussian_deposit(r, w, q):
    return (w / q) * np.exp(-np.pi * w * r**2 / q)


class TestLengthScaleFit:
    def test_recovers_the_length_scale_of_a_gaussian_deposit(self):
        w, q = 0.02, 6.0e5
        L = shape.fit_length_scale(RNG_R, gaussian_deposit(RNG_R, w, q))
        assert L == pytest.approx(np.sqrt(q / w), rel=1e-10)

    def test_non_decaying_deposit_gives_nan(self):
        assert np.isnan(shape.fit_length_scale(RNG_R, np.exp(1e-9 * RNG_R**2)))

    def test_vectorised_bootstrap_matches_the_loop(self):
        rng = np.random.default_rng(0)
        y = np.log(gaussian_deposit(RNG_R, 0.02, 6.0e5)) + 0.5 * rng.standard_normal(RNG_R.size)
        idx = rng.integers(0, RNG_R.size, (40, RNG_R.size))
        counts = np.stack([np.bincount(i, minlength=RNG_R.size) for i in idx]).astype(float)
        L_vec, sd_vec = shape.bootstrap_log_length_sd(RNG_R, y[:, None], counts)
        loop = np.array([shape.fit_length_scale(RNG_R[i], np.exp(y[i])) for i in idx])
        assert L_vec[0] == pytest.approx(shape.fit_length_scale(RNG_R, np.exp(y)), rel=1e-9)
        assert sd_vec[0] == pytest.approx(np.nanstd(np.log(loop)), rel=1e-8)


class TestDegeneracy:
    def test_transition_diameter_balances_the_two_drag_terms(self):
        from plume_inv.constants import NU_SEAWATER, RHO_TEPHRA, RHO_W, G

        c1, c2 = 24.0, 1.2
        d = float(shape.transition_diameter(c1, c2))
        r_sub = (RHO_TEPHRA - RHO_W) / RHO_W
        assert c1 * NU_SEAWATER == pytest.approx(np.sqrt(0.75 * c2 * r_sub * G * d**3), rel=1e-12)

    def test_chi2_depends_on_c1_squared_over_c2_alone(self):
        L = np.array([18e3, 9.9e3, 5.5e3, 4.4e3])
        sd = np.array([0.4, 0.12, 0.04, 0.03])
        c1 = np.linspace(15.0, 45.0, 20)
        for k in (10.0, 45.0, 300.0):
            w = settling.settling_velocity(MIDS[None, :], c1=c1[:, None], c2=(c1**2 / k)[:, None])
            chi2, q, _ = shape.shape_chi2(L[None, :], sd[None, :], w)
            assert np.ptp(chi2) < 1e-10
            # at fixed D* every w scales as 1/C1, so the profiled flux does too
            assert np.allclose(q * c1, q[0] * c1[0], rtol=1e-10)

    def test_chi2_zero_for_the_generating_law(self):
        w = settling.settling_velocity(MIDS, shape="long")
        L = np.sqrt(7.0e5 / w)
        chi2, q, _ = shape.shape_chi2(L, np.full(4, 0.1), w)
        assert chi2 == pytest.approx(0.0, abs=1e-20)
        assert q == pytest.approx(7.0e5, rel=1e-12)


class TestProfileTest:
    MU = np.log(np.geomspace(1e5, 4e6, 121))
    S = np.sqrt(np.linspace(0.0, 0.81, 10))  # uniform in s^2
    LQN = np.linspace(np.log(1e3), np.log(1e8), 360)

    def _rss(self, radii, ys, w):
        tabs = [
            shape.log_profile_table(r, wi, self.LQN, self.MU, self.S)
            for r, wi in zip(radii, w, strict=True)
        ]
        tot = sum(shape.profile_rss(y, t) for y, t in zip(ys, tabs, strict=True))
        return shape.refined_grid_min(tot, self.S.size, self.MU.size)

    def test_single_flux_row_is_the_steady_kernel(self):
        tab = shape.log_profile_table(RNG_R, 0.02, self.LQN, self.MU, self.S)
        q = np.exp(self.MU[7])
        assert np.allclose(tab[7], np.log(gaussian_deposit(RNG_R, 0.02, q)), rtol=0, atol=1e-10)

    def test_refined_minimum_is_exact_for_a_parabola(self):
        mu = np.linspace(-1, 1, 21)
        s = np.linspace(0, 1, 5)
        rss = ((mu[None, :] - 0.137) ** 2 + (s[:, None] - 0.41) ** 2 + 2.0).reshape(-1, 1)
        best, _, _ = shape.refined_grid_min(rss, s.size, mu.size)
        assert best[0] == pytest.approx(2.0, abs=1e-12)

    def test_flux_spread_biases_the_length_scale_test_and_not_the_profile_test(self):
        """A blocky deposit from a log-normal nu with log-sd 0.35: the log-linear
        length-scale test prefers the flatter sheet law, the profile test recovers
        blocky with the right spread."""
        q_nodes = np.exp(self.LQN)
        s_true, mu_true = 0.35, np.log(9.0e5)
        nu = np.exp(-0.5 * ((self.LQN - mu_true) / s_true) ** 2)
        w_b = settling.settling_velocity(MIDS, shape="blocky")
        radii = [RNG_R] * 4
        ys = [np.log(shape.quasi_steady_deposit(RNG_R, wi, q_nodes, nu, 1.0)) for wi in w_b]
        L = np.array([shape.fit_length_scale(RNG_R, np.exp(y)) for y in ys])
        sd = np.full(4, 0.05)
        chi_len = {
            s: float(shape.shape_chi2(L, sd, settling.settling_velocity(MIDS, shape=s))[0])
            for s in ("blocky", "long", "sheet")
        }
        assert min(chi_len, key=chi_len.get) == "sheet"
        rss = {}
        for s in ("blocky", "long", "sheet"):
            best, ks, _ = self._rss(radii, ys, settling.settling_velocity(MIDS, shape=s))
            rss[s] = float(best[0])
            if s == "blocky":
                assert abs(self.S[ks[0]] - s_true) <= 0.1
        assert min(rss, key=rss.get) == "blocky"
        assert rss["blocky"] < 0.1 * min(rss["long"], rss["sheet"])

    def test_one_fraction_cannot_tell_laws_apart(self):
        """With one fraction the kernel sees w/Q alone: every law fits equally."""
        y = [np.log(gaussian_deposit(RNG_R, 0.02, 6.0e5))]
        vals = []
        for s in ("blocky", "sheet"):
            w = [float(settling.settling_velocity(MIDS[2], shape=s))]
            best, _, _ = self._rss([RNG_R], y, w)
            vals.append(float(best[0]))
        # both reach the noise-free optimum up to the grid refinement error,
        # which is small against the misfit of noisy data (N sigma^2 ~ 10^2)
        assert max(abs(v) for v in vals) < 0.02


def test_ties_are_broken_at_random():
    rng = np.random.default_rng(1)
    picks = shape.classify_by_chi2(np.zeros((3000, 3)), rng)
    frac = np.bincount(picks, minlength=3) / picks.size
    assert np.allclose(frac, 1 / 3, atol=0.03)


def test_undefined_chi2_is_a_tie():
    rng = np.random.default_rng(2)
    picks = shape.classify_by_chi2(np.full((3000, 3), np.nan), rng)
    frac = np.bincount(picks, minlength=3) / picks.size
    assert np.allclose(frac, 1 / 3, atol=0.03)
    assert shape.classify_by_chi2(np.array([[np.nan, 1.0, 2.0]]), rng)[0] == 1
