"""Space-time deposition rate of the front-limited kernel (plume_inv.spacetime)."""

import numpy as np
import pytest

from plume_inv import forward as fwd
from plume_inv import kernel as K
from plume_inv import spacetime as S
from plume_inv import umbrella as U

Q0 = 7.6e5  # m^3 s^-1
W = 0.03  # m s^-1
MASS = 1.0e9  # kg
TAU = 15 * 3600.0  # s
N = 1.0e-3  # s^-1
LAM = 0.2


@pytest.fixture(scope="module")
def record():
    src = fwd.constant_history(Q0, MASS, TAU, n_t=1501)
    t_end = K.settling_window(TAU, Q0, N, W, LAM)
    t, q, m = K.extend_history(src.t, src.q, src.mdot, t_end, n_extra=1500)
    umb = U.solve_umbrella(lambda tt: float(np.interp(tt, t, q)), t, N, LAM)
    return t, q, m, umb


def test_time_integral_of_rate_is_the_front_limited_deposit(record):
    t, q, m, umb = record
    r = np.array([500.0, 2000.0, 4000.0, 6000.0, 8000.0])
    t_obs = np.linspace(0.0, t[-1], 8000)
    rate = S.deposition_rate_unsteady(r, t_obs, umb, t, q, m, W)
    integral = np.trapezoid(rate, t_obs, axis=1)
    kernel = K.deposit_unsteady(r, umb, t, q, m, W, 1.0, n_quad=4096)
    np.testing.assert_allclose(integral, kernel, rtol=3e-3)


def test_rate_vanishes_ahead_of_the_front(record):
    t, q, m, umb = record
    t_obs = np.array([3600.0, 5 * 3600.0, 10 * 3600.0])
    r_f = np.interp(t_obs, t, umb.front_radius)
    r = np.array([0.5, 0.99, 1.01, 1.5]) * r_f[1]
    rate = S.deposition_rate_unsteady(r, t_obs, umb, t, q, m, W)
    assert np.all(rate[r[:, None] > r_f[None, :]] == 0.0)
    assert np.all(rate[r[:, None] < r_f[None, :]] > 0.0)


def test_area_integral_of_rate_equals_settling_of_airborne_mass(record):
    """int D 2 pi r dr = (w / h) A(t), because 2 pi r dr = -dV' / h."""
    t, q, m, umb = record
    airborne = S.airborne_mass_unsteady(umb, t, m, W)
    for t0 in (4 * 3600.0, 12 * 3600.0, 20 * 3600.0):
        r_f = float(np.interp(t0, t, umb.front_radius))
        r = np.linspace(0.0, r_f, 4001)
        d = S.deposition_rate_unsteady(r, [t0], umb, t, q, m, W)[:, 0]
        lhs = np.trapezoid(2 * np.pi * r * d, r)
        rhs = W / float(np.interp(t0, t, umb.thickness)) * float(np.interp(t0, t, airborne))
        assert lhs == pytest.approx(rhs, rel=5e-3)


def test_deposited_fraction_is_monotone_and_complete(record):
    t, q, m, umb = record
    f = S.deposited_fraction_unsteady(t, umb, t, m, W)
    assert f[0] == 0.0
    assert np.all(np.diff(f) >= -1e-12)
    # The settling window leaves at most 1e-5 of the slowest class airborne.
    assert f[-1] == pytest.approx(1.0, abs=2e-5)


def test_deposited_fraction_matches_space_time_integral(record):
    t, q, m, umb = record
    t_obs = np.linspace(0.0, TAU, 3000)
    r = np.linspace(0.0, float(np.interp(TAU, t, umb.front_radius)), 1500)
    rate = S.deposition_rate_unsteady(r, t_obs, umb, t, q, m, W)
    on_floor = np.trapezoid(2 * np.pi * r * np.trapezoid(rate, t_obs, axis=1), r) / MASS
    f = float(S.deposited_fraction_unsteady(TAU, umb, t, m, W))
    assert on_floor == pytest.approx(f, rel=5e-3)


def test_unit_crossings_interpolate_and_skip_masked_segments():
    x = np.linspace(0.0, 4.0, 401)
    y = 1.0 + 0.5 * np.cos(np.pi * x / 2.0)  # crosses 1 at x = 1 and x = 3
    got = S.unit_crossings(x, y)
    np.testing.assert_allclose(got, [1.0, 3.0], atol=1e-6)
    y_masked = y.copy()
    y_masked[90:111] = np.nan
    np.testing.assert_allclose(S.unit_crossings(x, y_masked), [3.0], atol=1e-6)
    assert S.unit_crossings([0.0, 1.0], [0.5, 1.5]) == [pytest.approx(0.5)]
