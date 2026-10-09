"""Plan-view deposit and the vent misfit surface (``plume_inv.maps``).

The plan-view deposit must be the radial kernel evaluated at the distance from
the vent, and must conserve mass over the plane.  The misfit surface must reach
its minimum at the vent that generated noise-free data, and must be blind to a
constant rescaling of any one class, since the class amplitude is profiled.
"""

import numpy as np
import pytest

from plume_inv import kernel as K
from plume_inv.maps import deposit_xy, vent_misfit_grid

Q_NODES = np.geomspace(3e4, 3e7, 12)
NU = np.exp(-0.5 * ((np.log(Q_NODES) - np.log(1e6)) / 0.4) ** 2) * 1e6
VENT = (400.0, -700.0)


def test_deposit_xy_matches_radial_kernel():
    rng = np.random.default_rng(3)
    x = rng.uniform(-5000, 5000, 50)
    y = rng.uniform(-5000, 5000, 50)
    r = np.hypot(x - VENT[0], y - VENT[1])
    got = deposit_xy(x, y, VENT, Q_NODES, NU, 0.01, 0.3)
    want = K.deposit_from_nu(r, Q_NODES, NU, 0.01, 0.3)
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_deposit_xy_conserves_mass():
    # Narrow nu so the deposit fits inside the integration window.
    nu = np.zeros_like(Q_NODES)
    nu[3] = 5.0e6
    w = 0.03
    h = 20.0
    g = np.arange(-8000.0, 8000.0 + h, h)
    X, Y = np.meshgrid(g, g)
    om = deposit_xy(X, Y, VENT, Q_NODES, nu, w, 0.25)
    mass = om.sum() * h * h
    assert mass == pytest.approx(0.25 * nu.sum(), rel=1e-6)


def test_deposit_xy_rejects_mismatched_nu():
    with pytest.raises(ValueError):
        deposit_xy(0.0, 0.0, VENT, Q_NODES, NU[:-1], 0.01)


def _synthetic():
    rng = np.random.default_rng(7)
    xy = rng.uniform(-5000, 5000, size=(80, 2))
    w_s = [0.004, 0.012, 0.03]
    p = [0.2, 0.3, 0.5]
    obs = [
        deposit_xy(xy[:, 0], xy[:, 1], VENT, Q_NODES, NU, w, pi)
        for w, pi in zip(w_s, p, strict=True)
    ]
    return [xy] * 3, obs, w_s


def test_misfit_minimum_at_true_vent():
    core_xy, obs, w_s = _synthetic()
    gx = np.arange(-1000.0, 1801.0, 100.0)
    gy = np.arange(-2000.0, 601.0, 100.0)
    out = vent_misfit_grid(core_xy, obs, w_s, Q_NODES, NU, gx, gy)
    assert out["argmin_xy"] == pytest.approx(list(VENT), abs=1e-9)
    assert out["delta_nll"].min() == 0.0
    assert np.all(out["delta_nll"] >= 0.0)
    assert out["delta_nll"].shape == (gy.size, gx.size)
    assert out["n_obs"] == 240


def test_misfit_blind_to_class_amplitude():
    core_xy, obs, w_s = _synthetic()
    rng = np.random.default_rng(11)
    noisy = [o * np.exp(rng.normal(0, 0.3, o.size)) for o in obs]
    scaled = [noisy[0] * 17.0, noisy[1], noisy[2] * 0.02]
    gx = np.linspace(-1500, 1500, 7)
    gy = np.linspace(-2000, 1000, 6)
    a = vent_misfit_grid(core_xy, noisy, w_s, Q_NODES, NU, gx, gy)
    b = vent_misfit_grid(core_xy, scaled, w_s, Q_NODES, NU, gx, gy)
    np.testing.assert_allclose(a["delta_nll"], b["delta_nll"], atol=1e-9)
    assert a["sigma_at_min"] == pytest.approx(b["sigma_at_min"], rel=1e-12)
