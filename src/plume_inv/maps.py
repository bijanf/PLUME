r"""Map-view deposit and vent-position misfit for the quasi-steady kernel.

Two helpers used to draw the deposit in plan view and to map how well each
candidate vent position explains the cores.

Deposit in plan view
--------------------
For a vent at (x_v, y_v) and a discrete measure nu on the flux nodes Q_k, the
quasi-steady deposit of size class i at (x, y) is

    Omega_i(x, y) = p_i sum_k nu_k (w_i / Q_k) exp[-pi w_i r^2 / Q_k],
    r^2 = (x - x_v)^2 + (y - y_v)^2,

which is ``kernel.deposit_from_nu`` evaluated at the radius from the vent.

Vent misfit
-----------
With the shape of nu held fixed, the log-space residual of class i at a trial
vent is

    e_ij = log Omega_ij^obs - a_i - log (G_i(vent) nu)_j ,

where the class amplitude a_i = log(p_i) absorbs both the fraction weight and
the total mass.  Minimising over a_i gives the class mean of the unamplified
residual, and minimising the Gaussian negative log likelihood over the common
noise scale gives sigma^2 = RSS / n.  The profiled negative log likelihood is
then (n / 2) log(RSS) up to a constant, and its difference from the grid
minimum is returned.
"""

from __future__ import annotations

import numpy as np

__all__ = ["deposit_xy", "vent_misfit_grid"]


def deposit_xy(x, y, vent_xy, q_nodes, nu, w_s: float, p_i: float = 1.0):
    """Quasi-steady deposit (kg m^-2) at points (x, y) for a vent at ``vent_xy``.

    ``x`` and ``y`` broadcast against each other; the result has their shape.
    ``nu`` is the vector of masses (kg) on ``q_nodes``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    q = np.asarray(q_nodes, dtype=float)
    nu = np.asarray(nu, dtype=float)
    if q.shape != nu.shape:
        raise ValueError("q_nodes and nu must have the same shape")
    r2 = (x - vent_xy[0]) ** 2 + (y - vent_xy[1]) ** 2
    g = (w_s / q) * np.exp(-np.pi * w_s * r2[..., None] / q)
    return p_i * (g @ nu)


def vent_misfit_grid(core_xy, obs, w_s, q_nodes, nu, gx, gy):
    """Profiled negative log likelihood over a grid of trial vent positions.

    Parameters
    ----------
    core_xy : list of (n_i, 2) arrays
        Core coordinates per size class (m).
    obs : list of (n_i,) arrays
        Observed deposit per size class (kg m^-2), all positive.
    w_s : list of float
        Settling speed per size class (m s^-1).
    q_nodes, nu : arrays
        Flux nodes and the fixed measure whose shape is held.
    gx, gy : 1-D arrays
        Trial vent coordinates (m).

    Returns
    -------
    dict with ``delta_nll`` of shape (len(gy), len(gx)), measured from the grid
    minimum, ``nll_min_rss`` (the residual sum of squares at the minimum),
    ``sigma_at_min``, ``n_obs`` and ``argmin_xy``.
    """
    gx = np.asarray(gx, dtype=float)
    gy = np.asarray(gy, dtype=float)
    q = np.asarray(q_nodes, dtype=float)
    nu = np.asarray(nu, dtype=float)
    vx, vy = np.meshgrid(gx, gy)
    rss = np.zeros_like(vx)
    n_obs = 0
    for xy, o, w in zip(core_xy, obs, w_s, strict=True):
        xy = np.asarray(xy, dtype=float)
        lo = np.log(np.asarray(o, dtype=float))
        n_obs += lo.size
        coef = w / q
        for a in range(gy.size):  # one grid row at a time bounds memory
            r2 = (xy[:, 0][None, :] - gx[:, None]) ** 2 + (
                xy[:, 1][None, :] - gy[a]
            ) ** 2  # (nx, n_cores)
            g = (coef * np.exp(-np.pi * w * r2[..., None] / q)) @ nu
            e = lo[None, :] - np.log(g + 1e-300)
            e = e - e.mean(axis=-1, keepdims=True)  # profile the amplitude a_i
            rss[a] += (e**2).sum(axis=-1)
    nll = 0.5 * n_obs * np.log(rss)
    k = np.unravel_index(int(np.argmin(nll)), nll.shape)
    return {
        "delta_nll": nll - nll[k],
        "rss_min": float(rss[k]),
        "sigma_at_min": float(np.sqrt(rss[k] / n_obs)),
        "n_obs": int(n_obs),
        "argmin_xy": [float(vx[k]), float(vy[k])],
    }
