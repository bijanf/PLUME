"""Clast shape from the relative spread of dispersal length scales.

Since L_i^2 = Q / w_i for sieve fraction i, the ratio L_i / L_j = sqrt(w_j / w_i)
is independent of the umbrella flux Q, of the erupted mass and of the eruption
history.  This module holds the pieces used to score settling laws against a
set of per-fraction length scales, over the named shape classes and over the
continuous (C1, C2) plane of the Ferguson and Church law, and to simulate the
test on synthetic deposits.

Degeneracy of the ratio observable
----------------------------------
Writing the Ferguson and Church law as

    w = (R g D^2 / (C1 nu)) / (1 + (D / D*)^(3/2)),
    D* = ((C1 nu)^2 / (0.75 C2 R g))^(1/3),

every ratio w_i / w_j depends on (C1, C2) through the transition diameter D*
alone, that is through the single combination C1^2 / C2.  At fixed D* all w_i
scale as 1 / C1, so the profiled flux scales as 1 / C1 and the heat flux as
C1^(-4/3).  The deposit therefore constrains C1^2 / C2 and leaves the other
direction of the (C1, C2) plane to the prior.
"""

from __future__ import annotations

import numpy as np

from .constants import NU_SEAWATER, RHO_TEPHRA, RHO_W, G

__all__ = [
    "fit_length_scale",
    "bootstrap_length_scales",
    "transition_diameter",
    "shape_chi2",
    "slope_ols",
    "bootstrap_log_length_sd",
    "quasi_steady_deposit",
    "classify_by_chi2",
    "lognormal_node_weights",
    "log_profile_table",
    "profile_rss",
    "refined_grid_min",
]


def fit_length_scale(r, omega) -> float:
    """Length scale of one fraction from log Omega = log Omega_0 - pi r^2 / L^2.

    Ordinary least squares in log space.  Returns NaN when the fitted decay rate
    is not positive.
    """
    r = np.asarray(r, dtype=float)
    a = np.column_stack([np.ones(r.size), -np.pi * r**2])
    sol, *_ = np.linalg.lstsq(a, np.log(np.asarray(omega, dtype=float)), rcond=None)
    return (1.0 / np.sqrt(sol[1])) if sol[1] > 0 else np.nan


def bootstrap_length_scales(radii, obs, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap over cores, fraction by fraction, of :func:`fit_length_scale`.

    The loop order (draws outer, fractions inner, one generator) is fixed so
    that a given seed reproduces earlier results draw for draw.
    """
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(radii)))
    for b in range(n_boot):
        for i, (r, o) in enumerate(zip(radii, obs, strict=True)):
            idx = rng.integers(0, r.size, r.size)
            boot[b, i] = fit_length_scale(r[idx], o[idx])
    return boot


def transition_diameter(
    c1, c2, rho_s: float = RHO_TEPHRA, rho_w: float = RHO_W, nu: float = NU_SEAWATER, g: float = G
):
    """Diameter D* at which the Stokes and drag terms of the settling law are equal, m."""
    r_sub = (rho_s - rho_w) / rho_w
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    return ((c1 * nu) ** 2 / (0.75 * c2 * r_sub * g)) ** (1.0 / 3.0)


def shape_chi2(L, log_sd, w):
    """Chi-squared of per-fraction length scales against one settling law.

    Fits the single free flux Q in L_i = sqrt(Q / w_i) by weighted least squares
    in log L, with weights 1 / log_sd^2, and returns the residual sum.

    Parameters
    ----------
    L, log_sd : (..., n) arrays
        Length scales and the standard deviation of their logarithms.  A NaN in
        ``L`` or ``log_sd`` drops that fraction.
    w : (..., n) array
        Settling speeds of the fractions under the candidate law; broadcasts
        against ``L``.

    Returns
    -------
    chi2, q_hat, resid : arrays
        ``chi2`` and ``q_hat`` have the broadcast shape without the last axis.
    """
    L = np.asarray(L, dtype=float)
    log_sd = np.asarray(log_sd, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(L) & np.isfinite(log_sd) & (log_sd > 0)
    weights = np.where(ok, 1.0 / np.where(ok, log_sd, 1.0) ** 2, 0.0)
    log_l = np.where(ok, np.log(np.where(ok, L, 1.0)), 0.0)
    rs = log_l + 0.5 * np.log(w)
    half_log_q = np.sum(weights * rs, axis=-1) / np.sum(weights, axis=-1)
    resid = rs - half_log_q[..., None]
    chi2 = np.sum(weights * resid**2, axis=-1)
    return chi2, np.exp(2.0 * half_log_q), resid


def slope_ols(x, y, counts=None):
    """Least-squares slope of y on x, optionally with integer resampling counts.

    Parameters
    ----------
    x : (n,) array
    y : (n, R) array, R independent data sets sharing the abscissa.
    counts : (B, n) array or None
        Resampling multiplicities (a bootstrap).  With counts the result has
        shape (B, R); without, shape (R,).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if counts is None:
        counts = np.ones((1, x.size))
        squeeze = True
    else:
        counts = np.asarray(counts, dtype=float)
        squeeze = False
    xm = x.mean()
    xc = x - xm
    s0 = counts.sum(axis=1)[:, None]
    sx = (counts @ xc)[:, None]
    sxx = (counts @ xc**2)[:, None]
    sy = counts @ y
    sxy = counts @ (xc[:, None] * y)
    den = s0 * sxx - sx**2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (s0 * sxy - sx * sy) / den
    return b[0] if squeeze else b


def bootstrap_log_length_sd(r, log_omega, counts):
    """Point length scales and bootstrap sd of log L, for R data sets at once.

    ``r`` is (n,), ``log_omega`` is (n, R) and ``counts`` is (B, n).  The fit is
    that of :func:`fit_length_scale`; a non-positive decay rate gives NaN and is
    dropped from the bootstrap standard deviation, as in the single-set version.
    Returns (L, log_sd), each of shape (R,).
    """
    x = -np.pi * (np.asarray(r, dtype=float) / 1e3) ** 2  # km^2, for conditioning
    b0 = slope_ols(x, log_omega)
    bb = slope_ols(x, log_omega, counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        L = np.where(b0 > 0, 1e3 / np.sqrt(np.where(b0 > 0, b0, 1.0)), np.nan)
        log_lb = np.where(bb > 0, np.log(1e3 / np.sqrt(np.where(bb > 0, bb, 1.0))), np.nan)
    n_ok = np.sum(np.isfinite(log_lb), axis=0)
    with np.errstate(invalid="ignore"):
        mu = np.nansum(log_lb, axis=0) / np.maximum(n_ok, 1)
        var = np.nansum((log_lb - mu) ** 2, axis=0) / np.maximum(n_ok, 1)
    log_sd = np.where(n_ok > 1, np.sqrt(var), np.nan)
    return L, log_sd


def quasi_steady_deposit(r, w, q_nodes, nu_weights, mass):
    """Deposit per unit area of one fraction under the quasi-steady kernel, kg m^-2.

    Omega(r) = mass * sum_k nu_k (w / Q_k) exp(-pi w r^2 / Q_k), with ``nu_weights``
    normalised to unit sum; the same kernel as the inversion's likelihood.
    """
    r = np.asarray(r, dtype=float)
    q = np.asarray(q_nodes, dtype=float)
    nu = np.asarray(nu_weights, dtype=float)
    nu = nu / nu.sum()
    g = (w / q)[None, :] * np.exp(-np.pi * w * r[:, None] ** 2 / q[None, :])
    return mass * (g @ nu)


def classify_by_chi2(chi2, rng=None):
    """Index of the minimum chi-squared along the last axis, ties broken at random.

    With one sieve fraction every candidate fits exactly and all chi-squared
    values are zero; random tie-breaking then returns chance-level accuracy.  An
    undefined (NaN) chi-squared carries no preference and is treated as a tie.
    """
    chi2 = np.asarray(chi2, dtype=float)
    chi2 = np.where(np.isnan(chi2), np.inf, chi2)
    rng = np.random.default_rng(0) if rng is None else rng
    lo = np.min(chi2, axis=-1, keepdims=True)
    tie = chi2 <= lo + 1e-9 * np.maximum(1.0, np.abs(lo))
    noise = rng.random(chi2.shape)
    return np.argmax(np.where(tie, noise, -1.0), axis=-1)


# --------------------------------------------------------------------------
# Profile-likelihood classifier with a log-normal flux distribution
# --------------------------------------------------------------------------
# A spread of the umbrella flux makes each fraction's deposit a mixture of
# Gaussians in r, whose log-profile is convex in r^2.  A log-linear fit over a
# fixed set of radii then returns a length scale that depends on how far into
# the tail the cores reach, which differs between fractions, and the ratio test
# above is biased.  The classifier below fits the quasi-steady kernel itself,
# with nu(Q) log-normal in Q (log-mean mu, log-sd s, s = 0 a single flux), one
# free amplitude per fraction, and the same (mu, s) shared by every fraction.
# The per-fraction deposits are then the same function of w_i r^2 / Q, which is
# the exact content of the time-relabeling theorem, and the misfit scores the
# settling law with the flux spread profiled out.

LOG_FLOOR = -700.0


def lognormal_node_weights(log_q_nodes, mu, s):
    """Normalised weights of a log-normal nu on log-spaced nodes: (len(mu)*len(s), nodes).

    Rows run over mu fastest-varying last: index = i_s * len(mu) + i_mu.  Rows with
    s = 0 are returned as NaN and must be handled by the exact single-flux kernel.
    """
    lq = np.asarray(log_q_nodes, dtype=float)
    mu = np.asarray(mu, dtype=float)
    s = np.asarray(s, dtype=float)
    out = np.full((s.size, mu.size, lq.size), np.nan)
    for a, sa in enumerate(s):
        if sa <= 0:
            continue
        z = (lq[None, :] - mu[:, None]) / sa
        e = np.exp(-0.5 * z**2)
        out[a] = e / e.sum(axis=1, keepdims=True)
    return out.reshape(s.size * mu.size, lq.size)


def log_profile_table(r, w, log_q_nodes, mu, s, node_weights=None):
    """log F(r_j; w, mu, s) for every grid point: (len(s)*len(mu), len(r)).

    F = sum_k nu_k (w / Q_k) exp(-pi w r^2 / Q_k), the quasi-steady kernel under a
    log-normal nu; for s = 0 the single-flux kernel at Q = exp(mu) is used.
    """
    r2 = np.asarray(r, dtype=float) ** 2
    lq = np.asarray(log_q_nodes, dtype=float)
    mu = np.asarray(mu, dtype=float)
    s = np.asarray(s, dtype=float)
    if node_weights is None:
        node_weights = lognormal_node_weights(lq, mu, s)
    q = np.exp(lq)
    kern = (w / q)[None, :] * np.exp(-np.pi * w * r2[:, None] / q[None, :])  # (n, nodes)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(node_weights @ kern.T)  # (G, n)
    for a, sa in enumerate(s):
        if sa <= 0:
            qq = np.exp(mu)
            out[a * mu.size : (a + 1) * mu.size] = (
                np.log(w / qq)[:, None] - np.pi * w * r2[None, :] / qq[:, None]
            )
    # Rows whose kernel underflows are fits far from any optimum; a finite floor
    # keeps their misfit large and finite.
    return np.maximum(out, LOG_FLOOR)


def profile_rss(log_obs, table):
    """Residual sum of squares with a free additive offset, for every grid row.

    ``log_obs`` is (n, R) and ``table`` is (G, n); returns (G, R).
    """
    y = np.asarray(log_obs, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    n = y.shape[0]
    cross = table @ y
    ff = np.sum(table**2, axis=1)[:, None]
    fs = np.sum(table, axis=1)[:, None]
    yy = np.sum(y**2, axis=0)[None, :]
    ys = np.sum(y, axis=0)[None, :]
    return yy - 2.0 * cross + ff - (ys - fs) ** 2 / n


def refined_grid_min(rss, n_s: int, n_mu: int):
    """Minimum over a (s, mu) grid with parabolic refinement along mu, then s.

    ``rss`` is (n_s * n_mu, R) with mu fastest-varying.  Both axes must be
    uniformly spaced in the variable the misfit is close to parabolic in (log Q
    for mu; s^2 for the log-sd, since the spread enters log F at order s^2).
    Returns (minimum, index of best s, index of best mu), each of shape (R,).
    Interior minima are refined by a three-point parabola, which removes most
    of the discretisation error of a finite grid; minima on an edge keep the
    grid value.
    """
    a = np.asarray(rss, dtype=float).reshape(n_s, n_mu, -1)
    j = np.argmin(a, axis=1)  # (n_s, R)
    jc = np.clip(j, 1, n_mu - 2)
    y0 = np.take_along_axis(a, (jc - 1)[:, None, :], axis=1)[:, 0]
    y1 = np.take_along_axis(a, jc[:, None, :], axis=1)[:, 0]
    y2 = np.take_along_axis(a, (jc + 1)[:, None, :], axis=1)[:, 0]
    raw = np.take_along_axis(a, j[:, None, :], axis=1)[:, 0]
    den = y0 - 2.0 * y1 + y2
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = np.where((den > 0) & (j == jc), y1 - (y2 - y0) ** 2 / (8.0 * den), raw)
    per_s = np.minimum(ref, raw)  # (n_s, R)
    k = np.argmin(per_s, axis=0)  # (R,)
    if n_s >= 3:
        kc = np.clip(k, 1, n_s - 2)
        cols = np.arange(per_s.shape[1])
        z0, z1, z2 = per_s[kc - 1, cols], per_s[kc, cols], per_s[kc + 1, cols]
        zr = per_s[k, cols]
        dd = z0 - 2.0 * z1 + z2
        with np.errstate(invalid="ignore", divide="ignore"):
            best = np.where((dd > 0) & (k == kc), z1 - (z2 - z0) ** 2 / (8.0 * dd), zr)
        best = np.minimum(best, zr)
    else:
        best = per_s.min(axis=0)
    return best, k, j[k, np.arange(per_s.shape[1])]
