r"""Deposition kernels and the nu(Q) representation of the source.

Quasi-steady kernel
-------------------
For size class i with settling speed w_i, particle mass fraction p_i, particle
release rate Mdot_p(t) and umbrella flux Q(t),

    Omega_i(r) = int_0^tau Mdot_p(t) p_i (w_i / Q(t))
                 exp[- pi w_i r^2 / Q(t)] dt.                            (K1)

The kernel conserves mass exactly: int_0^inf Omega_i 2 pi r dr
= p_i int_0^tau Mdot_p dt, because int_0^inf 2 pi r exp(-a r^2) dr = pi / a.

nu(Q) representation (Theorem 1)
--------------------------------
(K1) depends on t only through Q(t), so with

    nu(B) := int_{ t : Q(t) in B } Mdot_p(t) dt                          (K2)

-- the pushforward of the released-mass measure under the map t -> Q(t) -- we
have, for every size class simultaneously,

    Omega_i(r) = p_i int_0^inf (w_i / Q) exp[- pi w_i r^2 / Q] nu(dQ).   (K3)

Two source histories with the same nu produce identical deposits.  nu, not the
ordering of Q(t), is what a deposit identifies (paper, Methods).

Laplace structure
-----------------
Substituting x = 1/Q, (K3) is

    Omega_i(r) / (p_i w_i) = int_0^inf x exp[- pi w_i r^2 x] nu~(dx)
                           = L[ x nu~ ](pi w_i r^2),                     (K4)

a Laplace transform of one fixed measure, sampled at abscissa pi w_i r^2.  Every
size class samples the *same* transform at a class-dependent scaling of r^2:
adding size classes widens the sampled abscissa range by w_max / w_min, which is
the precise sense in which polydispersity acts as a filter.  Inverting a
Laplace transform is exponentially ill-posed, so the singular values of the
discretised operator decay geometrically and only a finite number of modes of
nu is resolvable.

Unsteady, front-limited kernel
------------------------------
When the umbrella front is still advancing across the observation radii, (K1)
over-predicts the deposit at large r.  Labelling each umbrella parcel by the
volume V' injected before it, a box-model umbrella of thickness h(t) places that
parcel at radius r with

    pi r^2 h(t) = V(t) - V',   i.e.   r^2 = R_f(t)^2 (1 - V'/V(t)),      (K5)

and the parcel loses particles at rate w_i / h.  Bookkeeping (paper, Methods) gives

    Omega_i(r) = p_i w_i int_{t : R_f(t) >= r} chi(V'(r,t))
                 exp{ - w_i [Theta(t) - Theta(t'(V'(r,t)))] } dt,        (K6)

with chi(V') = Mdot_p(t')/Q(t') the particle loading per unit umbrella volume
and Theta(t) = int_0^t ds / h(s) the settling clock.  For constant Q and
constant h, (K6) reduces to (K1) exactly, minus the transit-time truncation
pi h r^2 / Q; the deposit therefore vanishes beyond

    R_max^2 = V_total / (pi h),                                          (K7)

which turns the observed maximum deposit radius into a lower bound on the
umbrella volume.
"""

from __future__ import annotations

import numpy as np

from .umbrella import UmbrellaSolution

__all__ = [
    "kernel_quasi_steady",
    "gaussian_deposit",
    "deposit_quasi_steady",
    "laplace_abscissa",
    "nu_from_history",
    "deposit_from_nu",
    "design_matrix",
    "deposit_unsteady",
    "settling_window",
    "extend_history",
    "energy_from_nu",
]


# --------------------------------------------------------------------------
# Quasi-steady (Gaussian) kernel
# --------------------------------------------------------------------------
def kernel_quasi_steady(r, q_umb, w_s):
    """g(r, Q) = (w / Q) exp[-pi w r^2 / Q]   [m^-2].

    Broadcasting-friendly: pass r and q_umb with shapes that broadcast.
    """
    r = np.asarray(r, dtype=float)
    q = np.asarray(q_umb, dtype=float)
    w = np.asarray(w_s, dtype=float)
    return (w / q) * np.exp(-np.pi * w * r**2 / q)


def gaussian_deposit(r, mass: float, q_umb: float, w_s: float):
    """Steady, monodisperse deposit  Omega = (M w / Q) exp[-pi (r/L)^2], L^2 = Q/w.

    ``mass`` is the total mass of this size class released, kg.  The result is
    kg m^-2.
    """
    return float(mass) * kernel_quasi_steady(r, q_umb, w_s)


def deposit_quasi_steady(r, t, q_t, mdot_t, w_s, p_i: float = 1.0):
    """Evaluate (K1) by trapezoidal quadrature on the supplied time grid.

    Parameters
    ----------
    r : array, shape (n_r,)
        Radii, m.
    t : array, shape (n_t,)
        Time grid, s, strictly increasing.
    q_t : array, shape (n_t,)
        Umbrella volume flux Q(t), m^3 s^-1, strictly positive.
    mdot_t : array, shape (n_t,)
        Particle release rate Mdot_p(t), kg s^-1, non-negative.
    w_s : float
        Settling speed of this class, m s^-1.
    p_i : float
        Mass fraction of this class.

    Returns
    -------
    array, shape (n_r,) -- deposit mass per unit area, kg m^-2.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    t = np.asarray(t, dtype=float)
    q_t = np.asarray(q_t, dtype=float)
    mdot_t = np.asarray(mdot_t, dtype=float)
    if not (t.shape == q_t.shape == mdot_t.shape):
        raise ValueError("t, q_t and mdot_t must have the same shape")
    if np.any(q_t <= 0):
        raise ValueError("Q(t) must be strictly positive")

    integrand = mdot_t[None, :] * p_i * kernel_quasi_steady(r[:, None], q_t[None, :], w_s)
    return np.trapezoid(integrand, t, axis=1)


def laplace_abscissa(w_s, r):
    """The Laplace abscissa lambda = pi w r^2 at which a class-i core samples (K4)."""
    return np.pi * np.asarray(w_s, dtype=float) * np.asarray(r, dtype=float) ** 2


# --------------------------------------------------------------------------
# nu(Q) representation
# --------------------------------------------------------------------------
def nu_from_history(t, q_t, mdot_t, q_edges):
    """Bin a source history into the measure nu of (K2).

    Returns
    -------
    nu : array, shape (len(q_edges) - 1,)
        Released particle mass, kg, falling in each Q bin.

    Notes
    -----
    Binning is exact in the limit of a fine time grid: the contribution of each
    time step is assigned to the bin containing the step-midpoint Q.  The total
    is conserved to the accuracy of the midpoint rule.
    """
    t = np.asarray(t, dtype=float)
    q_t = np.asarray(q_t, dtype=float)
    mdot_t = np.asarray(mdot_t, dtype=float)
    q_edges = np.asarray(q_edges, dtype=float)
    if np.any(np.diff(q_edges) <= 0):
        raise ValueError("q_edges must be strictly increasing")

    dt = np.diff(t)
    q_mid = 0.5 * (q_t[1:] + q_t[:-1])
    m_step = 0.5 * (mdot_t[1:] + mdot_t[:-1]) * dt
    idx = np.digitize(q_mid, q_edges) - 1
    nu = np.zeros(q_edges.size - 1)
    inside = (idx >= 0) & (idx < nu.size)
    np.add.at(nu, idx[inside], m_step[inside])
    return nu


def deposit_from_nu(r, q_nodes, nu, w_s, p_i: float = 1.0):
    """Evaluate (K3) for a discrete nu supported on ``q_nodes``.

    ``nu`` is a vector of masses (kg), not a density.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    q_nodes = np.asarray(q_nodes, dtype=float)
    nu = np.asarray(nu, dtype=float)
    if q_nodes.shape != nu.shape:
        raise ValueError("q_nodes and nu must have the same shape")
    g = kernel_quasi_steady(r[:, None], q_nodes[None, :], w_s)
    return p_i * (g @ nu)


def design_matrix(r, q_nodes, w_s, p_i: float = 1.0):
    """Forward operator A with Omega = A @ nu for one size class.

    Shape (len(r), len(q_nodes)); units m^-2, so A @ nu is kg m^-2.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    q_nodes = np.asarray(q_nodes, dtype=float)
    return p_i * kernel_quasi_steady(r[:, None], q_nodes[None, :], w_s)


def energy_from_nu(
    q_nodes, nu, n_buoy, epsilon, k_heat, c_f, mdot_over_q=None, exponent: float = 4.0 / 3.0
):
    r"""Total thermal energy implied by nu under the source link Mdot_p = c Q^m.

    With F0 = c_f (N^5 Q^4 / eps^2)^(1/3) and Phi = k F0, the released energy is

        E = int Phi(t) dt = k c_f (N^5/eps^2)^(1/3) int Q^(4/3) dt,

    and under Mdot_p = c Q^m the time measure is  dt = nu(dQ) / (c Q^m), so

        E = [k c_f (N^5/eps^2)^(1/3) / c] int Q^(4/3 - m) nu(dQ).

    Parameters
    ----------
    mdot_over_q : callable or None
        The function Q -> Mdot_p(Q) = c Q^m.  If None, a constant release rate
        is assumed (m = 0, c = Mdot_p) and the caller must scale the result.
    exponent : float
        The exponent of Q in Phi(Q); 4/3 for a point source.

    Returns
    -------
    float -- energy in J, up to the normalisation set by ``mdot_over_q``.
    """
    q_nodes = np.asarray(q_nodes, dtype=float)
    nu = np.asarray(nu, dtype=float)
    pref = k_heat * c_f * (n_buoy**5 / epsilon**2) ** (1.0 / 3.0)
    if mdot_over_q is None:
        weight = q_nodes**exponent
        return float(pref * np.sum(weight * nu))
    m_of_q = np.asarray(mdot_over_q(q_nodes), dtype=float)
    if np.any(m_of_q <= 0):
        raise ValueError("mdot_over_q must be strictly positive on q_nodes")
    return float(pref * np.sum(q_nodes**exponent / m_of_q * nu))


# --------------------------------------------------------------------------
# Unsteady, front-limited kernel
# --------------------------------------------------------------------------
def deposit_unsteady(
    r,
    umb: UmbrellaSolution,
    t,
    q_t,
    mdot_t,
    w_s,
    p_i: float = 1.0,
    n_quad: int = 1024,
):
    """Evaluate the front-limited kernel (K6).

    Parameters
    ----------
    r : array
        Radii, m.
    umb : UmbrellaSolution
        Umbrella history from :func:`plume_inv.umbrella.solve_umbrella`, on the
        same grid as ``t``.
    t, q_t, mdot_t : arrays
        Source history on the grid ``umb.t``.  The grid should extend well past
        the end of the eruption so that the particles still airborne when the
        source shuts off are allowed to land; use :func:`settling_window` to
        choose the end time and :func:`extend_history` to pad the history.
        ``q_t`` may be zero after the eruption, provided ``mdot_t`` is zero
        there too.
    w_s : float
        Settling speed, m s^-1.
    p_i : float
        Mass fraction of this class.
    n_quad : int
        Number of quadrature nodes in the per-radius time integral.

    Returns
    -------
    array, shape (n_r,) -- deposit, kg m^-2.

    Notes
    -----
    The integral for each radius runs from the time the front reaches that
    radius to the end of the record, on its own uniform grid, so the left
    endpoint -- where the integrand switches on -- is resolved exactly.  Radii
    beyond the final front position receive nothing.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    t = np.asarray(t, dtype=float)
    q_t = np.asarray(q_t, dtype=float)
    mdot_t = np.asarray(mdot_t, dtype=float)
    if not np.allclose(t, umb.t):
        raise ValueError("t must be the same grid as umb.t")
    if np.any(q_t < 0) or np.any(mdot_t < 0):
        raise ValueError("Q(t) and Mdot_p(t) must be non-negative")
    if np.any(mdot_t[q_t <= 0] > 0):
        raise ValueError("Mdot_p(t) must vanish wherever Q(t) does")

    v = umb.volume
    r_f = umb.front_radius
    h = umb.thickness

    # Settling clock Theta(t) = int_0^t ds / h(s).
    theta = np.concatenate(([0.0], np.cumsum(np.diff(t) * 0.5 * (1.0 / h[1:] + 1.0 / h[:-1]))))
    # Particle loading per unit umbrella volume, kg m^-3; zero where Q = 0.
    chi_t = np.divide(mdot_t, q_t, out=np.zeros_like(mdot_t), where=q_t > 0)
    v_mono = np.maximum.accumulate(v)

    t_arrive = np.interp(r, r_f, t, left=t[0], right=np.nan)
    out = np.zeros_like(r)

    for j, rj in enumerate(r):
        ta = t_arrive[j]
        if not np.isfinite(ta) or ta >= t[-1]:
            continue  # front never reached this radius
        tq = np.linspace(ta, t[-1], n_quad)
        vq = np.interp(tq, t, v)
        rfq = np.interp(tq, t, r_f)
        thq = np.interp(tq, t, theta)
        # Parcel label currently sitting at radius rj.
        vp = vq * (1.0 - np.minimum(rj**2 / np.maximum(rfq, 1e-12) ** 2, 1.0))
        vp = np.clip(vp, 0.0, v_mono[-1])
        t_rel = np.interp(vp, v_mono, t)  # release time of that parcel
        chi = np.interp(t_rel, t, chi_t)
        theta_rel = np.interp(t_rel, t, theta)
        integrand = chi * np.exp(-w_s * (thq - theta_rel))
        out[j] = p_i * w_s * np.trapezoid(integrand, tq)

    return out


def settling_window(
    t_eruption_end: float,
    q_umb_typical: float,
    n_buoy: float,
    w_s_min: float,
    lam: float,
    residual: float = 1e-5,
    max_factor: float = 200.0,
    growth: float = 1.25,
) -> float:
    """End time of the record needed to let (almost) all particles land.

    After the source shuts off, the intrusion keeps spreading (V constant,
    R_f ~ t^(1/3), h ~ t^(-2/3)), so the settling clock Theta(t) = int ds/h
    grows as t^(5/3) and the airborne fraction exp(-w Theta) goes to zero.  This
    returns the smallest multiple of ``t_eruption_end`` for which the slowest
    class has at most ``residual`` of its mass still airborne.

    Choosing this too small is not a harmless approximation.  The particles left
    airborne are exactly the ones that would have landed furthest out, so a
    truncated record biases the FAR FIELD -- where the deposit is smallest and
    the front-limitation signal lives -- while barely moving the total mass.
    Truncating at a residual of 2.4e-4 changes the deposit at 8 km by 10 %,
    which is why the default is tight and the step is 1.25 rather than 2.
    """
    from .umbrella import solve_umbrella  # local import: avoids a cycle

    if not 0.0 < residual < 1.0:
        raise ValueError("residual must lie in (0, 1)")
    n_efold = -np.log(residual)
    factor = 2.0
    while factor <= max_factor:
        t_end = factor * t_eruption_end
        grid = np.linspace(0.0, t_end, 6001)
        umb = solve_umbrella(
            lambda tt: q_umb_typical if tt <= t_eruption_end else 0.0,
            grid,
            n_buoy,
            lam,
        )
        theta_end = np.trapezoid(1.0 / umb.thickness[1:], grid[1:])
        if w_s_min * theta_end >= n_efold:
            return t_end
        factor *= growth
    return max_factor * t_eruption_end


def extend_history(t, q_t, mdot_t, t_end: float, n_extra: int = 2000):
    """Pad a source history with a quiescent tail Q = 0, Mdot_p = 0.

    Returns ``(t_ext, q_ext, mdot_ext)`` on a grid running to ``t_end``.  The
    shut-off is represented by a single repeated time so the quadrature sees a
    step rather than a ramp, which keeps int Q dt equal to the eruption volume.
    """
    t = np.asarray(t, dtype=float)
    q_t = np.asarray(q_t, dtype=float)
    mdot_t = np.asarray(mdot_t, dtype=float)
    if t_end <= t[-1]:
        raise ValueError("t_end must exceed the end of the eruption")
    eps = 1e-9 * (t[-1] - t[0])
    tail = np.linspace(t[-1] + eps, t_end, n_extra)
    t_ext = np.concatenate([t, tail])
    q_ext = np.concatenate([q_t, np.zeros_like(tail)])
    m_ext = np.concatenate([mdot_t, np.zeros_like(tail)])
    return t_ext, q_ext, m_ext
