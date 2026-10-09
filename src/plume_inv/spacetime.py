r"""Space-time structure of the unsteady, front-limited deposit.

:func:`plume_inv.kernel.deposit_unsteady` returns the final deposit (K6), the
time integral of a deposition rate.  This module exposes the integrand itself,
so the deposit can be followed in time as the umbrella front sweeps outwards.

Deposition rate
---------------
In the box-model umbrella the parcel labelled by the volume V' injected before
it sits at radius r with pi r^2 h(t) = V(t) - V', and carries the particle
concentration

    c(r, t) = chi(V') exp{ - w [Theta(t) - Theta(t'(V'))] },

with chi = Mdot_p / Q the loading at release and Theta(t) = int_0^t ds / h(s) the
settling clock.  Particles leave the base of the umbrella at the rate

    D(r, t) = p w c(r, t)          for r <= R_f(t),  zero beyond,          (S1)

in kg m^-2 s^-1, and int D dt over the record is (K6).

Mass on the seafloor
--------------------
Because every parcel loses particles at the same rate w / h(t), the airborne
mass obeys dA/dt = Mdot_p - (w / h) A, so

    A(t) = int_0^t Mdot_p(t') exp{ - w [Theta(t) - Theta(t')] } dt',       (S2)

and the mass on the seafloor is the released mass minus A.  Integrating (S1)
over the umbrella area with 2 pi r dr = - dV' / h gives the same total,
int D 2 pi r dr = (w / h) A, which is the consistency check used in the tests.
"""

from __future__ import annotations

import numpy as np

from .umbrella import UmbrellaSolution

__all__ = [
    "settling_clock",
    "deposition_rate_unsteady",
    "airborne_mass_unsteady",
    "deposited_fraction_unsteady",
    "unit_crossings",
]


def settling_clock(umb: UmbrellaSolution) -> np.ndarray:
    """Theta(t) = int_0^t ds / h(s) on ``umb.t``, s m^-1 (trapezoidal rule).

    The same quadrature as :func:`plume_inv.kernel.deposit_unsteady`.
    """
    t = np.asarray(umb.t, dtype=float)
    h = np.asarray(umb.thickness, dtype=float)
    return np.concatenate(([0.0], np.cumsum(np.diff(t) * 0.5 * (1.0 / h[1:] + 1.0 / h[:-1]))))


def deposition_rate_unsteady(
    r, t_obs, umb: UmbrellaSolution, t, q_t, mdot_t, w_s: float, p_i: float = 1.0
) -> np.ndarray:
    """Deposition rate (S1) of one size class on a (radius, time) grid.

    Parameters
    ----------
    r : array, shape (n_r,)
        Radii, m.
    t_obs : array, shape (n_t,)
        Times at which to evaluate the rate, s, within ``[t[0], t[-1]]``.
    umb, t, q_t, mdot_t, w_s, p_i
        As for :func:`plume_inv.kernel.deposit_unsteady`; ``t`` must equal
        ``umb.t``.

    Returns
    -------
    array, shape (n_r, n_t) -- deposition rate, kg m^-2 s^-1.  Zero wherever the
    front has not yet reached the radius.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    t_obs = np.atleast_1d(np.asarray(t_obs, dtype=float))
    t = np.asarray(t, dtype=float)
    q_t = np.asarray(q_t, dtype=float)
    mdot_t = np.asarray(mdot_t, dtype=float)
    if not np.allclose(t, umb.t):
        raise ValueError("t must be the same grid as umb.t")
    if np.any(q_t < 0) or np.any(mdot_t < 0):
        raise ValueError("Q(t) and Mdot_p(t) must be non-negative")
    if np.any(mdot_t[q_t <= 0] > 0):
        raise ValueError("Mdot_p(t) must vanish wherever Q(t) does")

    theta = settling_clock(umb)
    chi_t = np.divide(mdot_t, q_t, out=np.zeros_like(mdot_t), where=q_t > 0)
    v_mono = np.maximum.accumulate(umb.volume)

    vq = np.interp(t_obs, t, umb.volume)
    rfq = np.interp(t_obs, t, umb.front_radius)
    thq = np.interp(t_obs, t, theta)

    rr = r[:, None]
    inside = rr <= rfq[None, :]
    frac = np.minimum(rr**2 / np.maximum(rfq[None, :], 1e-12) ** 2, 1.0)
    vp = np.clip(vq[None, :] * (1.0 - frac), 0.0, v_mono[-1])
    t_rel = np.interp(vp.ravel(), v_mono, t).reshape(vp.shape)
    chi = np.interp(t_rel.ravel(), t, chi_t).reshape(vp.shape)
    theta_rel = np.interp(t_rel.ravel(), t, theta).reshape(vp.shape)
    rate = p_i * w_s * chi * np.exp(-w_s * (thq[None, :] - theta_rel))
    return np.where(inside, rate, 0.0)


def airborne_mass_unsteady(
    umb: UmbrellaSolution, t, mdot_t, w_s: float, p_i: float = 1.0
) -> np.ndarray:
    """Airborne mass A(t) of one size class, Eq. (S2), on the grid ``t``, kg.

    Integrated step by step, A_{k+1} = A_k e^{-w dTheta} + released mass of the
    step discounted by the trapezoidal rule, which stays finite however large
    w Theta becomes.
    """
    t = np.asarray(t, dtype=float)
    mdot_t = np.asarray(mdot_t, dtype=float)
    if not np.allclose(t, umb.t):
        raise ValueError("t must be the same grid as umb.t")
    theta = settling_clock(umb)
    decay = np.exp(-w_s * np.diff(theta))
    dt = np.diff(t)
    a = np.zeros_like(t)
    for k in range(t.size - 1):
        a[k + 1] = a[k] * decay[k] + 0.5 * dt[k] * (mdot_t[k] * decay[k] + mdot_t[k + 1])
    return p_i * a


def deposited_fraction_unsteady(t_obs, umb: UmbrellaSolution, t, mdot_t, w_s: float) -> np.ndarray:
    """Fraction of the class's total released mass lying on the seafloor at ``t_obs``.

    Equal to [released(t) - A(t)] / released(t_end), with A from Eq. (S2).  It
    starts at zero and approaches one minus the residual left airborne at the end
    of the record.
    """
    t = np.asarray(t, dtype=float)
    mdot_t = np.asarray(mdot_t, dtype=float)
    released = np.concatenate(([0.0], np.cumsum(0.5 * np.diff(t) * (mdot_t[1:] + mdot_t[:-1]))))
    total = released[-1]
    if total <= 0:
        raise ValueError("the history releases no particles")
    down = released - airborne_mass_unsteady(umb, t, mdot_t, w_s, 1.0)
    return np.interp(np.asarray(t_obs, dtype=float), t, down / total)


def unit_crossings(x, y, level: float = 1.0) -> list[float]:
    """Abscissae where the finite parts of y cross ``level``, by linear interpolation.

    Segments with a non-finite end point (masked values) are skipped, so a
    crossing is reported only where both neighbouring samples are defined.
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(y, dtype=float) - level
    out: list[float] = []
    for k in range(x.size - 1):
        a, b = d[k], d[k + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0.0:
            out.append(float(x[k]))
        elif a * b < 0.0:
            out.append(float(x[k] - a * (x[k + 1] - x[k]) / (b - a)))
    return out
