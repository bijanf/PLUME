"""Umbrella-cloud dynamics.

Two descriptions are provided.

**Quasi-steady.**  The umbrella has already spread past every observation point
and carries the instantaneous source flux, 2 pi r h u = Q(t).  This is the
assumption behind the Gaussian deposit of Pegler & Ferguson (2021).

**Unsteady (front-limited).**  The umbrella is a radial intrusion of finite
extent fed by Q(t).  In a box (uniform-thickness) approximation, volume
conservation and an inertia-buoyancy front condition give

    dV/dt   = Q(t)
    h(t)    = V(t) / (pi R_f(t)^2)
    dR_f/dt = lambda N h(t) = lambda N V(t) / (pi R_f(t)^2),

one ODE system with a single new parameter lambda.  For constant Q this
integrates in closed form to

    R_f(t) = (3 lambda N Q t^2 / (2 pi))^(1/3),

which is the constant-flux intrusion law attributed to Costa, Folch &
Macedonio (2013).  Writing the front condition rather than the
closed form is what lets the model handle an unsteady source and the
post-eruption phase, where V is constant and the intrusion keeps spreading as
R_f ~ t^(1/3).

The consequences for the deposit are derived in the paper (Methods) and
implemented in :mod:`plume_inv.kernel`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

from .constants import LAMBDA_FRONT

__all__ = [
    "front_radius_constant_q",
    "thickness_constant_q",
    "UmbrellaSolution",
    "solve_umbrella",
]


def front_radius_constant_q(t, q_umb: float, n_buoy: float, lam: float = LAMBDA_FRONT):
    """Closed-form intrusion front radius for a constant source flux, m.

    R_f(t) = (3 lambda N Q t^2 / (2 pi))^(1/3)
    """
    t = np.asarray(t, dtype=float)
    return np.cbrt(3.0 * lam * n_buoy * q_umb * t**2 / (2.0 * np.pi))


def thickness_constant_q(t, q_umb: float, n_buoy: float, lam: float = LAMBDA_FRONT):
    """Box-model umbrella thickness for a constant source flux, m.

    h = V / (pi R_f^2) with V = Q t; thins as t^(-1/3).
    """
    t = np.asarray(t, dtype=float)
    r_f = front_radius_constant_q(t, q_umb, n_buoy, lam)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = q_umb * t / (np.pi * r_f**2)
    return h


@dataclass(frozen=True)
class UmbrellaSolution:
    """Time histories of the umbrella: injected volume, front radius, thickness.

    ``volume``, ``front_radius`` and ``thickness`` are evaluated on ``t``.  The
    interpolators are monotone (PCHIP) so that inverting ``volume`` and
    ``front_radius`` is safe.
    """

    t: np.ndarray
    volume: np.ndarray
    front_radius: np.ndarray
    thickness: np.ndarray
    n_buoy: float
    lam: float

    def volume_at(self, t):
        return PchipInterpolator(self.t, self.volume, extrapolate=True)(t)

    def front_radius_at(self, t):
        return PchipInterpolator(self.t, self.front_radius, extrapolate=True)(t)

    def thickness_at(self, t):
        return PchipInterpolator(self.t, self.thickness, extrapolate=True)(t)

    def time_front_reaches(self, r):
        """First time at which the front radius equals r (NaN if it never does)."""
        r = np.asarray(r, dtype=float)
        out = np.interp(r, self.front_radius, self.t, left=self.t[0], right=np.nan)
        return out

    @property
    def max_front_radius(self) -> float:
        return float(self.front_radius[-1])

    @property
    def total_volume(self) -> float:
        return float(self.volume[-1])


def solve_umbrella(
    q_func: Callable[[float], float],
    t_eval: np.ndarray,
    n_buoy: float,
    lam: float = LAMBDA_FRONT,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> UmbrellaSolution:
    """Integrate the box-model umbrella for an arbitrary source history Q(t).

    Parameters
    ----------
    q_func : callable
        Source volume flux Q(t), m^3 s^-1.  Must be non-negative.
    t_eval : array
        Strictly increasing times at which to report, s.  ``t_eval[0]`` is the
        eruption start and must be >= 0.
    n_buoy : float
        Ambient buoyancy frequency, s^-1.
    lam : float
        Front coefficient in dR/dt = lambda N h.

    Notes
    -----
    The system is singular at t = 0 (V = R_f = 0 with h = 0/0).  Integration
    starts from the constant-flux similarity solution evaluated at a small
    ``t0``, using Q(t0); the solution is insensitive to ``t0`` because the
    similarity solution is exact as t -> 0 for any Q that is continuous and
    positive there.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.ndim != 1 or t_eval.size < 2:
        raise ValueError("t_eval must be a 1-D array with at least two entries")
    if np.any(np.diff(t_eval) <= 0):
        raise ValueError("t_eval must be strictly increasing")
    if t_eval[0] < 0:
        raise ValueError("t_eval must start at or after t = 0")

    t_end = float(t_eval[-1])
    t0 = max(1e-9 * t_end, 1e-6)
    q0 = float(q_func(t0))
    if q0 <= 0:
        raise ValueError("Q(t) must be positive at the start of the eruption")

    v0 = q0 * t0
    r0 = float(front_radius_constant_q(t0, q0, n_buoy, lam))

    def rhs(t, y):
        v, r = y
        q = max(float(q_func(t)), 0.0)
        r = max(r, 1e-12)
        return [q, lam * n_buoy * v / (np.pi * r**2)]

    sol = solve_ivp(
        rhs,
        (t0, t_end),
        [v0, r0],
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"umbrella integration failed: {sol.message}")

    t_clip = np.clip(t_eval, t0, t_end)
    v, r = sol.sol(t_clip)
    v = np.maximum(v, 0.0)
    r = np.maximum(r, 1e-12)
    h = v / (np.pi * r**2)

    return UmbrellaSolution(
        t=np.asarray(t_eval, dtype=float),
        volume=v,
        front_radius=r,
        thickness=h,
        n_buoy=n_buoy,
        lam=lam,
    )
