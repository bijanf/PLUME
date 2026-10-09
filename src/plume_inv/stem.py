"""Plume-stem closures: from buoyancy flux to umbrella volume flux and back.

Two routes are provided and cross-checked against each other:

1. *Closures* -- the algebraic relations used by Pegler & Ferguson (2021),
   valid for a maintained point (axisymmetric) or line (planar) source in a
   uniformly stratified, quiescent ambient.
2. *Direct integration* of the Morton-Taylor-Turner (1956) plume equations,
   which fixes the numerical coefficients rather than assuming them.

The MTT system for a Boussinesq top-hat plume from a point source in a linearly
stratified ambient, in terms of the volume, momentum and buoyancy fluxes
Q = pi b^2 w, M = pi b^2 w^2, F = pi b^2 w g' is

    dQ/dz = 2 eps sqrt(pi M)
    dM/dz = F Q / M
    dF/dz = -N^2 Q

with F(0) = F0, Q(0) = M(0) = 0 for an ideal point source.  The neutral level
z_n is where F = 0; the top of rise z_max is where M = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .constants import (
    ALPHA_T,
    C_F_PLANAR,
    C_F_POINT,
    C_P,
    C_Q_POINT,
    C_TRANSITION,
    EPSILON_ENTRAIN,
    RHO_W,
    G,
)

__all__ = [
    "heat_flux_constant",
    "heat_flux",
    "buoyancy_flux_from_heat",
    "q_umb_point",
    "f0_point",
    "q_umb_planar",
    "f0_planar",
    "transition_length",
    "gaussian_length_scale",
    "StemSolution",
    "solve_stem_mtt",
    "PlanarStemSolution",
    "solve_stem_mtt_planar",
    "solve_stem_mtt_profile",
    "planar_rise_height",
    "point_rise_height",
    "fissure_length_for_rise",
    "C_ZMAX_PLANAR",
    "C_ZMAX_POINT",
    "C_ZNEUTRAL_POINT",
]

# Universal MTT rise-height coefficients, obtained by direct integration and
# verified in tests/test_stem.py::TestRiseHeightCoefficients to be independent
# of F0, N and eps: the relative spread over F0 in [1e1, 1e4], N in
# [5e-4, 2e-3] and eps in [0.05, 0.15] is below 1e-10 for all three.
C_ZMAX_POINT = 1.3661222
"""z_max = C_ZMAX_POINT * (F0 / (eps^2 N^3))^(1/4), axisymmetric source."""

C_ZNEUTRAL_POINT = 1.0377636
"""z_n = C_ZNEUTRAL_POINT * (F0 / (eps^2 N^3))^(1/4), axisymmetric source."""

C_ZMAX_PLANAR = 1.6609899
"""z_max = C_ZMAX_PLANAR * (f0 / (eps N^3))^(1/3) per unit length, line source.

Beware: quoting the coefficient against (f0/N^3)^(1/3) instead hides an
eps^(-1/3) and is correct only at the eps it was evaluated for.
"""


# --------------------------------------------------------------------------
# Heat flux
# --------------------------------------------------------------------------
def heat_flux_constant(
    rho: float = RHO_W, c_p: float = C_P, alpha: float = ALPHA_T, g: float = G
) -> float:
    """k = rho c_p / (alpha g), in W s^3 m^-4 (equivalently kg m^-2)."""
    return rho * c_p / (alpha * g)


def heat_flux(f0, rho: float = RHO_W, c_p: float = C_P, alpha: float = ALPHA_T, g: float = G):
    """Seafloor heat flux Phi = k F0 (W) from the buoyancy flux F0 (m^4 s^-3)."""
    return heat_flux_constant(rho, c_p, alpha, g) * np.asarray(f0, dtype=float)


def buoyancy_flux_from_heat(
    phi, rho: float = RHO_W, c_p: float = C_P, alpha: float = ALPHA_T, g: float = G
):
    """Inverse of :func:`heat_flux`: F0 = Phi / k (m^4 s^-3)."""
    return np.asarray(phi, dtype=float) / heat_flux_constant(rho, c_p, alpha, g)


# --------------------------------------------------------------------------
# Point (axisymmetric) source closure
# --------------------------------------------------------------------------
def q_umb_point(f0, n_buoy: float, epsilon: float = EPSILON_ENTRAIN, c_q: float = C_Q_POINT):
    """Umbrella volume flux from the buoyancy flux, axisymmetric source.

    Q_umb = c_q * (eps^2 F0^3 / N^5)^(1/4)   [m^3 s^-1]
    """
    f0 = np.asarray(f0, dtype=float)
    return c_q * (epsilon**2 * f0**3 / n_buoy**5) ** 0.25


def f0_point(q_umb, n_buoy: float, epsilon: float = EPSILON_ENTRAIN, c_f: float = C_F_POINT):
    """Buoyancy flux from the umbrella volume flux, axisymmetric source.

    F0 = c_f * (N^5 Q^4 / eps^2)^(1/3)   [m^4 s^-3]

    Exact inverse of :func:`q_umb_point` when c_f = c_q**(-4/3).
    """
    q_umb = np.asarray(q_umb, dtype=float)
    return c_f * (n_buoy**5 * q_umb**4 / epsilon**2) ** (1.0 / 3.0)


# --------------------------------------------------------------------------
# Planar (line / fissure) source closure
# --------------------------------------------------------------------------
def f0_planar(
    q_umb,
    n_buoy: float,
    fissure_length: float,
    epsilon: float = EPSILON_ENTRAIN,
    c_f: float = C_F_PLANAR,
):
    """Buoyancy flux from the umbrella volume flux, planar source of length l.

    F0 = c_f * (N^3 Q^3 / (eps l))^(1/2)   [m^4 s^-3]
    """
    q_umb = np.asarray(q_umb, dtype=float)
    return c_f * (n_buoy**3 * q_umb**3 / (epsilon * fissure_length)) ** 0.5


def q_umb_planar(
    f0,
    n_buoy: float,
    fissure_length: float,
    epsilon: float = EPSILON_ENTRAIN,
    c_f: float = C_F_PLANAR,
):
    """Exact inverse of :func:`f0_planar`."""
    f0 = np.asarray(f0, dtype=float)
    return ((f0 / c_f) ** 2 * epsilon * fissure_length / n_buoy**3) ** (1.0 / 3.0)


def transition_length(
    q_umb, n_buoy: float, epsilon: float = EPSILON_ENTRAIN, c: float = C_TRANSITION
):
    """Point/planar transition length l* = c (eps Q / N)^(1/3)   [m].

    A source fissure shorter than l* behaves as a point source; longer, as a
    line source.
    """
    q_umb = np.asarray(q_umb, dtype=float)
    return c * (epsilon * q_umb / n_buoy) ** (1.0 / 3.0)


def gaussian_length_scale(q_umb, w_s):
    """L = sqrt(Q_umb / w_s)   [m] -- the deposit length scale of the steady kernel."""
    return np.sqrt(np.asarray(q_umb, dtype=float) / np.asarray(w_s, dtype=float))


# --------------------------------------------------------------------------
# Direct MTT integration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class StemSolution:
    """Result of integrating the MTT plume equations."""

    z: np.ndarray
    q: np.ndarray
    m: np.ndarray
    f: np.ndarray
    z_neutral: float
    z_max: float
    q_neutral: float
    q_max: float

    @property
    def c_q_neutral(self) -> float:
        """Empirical coefficient c_q such that Q(z_n) = c_q (eps^2 F0^3/N^5)^(1/4)."""
        return self._coefficient(self.q_neutral)

    @property
    def c_q_max(self) -> float:
        """Empirical coefficient c_q such that Q(z_max) = c_q (eps^2 F0^3/N^5)^(1/4)."""
        return self._coefficient(self.q_max)

    def _coefficient(self, q_value: float) -> float:
        return q_value / self._q_scale

    _q_scale: float = 1.0


def solve_stem_mtt(
    f0: float,
    n_buoy: float,
    epsilon: float = EPSILON_ENTRAIN,
    z0_frac: float = 1e-6,
    rtol: float = 1e-10,
    atol: float = 1e-14,
    n_out: int = 2001,
) -> StemSolution:
    """Integrate the MTT point-source plume equations in a linear stratification.

    The ideal point source Q = M = 0 is singular, so integration starts from the
    unstratified pure-plume similarity solution at a small height
    ``z0 = z0_frac * l_rise``, where ``l_rise = (F0 / (eps^2 N^3))^(1/4)`` is the
    rise-height scale.  The result is independent of ``z0_frac`` to the quoted
    tolerance because the similarity solution is exact as z -> 0.

    Pure-plume similarity solution in a uniform ambient (top-hat, entrainment
    coefficient eps):

        Q(z) = (6 eps / 5) (9 eps / 10)^(1/3) pi^(2/3) F0^(1/3) z^(5/3)
        M(z) = (9 eps / 10)^(2/3) pi^(1/3) F0^(2/3) z^(4/3)
        F(z) = F0

    Returns
    -------
    StemSolution
        Profiles and the derived neutral level, top of rise and fluxes.
    """
    if f0 <= 0 or n_buoy <= 0 or epsilon <= 0:
        raise ValueError("f0, n_buoy and epsilon must all be positive")

    l_rise = (f0 / (epsilon**2 * n_buoy**3)) ** 0.25
    z0 = z0_frac * l_rise

    a = (9.0 * epsilon / 10.0) ** (1.0 / 3.0)
    q0 = (6.0 * epsilon / 5.0) * a * np.pi ** (2.0 / 3.0) * f0 ** (1.0 / 3.0) * z0 ** (5.0 / 3.0)
    m0 = a**2 * np.pi ** (1.0 / 3.0) * f0 ** (2.0 / 3.0) * z0 ** (4.0 / 3.0)

    # Integrate in (Q, P, F) with P = M^2.  The MTT momentum equation
    # dM/dz = F Q / M has a square-root singularity at the top of rise, but
    # dP/dz = 2 F Q is perfectly smooth there, so this change of variable
    # removes the stiffness and lets the top-of-rise event be detected cleanly.
    def rhs(_z, y):
        q, p_sq, f = y
        p_sq = max(p_sq, 0.0)
        return [
            2.0 * epsilon * np.sqrt(np.pi) * p_sq**0.25,
            2.0 * f * q,
            -(n_buoy**2) * q,
        ]

    def hit_neutral(_z, y):
        return y[2]

    def hit_top(_z, y):
        return y[1]

    hit_top.terminal = True
    hit_top.direction = -1
    hit_neutral.direction = -1

    z_span = (z0, 20.0 * l_rise)
    z_eval = np.linspace(z0, 20.0 * l_rise, n_out)
    sol = solve_ivp(
        rhs,
        z_span,
        [q0, m0**2, f0],
        method="DOP853",
        rtol=rtol,
        atol=atol,
        events=(hit_neutral, hit_top),
        t_eval=z_eval,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"MTT integration failed: {sol.message}")
    if sol.t_events[0].size == 0 or sol.t_events[1].size == 0:
        raise RuntimeError("MTT integration did not reach the neutral level / top of rise")

    z_neutral = float(sol.t_events[0][0])
    q_neutral = float(sol.sol(z_neutral)[0])
    z_max = float(sol.t_events[1][0])
    q_max = float(sol.sol(z_max)[0])

    keep = sol.t <= z_max
    q_scale = (epsilon**2 * f0**3 / n_buoy**5) ** 0.25

    return StemSolution(
        z=sol.t[keep],
        q=sol.y[0][keep],
        m=np.sqrt(np.maximum(sol.y[1][keep], 0.0)),
        f=sol.y[2][keep],
        z_neutral=z_neutral,
        z_max=z_max,
        q_neutral=q_neutral,
        q_max=q_max,
        _q_scale=q_scale,
    )


@dataclass(frozen=True)
class PlanarStemSolution:
    """Result of integrating the MTT line-plume equations (per unit length)."""

    z: np.ndarray
    q: np.ndarray  # volume flux per unit length, m^2 s^-1
    m: np.ndarray  # momentum flux per unit length, m^3 s^-2
    f: np.ndarray  # buoyancy flux per unit length, m^3 s^-3
    z_neutral: float
    z_max: float
    q_neutral: float
    q_max: float
    _q_scale: float = 1.0

    @property
    def c_q_neutral(self) -> float:
        """c such that q(z_n) = c (eps f0^2 / N^3)^(1/3)."""
        return self.q_neutral / self._q_scale

    @property
    def c_q_max(self) -> float:
        """c such that q(z_max) = c (eps f0^2 / N^3)^(1/3)."""
        return self.q_max / self._q_scale


def solve_stem_mtt_planar(
    f0_per_length: float,
    n_buoy: float,
    epsilon: float = EPSILON_ENTRAIN,
    z0_frac: float = 1e-6,
    rtol: float = 1e-10,
    atol: float = 1e-14,
    n_out: int = 2001,
) -> PlanarStemSolution:
    """Integrate the MTT line-plume equations in a linear stratification.

    Per unit fissure length, with q = 2 b w, m = 2 b w^2, f = 2 b w g',

        dq/dz = 2 eps m / q,    dm/dz = f q / m,    df/dz = -N^2 q.

    Integration starts from the exact pure-line-plume similarity solution
    w0 = (f0 / 2 eps)^(1/3), b = eps z, which makes the point-source singularity
    at z = 0 removable.

    Note on the similarity variables.  The VOLUME-FLUX coefficient is universal:
    q(z_max) = 2.11231 (eps f0^2 / N^3)^(1/3) for every f0, N and eps.  The RISE
    HEIGHT is not, in that grouping: it obeys

        z_max = 1.6609899 (f0 / (eps N^3))^(1/3),

    which carries an explicit eps^(-1/3).  Quoting z_max / (f0/N^3)^(1/3) as a
    constant -- 3.5785 -- is only correct at eps = 0.1, and it is 4.509 at
    eps = 0.05 and 3.126 at eps = 0.15.  Use :func:`planar_rise_height`.
    """
    if f0_per_length <= 0 or n_buoy <= 0 or epsilon <= 0:
        raise ValueError("f0_per_length, n_buoy and epsilon must all be positive")

    l_rise = (f0_per_length / n_buoy**3) ** (1.0 / 3.0)
    z0 = z0_frac * l_rise
    w0 = (f0_per_length / (2.0 * epsilon)) ** (1.0 / 3.0)
    q0 = 2.0 * epsilon * z0 * w0
    m0 = 2.0 * epsilon * z0 * w0**2

    def rhs(_z, y):
        q, p_sq, f = y
        p_sq = max(p_sq, 0.0)
        q = max(q, 1e-300)
        return [2.0 * epsilon * np.sqrt(p_sq) / q, 2.0 * f * q, -(n_buoy**2) * q]

    def hit_neutral(_z, y):
        return y[2]

    def hit_top(_z, y):
        return y[1]

    hit_top.terminal = True
    hit_top.direction = -1
    hit_neutral.direction = -1

    z_eval = np.linspace(z0, 20.0 * l_rise, n_out)
    sol = solve_ivp(
        rhs,
        (z0, 20.0 * l_rise),
        [q0, m0**2, f0_per_length],
        method="DOP853",
        rtol=rtol,
        atol=atol,
        events=(hit_neutral, hit_top),
        t_eval=z_eval,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"planar MTT integration failed: {sol.message}")
    if sol.t_events[0].size == 0 or sol.t_events[1].size == 0:
        raise RuntimeError("planar MTT integration did not reach neutral level / top of rise")

    z_neutral = float(sol.t_events[0][0])
    z_max = float(sol.t_events[1][0])
    keep = sol.t <= z_max
    q_scale = (epsilon * f0_per_length**2 / n_buoy**3) ** (1.0 / 3.0)

    return PlanarStemSolution(
        z=sol.t[keep],
        q=sol.y[0][keep],
        m=np.sqrt(np.maximum(sol.y[1][keep], 0.0)),
        f=sol.y[2][keep],
        z_neutral=z_neutral,
        z_max=z_max,
        q_neutral=float(sol.sol(z_neutral)[0]),
        q_max=float(sol.sol(z_max)[0]),
        _q_scale=q_scale,
    )


def point_rise_height(f0, n_buoy: float, epsilon: float = EPSILON_ENTRAIN, level: str = "max"):
    """Rise height of an axisymmetric plume above the source, m.

    z = C (F0 / (eps^2 N^3))^(1/4) with C = 1.36612 at the top of rise and
    1.03776 at the neutral level.
    """
    c = {"max": C_ZMAX_POINT, "neutral": C_ZNEUTRAL_POINT}[level]
    f0 = np.asarray(f0, dtype=float)
    return c * (f0 / (epsilon**2 * n_buoy**3)) ** 0.25


def planar_rise_height(
    f0_total, n_buoy: float, fissure_length: float, epsilon: float = EPSILON_ENTRAIN
):
    """Top-of-rise height of a line plume above the source, m.

    z_max = C_ZMAX_PLANAR * (f0_total / (l eps N^3))^(1/3).  Note the explicit
    dependence on the entrainment coefficient.
    """
    f0_total = np.asarray(f0_total, dtype=float)
    return C_ZMAX_PLANAR * (f0_total / (fissure_length * epsilon * n_buoy**3)) ** (1.0 / 3.0)


def fissure_length_for_rise(
    z_target: float, f0_total: float, n_buoy: float, epsilon: float = EPSILON_ENTRAIN
) -> float:
    """Fissure length l for which a line plume of total buoyancy flux f0_total
    rises exactly z_target, m.  Inverse of :func:`planar_rise_height`.
    """
    return float(C_ZMAX_PLANAR**3 * f0_total / (epsilon * n_buoy**3 * z_target**3))


def solve_stem_mtt_profile(
    f0: float,
    n2_of_z,
    epsilon: float = EPSILON_ENTRAIN,
    z_ceiling: float | None = None,
    n_scale: float | None = None,
    z0_frac: float = 1e-6,
    rtol: float = 1e-10,
    atol: float = 1e-14,
    n_out: int = 4001,
) -> StemSolution:
    """Integrate the MTT point-source plume through an ARBITRARY N^2(z) profile.

    Identical to :func:`solve_stem_mtt` except that ``dF/dz = -N^2(z) Q`` uses a
    depth-dependent stratification.  This matters whenever the plume rises far
    enough to leave the layer whose N was used to characterise it: a uniform-N
    estimate made from the deepest kilometre will badly over-estimate the rise
    height of a plume that would reach the thermocline.

    Parameters
    ----------
    f0 : float
        Source buoyancy flux, m^4 s^-3.
    n2_of_z : callable
        ``N^2`` in s^-2 as a function of height above the source, m.  It must be
        defined (and non-negative) for all heights the plume can reach.
    z_ceiling : float or None
        Height of the sea surface above the source, m.  If the plume is still
        rising at this height, integration stops there and ``z_max`` is set to
        it; check ``reached_ceiling``.
    n_scale : float or None
        Representative N used only to set the integration range and the reported
        similarity scale.  Defaults to ``sqrt(n2_of_z(0))``.
    """
    if f0 <= 0 or epsilon <= 0:
        raise ValueError("f0 and epsilon must be positive")
    n_ref = n_scale if n_scale is not None else float(np.sqrt(max(n2_of_z(0.0), 1e-30)))
    if n_ref <= 0:
        raise ValueError("could not establish a positive reference N")

    l_rise = (f0 / (epsilon**2 * n_ref**3)) ** 0.25
    z_top = z_ceiling if z_ceiling is not None else 20.0 * l_rise
    z0 = z0_frac * l_rise

    a = (9.0 * epsilon / 10.0) ** (1.0 / 3.0)
    q0 = (6.0 * epsilon / 5.0) * a * np.pi ** (2.0 / 3.0) * f0 ** (1.0 / 3.0) * z0 ** (5.0 / 3.0)
    m0 = a**2 * np.pi ** (1.0 / 3.0) * f0 ** (2.0 / 3.0) * z0 ** (4.0 / 3.0)

    def rhs(z, y):
        q, p_sq, f = y
        p_sq = max(p_sq, 0.0)
        return [
            2.0 * epsilon * np.sqrt(np.pi) * p_sq**0.25,
            2.0 * f * q,
            -max(float(n2_of_z(z)), 0.0) * q,
        ]

    def hit_neutral(_z, y):
        return y[2]

    def hit_top(_z, y):
        return y[1]

    hit_top.terminal = True
    hit_top.direction = -1
    hit_neutral.direction = -1

    z_eval = np.linspace(z0, z_top, n_out)
    sol = solve_ivp(
        rhs,
        (z0, z_top),
        [q0, m0**2, f0],
        method="DOP853",
        rtol=rtol,
        atol=atol,
        events=(hit_neutral, hit_top),
        t_eval=z_eval,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"MTT (profile) integration failed: {sol.message}")

    reached_ceiling = sol.t_events[1].size == 0
    z_max = z_top if reached_ceiling else float(sol.t_events[1][0])
    z_neutral = float(sol.t_events[0][0]) if sol.t_events[0].size else float("nan")
    q_max = float(sol.sol(z_max)[0])
    q_neutral = float(sol.sol(z_neutral)[0]) if np.isfinite(z_neutral) else float("nan")

    keep = sol.t <= z_max
    out = StemSolution(
        z=sol.t[keep],
        q=sol.y[0][keep],
        m=np.sqrt(np.maximum(sol.y[1][keep], 0.0)),
        f=sol.y[2][keep],
        z_neutral=z_neutral,
        z_max=z_max,
        q_neutral=q_neutral,
        q_max=q_max,
        _q_scale=(epsilon**2 * f0**3 / n_ref**5) ** 0.25,
    )
    object.__setattr__(out, "reached_ceiling", bool(reached_ceiling))
    return out
