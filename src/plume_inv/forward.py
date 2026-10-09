"""Deposit prediction for all size classes at all core locations.

The forward map is

    (source history, size classes, environment)  ->  Omega_{ij}   [kg m^-2]

for core j and size class i.  Three effects sit between the axisymmetric kernel
of :mod:`plume_inv.kernel` and an observable core:

1. **Vent location.**  Radii are measured from the source, whose position is
   uncertain by ~800 m at NESCA and is therefore a sampled parameter.
2. **Fall-time drift.**  Particles leave the umbrella at height ``z_neutral``
   above the seafloor and fall for ``z_neutral / w_i``.  A depth-uniform ambient
   current U translates the whole deposit of class i by ``U z_neutral / w_i`` --
   *size-dependent*, because slower-settling classes drift further.  The
   relative offset between size classes is an observable that no single-class
   inversion can use.
3. **Umbrella-level crossflow** distorts the deposit into an anisotropic shape
   at first order in U tau_d / L.  It is not applied here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from .constants import EPSILON_ENTRAIN, LAMBDA_FRONT
from .kernel import (
    deposit_quasi_steady,
    deposit_unsteady,
    extend_history,
    settling_window,
)
from .umbrella import solve_umbrella

__all__ = [
    "SizeClass",
    "SourceHistory",
    "ForwardConfig",
    "predict_deposit",
    "constant_history",
    "waning_history",
    "two_pulse_history",
]


@dataclass(frozen=True)
class SizeClass:
    """One sieve fraction."""

    name: str
    w_s: float  # settling speed, m s^-1
    p: float  # mass fraction of the total particle release
    d_lower: float | None = None  # sieve lower bound, m
    d_upper: float | None = None  # sieve upper bound, m

    def __post_init__(self) -> None:
        if self.w_s <= 0:
            raise ValueError(f"{self.name}: settling speed must be positive")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"{self.name}: mass fraction must lie in [0, 1]")


@dataclass(frozen=True)
class SourceHistory:
    """Source history on a common time grid."""

    t: np.ndarray  # s, strictly increasing, t[0] >= 0
    q: np.ndarray  # umbrella volume flux, m^3 s^-1, > 0
    mdot: np.ndarray  # particle release rate, kg s^-1, >= 0

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=float)
        if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0):
            raise ValueError("t must be a strictly increasing 1-D grid")
        if np.shape(self.q) != t.shape or np.shape(self.mdot) != t.shape:
            raise ValueError("q and mdot must match the shape of t")
        if np.any(np.asarray(self.q) <= 0):
            raise ValueError("Q(t) must be strictly positive")
        if np.any(np.asarray(self.mdot) < 0):
            raise ValueError("Mdot_p(t) must be non-negative")

    @property
    def total_particle_mass(self) -> float:
        """int Mdot_p dt, kg."""
        return float(np.trapezoid(self.mdot, self.t))

    @property
    def umbrella_volume(self) -> float:
        """int Q dt, m^3 -- the total volume delivered to the umbrella."""
        return float(np.trapezoid(self.q, self.t))

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])


@dataclass(frozen=True)
class ForwardConfig:
    """Environment and model switches for the forward map."""

    n_buoy: float = 1.0e-3
    epsilon: float = EPSILON_ENTRAIN
    lam: float = LAMBDA_FRONT
    kernel: Literal["quasi_steady", "unsteady"] = "quasi_steady"
    vent_xy: tuple[float, float] = (0.0, 0.0)
    current_uv: tuple[float, float] = (0.0, 0.0)  # m s^-1, depth-uniform
    z_neutral: float = 0.0  # m above the seafloor
    n_quad_unsteady: int = 1024
    thickness_cap: float = np.inf  # m, optional cap on h(t)
    t_end: float | None = None  # s, end of the settling record
    settling_residual: float = 1e-5  # airborne fraction allowed
    n_tail: int = 2000  # grid points in the tail
    notes: str = field(default="")

    def drift(self, w_s: float) -> np.ndarray:
        """Fall-time drift vector U z_n / w_s for one size class, m."""
        return np.asarray(self.current_uv, dtype=float) * self.z_neutral / w_s


def predict_deposit(
    core_xy,
    classes: Sequence[SizeClass],
    source: SourceHistory,
    config: ForwardConfig | None = None,
) -> np.ndarray:
    """Predicted deposit mass per unit area.

    Parameters
    ----------
    core_xy : array, shape (n_cores, 2)
        Core positions in a local Cartesian frame, m.
    classes : sequence of SizeClass
    source : SourceHistory
    config : ForwardConfig

    Returns
    -------
    array, shape (n_cores, n_classes) -- kg m^-2.
    """
    cfg = config or ForwardConfig()
    xy = np.atleast_2d(np.asarray(core_xy, dtype=float))
    if xy.shape[1] != 2:
        raise ValueError("core_xy must have shape (n_cores, 2)")

    t = np.asarray(source.t, dtype=float)
    q = np.asarray(source.q, dtype=float)
    mdot = np.asarray(source.mdot, dtype=float)
    vent = np.asarray(cfg.vent_xy, dtype=float)

    umb = None
    if cfg.kernel == "unsteady":
        # Extend the record past the eruption so the particles still airborne
        # at shut-off are allowed to land (the intrusion keeps spreading).
        w_min = min(c.w_s for c in classes)
        t_end = (
            cfg.t_end
            if cfg.t_end is not None
            else settling_window(
                float(t[-1]),
                float(np.mean(q)),
                cfg.n_buoy,
                w_min,
                cfg.lam,
                residual=cfg.settling_residual,
            )
        )
        t, q, mdot = extend_history(t, q, mdot, t_end, n_extra=cfg.n_tail)
        umb = solve_umbrella(lambda tt: float(np.interp(tt, t, q)), t, cfg.n_buoy, cfg.lam)
        if np.isfinite(cfg.thickness_cap):
            umb = replace(umb, thickness=np.minimum(umb.thickness, cfg.thickness_cap))
    elif cfg.kernel != "quasi_steady":
        raise ValueError(f"unknown kernel {cfg.kernel!r}")

    out = np.zeros((xy.shape[0], len(classes)))
    for i, cls in enumerate(classes):
        centre = vent + cfg.drift(cls.w_s)
        r = np.linalg.norm(xy - centre[None, :], axis=1)
        if cfg.kernel == "quasi_steady":
            out[:, i] = deposit_quasi_steady(r, t, q, mdot, cls.w_s, cls.p)
        else:
            out[:, i] = deposit_unsteady(
                r, umb, t, q, mdot, cls.w_s, cls.p, n_quad=cfg.n_quad_unsteady
            )
    return out


# --------------------------------------------------------------------------
# Synthetic source histories (truth cases for the forward checks and synthetic tests)
# --------------------------------------------------------------------------
def constant_history(q_umb: float, mass: float, duration: float, n_t: int = 2001) -> SourceHistory:
    """Constant Q and constant Mdot_p over [0, duration]."""
    t = np.linspace(0.0, duration, n_t)
    return SourceHistory(
        t=t, q=np.full_like(t, float(q_umb)), mdot=np.full_like(t, mass / duration)
    )


def waning_history(
    q0: float,
    decay_time: float,
    mass: float,
    duration: float,
    m_exponent: float = 4.0 / 3.0,
    n_t: int = 2001,
) -> SourceHistory:
    """Exponentially waning Q with the source link Mdot_p proportional to Q^m."""
    t = np.linspace(0.0, duration, n_t)
    q = q0 * np.exp(-t / decay_time)
    shape = q**m_exponent
    mdot = shape * (mass / np.trapezoid(shape, t))
    return SourceHistory(t=t, q=q, mdot=mdot)


def two_pulse_history(
    q_lo: float,
    q_hi: float,
    t_on: float,
    t_off: float,
    mass: float,
    duration: float,
    m_exponent: float = 4.0 / 3.0,
    width: float = 0.02,
    n_t: int = 4001,
) -> SourceHistory:
    """A background flux with one smooth super-imposed pulse of amplitude q_hi.

    ``width`` is the pulse edge width as a fraction of ``duration``; the edges
    are smoothed with tanh so the history is differentiable, which keeps the
    quadrature second-order accurate.
    """
    t = np.linspace(0.0, duration, n_t)
    s = width * duration
    window = 0.5 * (np.tanh((t - t_on) / s) - np.tanh((t - t_off) / s))
    q = q_lo + (q_hi - q_lo) * window
    shape = q**m_exponent
    mdot = shape * (mass / np.trapezoid(shape, t))
    return SourceHistory(t=t, q=q, mdot=mdot)
