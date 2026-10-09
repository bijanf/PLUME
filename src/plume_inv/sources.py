r"""Candidate heat sources for a megaplume, with physically bounded parameters.

Four mechanisms, each predicting a heat-flux history Phi(t) at the seafloor.
Phi(t) maps to the umbrella flux through the stem closure
(``stem.q_umb_point(Phi/k)``), hence to nu(Q), hence to the deposit -- so the
same likelihood serves all four and the nonparametric nu(Q) model.

Two corrections to simplified formulations
------------------------------------------
1. **Lava cooling is a convolution, not a product.**  A simplified form writes
   ``Phi_mag = A_lava(t) k dT / sqrt(pi kappa t)``.  The half-space solution
   measures ``t`` from the emplacement of each surface element, so multiplying
   it by a *growing* area with a single global clock treats every square metre
   as emplaced at t = 0.  The correct form convolves over emplacement age:

       Phi(t) = int_0^t Adot(s) * k dT / sqrt(pi kappa (t - s)) ds.

   For a flow emplaced instantaneously the two agree; for one emplaced over the
   eruption the simplified form over-estimates the early flux.
2. **The reservoir cannot both shrink and depressurise.**  A simplified form writes both
   ``dV_r/dt = -Q_h`` and ``dP/dt = -Q_h/(V_r beta_r)``.  The first is a
   reservoir that empties like a tank; the second is a fixed-volume compressible
   reservoir bleeding pressure.  Using both double-counts the depletion and puts
   a spurious finite-time singularity at ``V_r -> 0``.  Only the compressible
   form is implemented here, which is the one Wilcock (1997) uses.

Provenance
----------
The material properties below are standard basalt and seawater values.  Three of
the four mechanism papers (Palmer & Ernst 1998; Lowell & Germanovich 1995;
Cann & Strens 1989) could not be obtained, so the ranges are **not** taken from
the primary sources and are marked ``NEEDS_PRIMARY_SOURCE``.  They are wide
enough to be conservative, and the conclusions below are checked against the
whole range rather than a point value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "BasaltProperties",
    "BASALT",
    "NEEDS_PRIMARY_SOURCE",
    "phi_lava_cooling",
    "lava_heat_budget",
    "phi_dyke",
    "phi_volatile",
    "volatile_heat_budget",
    "phi_hydrothermal",
    "MECHANISMS",
]

NEEDS_PRIMARY_SOURCE = (
    "Standard basalt/seawater values; the primary sources (Palmer & Ernst 1998, "
    "Lowell & Germanovich 1995, Cann & Strens 1989) are not yet obtained, so "
    "these ranges are conservative placeholders, not quoted values."
)


@dataclass(frozen=True)
class BasaltProperties:
    """Thermal properties of basalt and of the seawater it heats."""

    k_therm: float = 2.0  # W m^-1 K^-1, range 1.5-2.5
    kappa: float = 7.0e-7  # m^2 s^-1, range 5e-7-1e-6
    rho_m: float = 2800.0  # kg m^-3, range 2700-2900
    c_m: float = 1100.0  # J kg^-1 K^-1, range 1000-1200
    latent: float = 4.0e5  # J kg^-1, latent heat of crystallisation
    delta_t: float = 1150.0  # K, magma ~1150 degC above 2 degC seawater
    rho_w: float = 1027.0
    c_w: float = 4200.0
    provenance: str = NEEDS_PRIMARY_SOURCE


BASALT = BasaltProperties()


# --------------------------------------------------------------------------
# 1. Conductive cooling of erupted lava
# --------------------------------------------------------------------------
def phi_lava_cooling(
    t, area_total: float, duration: float, props: BasaltProperties = BASALT, n_quad: int = 400
):
    """Heat flux from a lava flow emplaced at a constant areal rate, W.

    Convolution of the half-space conduction solution over emplacement age:

        Phi(t) = int_0^min(t,tau) (A/tau) k dT / sqrt(pi kappa (t - s)) ds
               = (A/tau) k dT * 2 sqrt(min(t,tau') ) / sqrt(pi kappa) ... (done
    numerically, because the integrable square-root singularity at s = t is
    handled by substituting u = sqrt(t - s)).
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if area_total <= 0 or duration <= 0:
        raise ValueError("area_total and duration must be positive")
    a_dot = area_total / duration
    pref = props.k_therm * props.delta_t / np.sqrt(np.pi * props.kappa)
    out = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti <= 0:
            continue
        s_hi = min(ti, duration)
        # u = sqrt(t - s): ds = -2u du, integrand k dT/(sqrt(pi kappa) u) -> 2 du
        u_lo, u_hi = np.sqrt(ti - s_hi), np.sqrt(ti)
        u = np.linspace(u_lo, u_hi, n_quad)
        out[i] = a_dot * pref * 2.0 * np.trapezoid(np.ones_like(u), u)
    return out


def lava_heat_budget(volume: float, props: BasaltProperties = BASALT) -> float:
    """Total heat available from cooling and crystallising a lava volume, J."""
    return float(volume * props.rho_m * (props.c_m * props.delta_t + props.latent))


# --------------------------------------------------------------------------
# 2. Conductive heating by a dyke
# --------------------------------------------------------------------------
def phi_dyke(t, length: float, height: float, props: BasaltProperties = BASALT):
    """Heat flux from both faces of a dyke into the country rock, W.

    Phi = 2 L H k dT / sqrt(pi kappa t): the half-space solution applied to two
    faces of area L x H, with t measured from intrusion.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.zeros_like(t)
    good = t > 0
    out[good] = (
        2.0
        * length
        * height
        * props.k_therm
        * props.delta_t
        / np.sqrt(np.pi * props.kappa * t[good])
    )
    return out


# --------------------------------------------------------------------------
# 3. Volatile exsolution
# --------------------------------------------------------------------------
def volatile_heat_budget(
    volume: float, co2_wt_frac: float, c_p_co2: float = 2250.0, props: BasaltProperties = BASALT
) -> float:
    """Total heat carried by exsolved CO2 cooling from magma to seawater, J."""
    mass_co2 = volume * props.rho_m * co2_wt_frac
    return float(mass_co2 * c_p_co2 * props.delta_t)


def phi_volatile(
    t,
    volume: float,
    co2_wt_frac: float,
    duration: float,
    c_p_co2: float = 2250.0,
    props: BasaltProperties = BASALT,
):
    """Heat flux from CO2 exsolution, released at a constant rate over ``duration``."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    total = volatile_heat_budget(volume, co2_wt_frac, c_p_co2, props)
    return np.where((t >= 0) & (t <= duration), total / duration, 0.0)


# --------------------------------------------------------------------------
# 4. Evacuation of a pre-existing hydrothermal reservoir
# --------------------------------------------------------------------------
def phi_hydrothermal(
    t,
    permeability: float,
    area: float,
    thickness: float,
    delta_p0: float,
    reservoir_temp: float,
    compressibility: float = 1e-9,
    viscosity: float = 1.5e-4,
    props: BasaltProperties = BASALT,
):
    """Heat flux from Darcy discharge of a compressible reservoir, W.

    Fixed-volume compressible reservoir (Wilcock 1997), NOT a reservoir that
    also empties as ``dV_r/dt = -Q_h``:

        Q_h(t) = (k A / (mu L)) dP(t),   d(dP)/dt = -Q_h / (V beta)

    which integrates to an exponential decay with time constant
    ``tau_h = V beta mu L / (k A)``, and

        Phi(t) = rho_w c_w Q_h(t) (T_r - T_w).

    Parameters
    ----------
    permeability : float
        m^2.  Crustal permeability spans many orders of magnitude; this is the
        parameter the evidence is most sensitive to.
    area, thickness : float
        Reservoir plan area (m^2) and thickness (m); volume is their product and
        ``thickness`` doubles as the Darcy path length.
    delta_p0 : float
        Initial overpressure, Pa.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    volume = area * thickness
    conductance = permeability * area / (viscosity * thickness)
    tau_h = volume * compressibility / conductance
    q_h = conductance * delta_p0 * np.exp(-t / tau_h)
    return (
        (props.rho_w * props.c_w * q_h * (reservoir_temp - 2.0)),
        float(tau_h),
        float(props.rho_w * props.c_w * conductance * delta_p0 * (reservoir_temp - 2.0) * tau_h),
    )


MECHANISMS = ("lava_cooling", "dyke_heating", "volatile_exsolution", "hydrothermal_evacuation")
