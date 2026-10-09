"""Physical constants and benchmark parameter values.

Every value carries a provenance string.  Values marked ``PF2021`` are taken
from Pegler & Ferguson (2021), Nat. Commun. 12, 2292,
doi:10.1038/s41467-021-22439-y, and were checked against the published text.

Nothing in this module may be changed without updating its provenance entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Universal
# --------------------------------------------------------------------------
G = 9.81
"""Gravitational acceleration, m s^-2."""

# --------------------------------------------------------------------------
# Deep NE Pacific seawater at the NESCA neutral level (~2.5-3.3 km depth)
# --------------------------------------------------------------------------
RHO_W = 1027.0
"""Seawater density, kg m^-3.  PF2021."""

C_P = 4200.0
"""Seawater specific heat capacity, J kg^-1 K^-1.  PF2021."""

ALPHA_T = 2.1e-4
"""Seawater thermal expansion coefficient, K^-1.  PF2021.

This is a deep-ocean value; at 2 degC, 35 g/kg and 300 bar TEOS-10 gives
alpha ~ 1.5e-4 K^-1 near the surface pressure and rises with pressure.  The
sensitivity of the heat-flux constant k to alpha is exactly inverse, so this
choice is carried through the error budget explicitly.
"""

NU_SEAWATER = 1.6e-6
"""Kinematic viscosity of seawater at ~2 degC, 35 g/kg, m^2 s^-1.

Used only by the settling law.  Provenance: standard seawater property tables;
re-derived with gsw in ``experiments/`` when the WOA profile is available.
"""

# --------------------------------------------------------------------------
# Plume / umbrella
# --------------------------------------------------------------------------
EPSILON_ENTRAIN = 0.1
"""Top-hat entrainment coefficient of the plume stem.  PF2021."""

N_BUOY_PF2021 = 1.0e-3
"""Ambient buoyancy frequency used by PF2021, s^-1 (from Argo profiles).

Superseded for our own inversions by ``results/stratification.json``.
"""

C_Q_POINT = 3.52
"""Coefficient in Q_umb = C_Q_POINT * (eps^2 F0^3 / N^5)^(1/4).  PF2021."""

C_F_POINT = 0.187
"""Coefficient in F0 = C_F_POINT * (N^5 Q^4 / eps^2)^(1/3).  PF2021.

The two published coefficients are rounded reports of one underlying constant:
C_Q_POINT**(-4/3) = 0.186757, which rounds to 0.187, and C_F_POINT**(-3/4) =
3.52107, which rounds to 3.52.  They are therefore inverse to each other only to
the 3 significant figures in which they are published -- round-tripping Q through
f0_point and q_umb_point changes it by 0.13 %.  Direct MTT integration
(``stem.solve_stem_mtt``) gives 3.52041 for the underlying constant, so both
published values are correct to their stated precision.
"""

C_F_PLANAR = 0.326
"""Coefficient in F0 = C_F_PLANAR * (N^3 Q^3 / (eps l))^(1/2).  PF2021.

Direct integration of the MTT line-plume equations gives the equivalent
volume-flux coefficient 2.11231 against the 2.11116 implied by 0.326, i.e.
agreement to 5.4e-4; see ``stem.solve_stem_mtt_planar``.
"""

C_TRANSITION = 3.0
"""Coefficient in the point/planar transition length l* = C * (eps Q / N)^(1/3).  PF2021."""

LAMBDA_FRONT = 0.2
"""Intrusion-front coefficient in dR/dt = LAMBDA_FRONT * N * h.

The constant-flux law R_f = (3 lambda N Q t^2 / 2pi)^(1/3) with lambda ~ 0.2 is
attributed to Costa, Folch & Macedonio (2013), GRL 40, 4823-4827.  That closed
form follows from this front condition together with volume conservation (see
:mod:`plume_inv.umbrella` and the paper, Methods).  The value is uncertain;
0.1-0.4 is propagated in sensitivity runs.
"""

# --------------------------------------------------------------------------
# Tephra
# --------------------------------------------------------------------------
RHO_TEPHRA = 2600.0
"""Bulk density of vesicular basaltic glass shards, kg m^-3.

Dense MORB glass is ~2900 kg m^-3; deep-sea limu/shard tephra is vesicular.
2600 is the mid-range working value; 2400-2900 is carried in sensitivity runs.
"""

W_S_PF2021 = 0.03
"""Settling speed of the 250-500 um fraction used by PF2021, m s^-1."""

W_S_PF2021_SIGMA = 0.01
"""1-sigma uncertainty on W_S_PF2021, m s^-1.  PF2021 quote 3 +/- 1 cm s^-1."""


MEGAPLUME_HEAT_J = (1.0e16, 1.0e17)
"""Heat content of megaplume-forming fluid, J (lower, upper).

Baker et al. (1989), J. Geophys. Res. 94, 9237-9250, doi:10.1029/JB094iB07p09237,
abstract: megaplume-forming fluids carry 10^16-10^17 J (see also
``results/heat_flux_context.json``).  Individual events reach 2.4e17 J
(Murton et al. 2006).  An earlier version used
(2e16, 2e17), which is Pegler & Ferguson's (2021) estimate for NESCA itself
(``BenchmarkPF2021.heat_range_J``), not a range of observed megaplumes.
"""


@dataclass(frozen=True)
class HeatFluxConstant:
    """k = rho c / (alpha g), converting buoyancy flux (m^4 s^-3) to heat flux (W)."""

    rho: float = RHO_W
    c_p: float = C_P
    alpha: float = ALPHA_T
    g: float = G

    @property
    def value(self) -> float:
        """k in W s^3 m^-4 (equivalently kg m^-2)."""
        return self.rho * self.c_p / (self.alpha * self.g)


K_HEAT = HeatFluxConstant().value
"""k = rho c / (alpha g) ~= 2.1e9 W s^3 m^-4.  PF2021."""


@dataclass(frozen=True)
class BenchmarkPF2021:
    """Headline numbers of Pegler & Ferguson (2021) for the NESCA deposit.

    Used only by the benchmark-reproduction test.  Uncertainties are the
    published 1-sigma values.
    """

    q_umb: float = 7.6e5
    q_umb_sigma: float = 3.6e5
    phi_heat: float = 1.5e12
    phi_heat_sigma: float = 0.9e12
    n_buoy: float = N_BUOY_PF2021
    w_s: float = W_S_PF2021
    epsilon: float = EPSILON_ENTRAIN
    vent_location_sigma: float = 800.0
    duration_range_h: tuple[float, float] = (10.0, 20.0)
    volume_range_km3: tuple[float, float] = (15.0, 80.0)
    heat_range_J: tuple[float, float] = (2e16, 20e16)
    lava_area_km2: float = 15.0
    lava_volume_total_m3: float = 4.5e7
    lava_volume_high_effusion_m3: float = 1.5e7
    provenance: str = field(
        default=(
            "Pegler & Ferguson (2021) Nat. Commun. 12, 2292, "
            "doi:10.1038/s41467-021-22439-y; values transcribed from the "
            "published text."
        )
    )


BENCHMARK = BenchmarkPF2021()
