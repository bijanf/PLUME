"""Settling velocity of basaltic tephra in cold seawater.

The inversion needs a settling law, not a single number: joint inversion of
several sieve fractions only makes sense if the w_s of each fraction follows
from one physical model with shared, explicit uncertainty.

The law is Ferguson & Church (2004), J. Sediment. Res. 74, 933-937, their
Eq. (4):

    w = R g D^2 / (C1 nu + (0.75 C2 R g D^3)^(1/2)),

with R = (rho_s - rho_w)/rho_w the submerged specific gravity, D the grain
diameter, nu the kinematic viscosity.  It interpolates smoothly between the
Stokes limit (w ~ D^2, set by C1) and the constant-drag limit (w ~ D^(1/2), set
by C2) with no regime switching, which matters because 63 um - 2 mm tephra
straddles both.

The coefficients
----------------
Ferguson & Church calibrated on quartz sediment: C1 = 18, C2 = 0.4 for smooth
spheres, and 24 / 1.2 for angular natural grains.  **Barreyre, Soule & Sohn
(2011), J. Volcanol. Geotherm. Res. 205, 84-93** re-fitted the same equation to
tank measurements of REAL DEEP-SEA VOLCANICLASTS from the Gakkel Ridge, by
non-linear maximum likelihood, and report coefficients per clast shape (their
Table 2).  Those are the right coefficients for this project and are in
``SHAPE_CLASSES``.

Why shape matters more than anything else here
----------------------------------------------
At the 250-500 um sieve midpoint, with rho_s = 2600 kg/m^3 and nu = 1.6e-6
m^2/s, the shape classes give

    blocky  3.01 cm/s      long  2.04 cm/s      sheet  1.38 cm/s

against the 3.0 +/- 1.0 cm/s that Pegler & Ferguson (2021) assumed.  The blocky
value reproduces their number almost exactly -- an independent confirmation from
measurements they did not use.  But NESCA's tephra is limu o Pele, bubble-wall
fragments, which are SHEET clasts: the workbook of core coordinates is literally
named "LimuSamples".  If the deposit is sheet-dominated then w is 0.46x the
assumed value, Q = w L^2 falls by the same factor and Phi ~ Q^(4/3) by 0.36x --
a factor 2.8 in the headline heat flux.  Shape is therefore a first-order
uncertainty that the benchmark did not propagate.

**And the deposit can settle it.**  Since L_i^2 = Q/w_i, the ratio of length
scales between two size fractions, L_i/L_j = sqrt(w_j/w_i), is independent of Q.
Across the four NESCA fractions that observable ranges from 3.63 for sheet
clasts to 5.31 for spheres -- a 46 % spread that a polydisperse inversion can
measure and a single-fraction inversion cannot.  Inferring the shape class from
the deposit, and propagating it into Q and Phi, is a contribution of this paper.

Caveats, recorded rather than hidden
------------------------------------
* Barreyre et al. do not state the clast density or, unambiguously, the
  viscosity behind their fit: the text gives nu = 1.83e-6 m^2/s "for seawater at
  20 degC" in one place and 9e-7 m^2/s for the tank in another, and neither is
  the standard 1.05e-6 at 20 degC.  Their coefficients were fitted jointly with
  whatever R and nu they used, so reproducing their published curves exactly is
  not possible from the paper alone.
* Their quoted anchors -- 12, 8 and 4 cm/s at D = 1 mm for blocky, long and
  sheet -- require rho_s of 3100-3700 kg/m^3 to reproduce, which is too dense
  for basaltic glass.  At a physical 2700 kg/m^3 this module gives 9.3, 6.7 and
  3.6 cm/s.  The anchors are quoted in the text as read off their Fig. 5, so the
  ~10-25 % shortfall is most likely figure-reading, but it is an open
  discrepancy and ``rho_s``, ``c1`` and ``c2`` are sampled, not fixed.
* ``nu`` matters at these grain sizes: warming from 2 degC to 20 degC raises w
  by 14-30 % depending on shape.  Use the deep-water value.
"""

from __future__ import annotations

import numpy as np

from .constants import NU_SEAWATER, RHO_TEPHRA, RHO_W, G

__all__ = [
    "settling_velocity",
    "sieve_midpoint",
    "phi_to_metres",
    "metres_to_phi",
    "shape_coefficients",
    "length_scale_ratio",
    "SHAPE_CLASSES",
    "VOLCANICLAST_SHAPES",
    "DEFAULT_SHAPE",
    "C1_STOKES",
    "C2_SPHERE",
]

#: (C1, C2) per clast shape.  The first two rows are Ferguson & Church (2004)
#: reference materials; the last three are Barreyre, Soule & Sohn (2011)
#: Table 2, fitted to deep-sea basaltic volcaniclasts from the Gakkel Ridge.
SHAPE_CLASSES = {
    "spheres": (18.0, 0.4),
    "quartz_sand": (24.0, 1.2),
    "blocky": (19.3, 2.0),
    "long": (30.7, 3.7),
    "sheet": (31.1, 14.8),
}

#: Shapes measured on real deep-sea volcaniclasts -- the admissible set for this
#: project.  ``spheres`` and ``quartz_sand`` are reference materials only.
VOLCANICLAST_SHAPES = ("blocky", "long", "sheet")

DEFAULT_SHAPE = "blocky"
"""Working default.

Chosen because it reproduces the benchmark's assumed 3 cm/s at 250-500 um, so
results are directly comparable to Pegler & Ferguson (2021) unless the shape is
deliberately varied.  It is NOT an assertion that NESCA's clasts are blocky --
the limu o Pele in these cores are sheet-like, and the shape is a sampled
parameter whose posterior the polydisperse inversion is meant to produce.
"""

C1_STOKES = 18.0
"""Stokes-limit constant for perfect spheres.  Retained for the limit test."""

C2_SPHERE = 0.4
"""Smooth-sphere drag constant.  A reference material, not a tephra value."""


def shape_coefficients(shape: str) -> tuple[float, float]:
    """(C1, C2) for a named clast shape.  See ``SHAPE_CLASSES``."""
    try:
        return SHAPE_CLASSES[shape]
    except KeyError:
        raise ValueError(
            f"unknown clast shape {shape!r}; choose from {sorted(SHAPE_CLASSES)}"
        ) from None


def settling_velocity(
    diameter,
    rho_s: float = RHO_TEPHRA,
    rho_w: float = RHO_W,
    nu: float = NU_SEAWATER,
    g: float = G,
    c1: float | None = None,
    c2: float | None = None,
    shape: str = DEFAULT_SHAPE,
):
    """Terminal settling velocity in still fluid, m s^-1.

    Parameters
    ----------
    diameter : float or array
        Grain diameter, m.
    rho_s, rho_w : float
        Grain and fluid density, kg m^-3.
    nu : float
        Kinematic viscosity of the fluid, m^2 s^-1.
    c1, c2 : float or None
        Ferguson & Church Eq. (4) constants.  If either is None it is taken from
        ``shape``.  Pass them explicitly only to explore off-grid values.
    shape : str
        Clast shape, a key of ``SHAPE_CLASSES``.  For deep-sea tephra use one of
        ``VOLCANICLAST_SHAPES``; these come from Barreyre et al. (2011) Table 2
        and are measurements on real volcaniclasts, not sediment analogues.
    """
    if c1 is None or c2 is None:
        default_c1, default_c2 = shape_coefficients(shape)
        c1 = default_c1 if c1 is None else c1
        c2 = default_c2 if c2 is None else c2
    d = np.asarray(diameter, dtype=float)
    if np.any(d <= 0):
        raise ValueError("diameter must be positive")
    r_sub = (rho_s - rho_w) / rho_w
    if r_sub <= 0:
        raise ValueError("grain density must exceed fluid density")
    return r_sub * g * d**2 / (c1 * nu + np.sqrt(0.75 * c2 * r_sub * g * d**3))


def phi_to_metres(phi):
    """Krumbein phi to diameter in metres:  d = 2^-phi mm."""
    return 1e-3 * 2.0 ** (-np.asarray(phi, dtype=float))


def metres_to_phi(d):
    """Diameter in metres to Krumbein phi."""
    return -np.log2(np.asarray(d, dtype=float) * 1e3)


def sieve_midpoint(d_lower: float, d_upper: float) -> float:
    """Geometric-mean diameter of a sieve fraction, m.

    The geometric mean is the right central value on the phi scale, on which
    sieve boundaries are defined.
    """
    if d_lower <= 0 or d_upper <= d_lower:
        raise ValueError("require 0 < d_lower < d_upper")
    return float(np.sqrt(d_lower * d_upper))


def length_scale_ratio(
    d_coarse: float, d_fine: float, shape: str = DEFAULT_SHAPE, **kwargs
) -> float:
    """L(fine)/L(coarse) = sqrt(w_coarse/w_fine), the shape-diagnostic observable.

    Because L_i^2 = Q/w_i, this ratio is independent of the umbrella flux, of the
    total erupted mass and of the eruption history.  It depends only on the
    settling law, so it is what a polydisperse deposit can use to identify the
    clast shape.  Across the four NESCA fractions it runs from 3.63 for sheet
    clasts to 5.31 for spheres.
    """
    w_c = float(settling_velocity(d_coarse, shape=shape, **kwargs))
    w_f = float(settling_velocity(d_fine, shape=shape, **kwargs))
    return float(np.sqrt(w_c / w_f))
