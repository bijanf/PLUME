"""The megaplume as an extreme heat flux, and the energy maps of two sources.

Helpers for placing an inferred heat flux on a ladder of literature values, for
converting a heat content into a mean power over an eruption duration, and for
mapping the heat that conductive sources deliver over two of their parameters.

The energy integrals follow ``experiments/source_discrimination/run.py``
exactly: a trapezoid rule over ``n`` equally spaced times from ``t0 = 1 s`` to
the eruption duration.  Both conductive sources are linear in their area
(lava: flow area; dyke: length times height), so a map is one call per
duration, scaled by area.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from scipy.optimize import brentq

from . import sources

__all__ = [
    "energy_delivered",
    "mean_power",
    "lava_energy_curve",
    "lava_energy_map",
    "dyke_energy_map",
    "crossing_duration",
    "verified_entries",
    "envelope_W",
    "envelope_J",
]

N_TIME = 20000
T0_S = 1.0


def energy_delivered(
    phi_of_t: Callable[[np.ndarray], np.ndarray],
    duration_s: float,
    n: int = N_TIME,
    t0: float = T0_S,
) -> float:
    """Heat delivered by a flux history from ``t0`` to ``duration_s``, J."""
    tt = np.linspace(t0, float(duration_s), n)
    return float(np.trapezoid(phi_of_t(tt), tt))


def mean_power(energy_j, duration_s):
    """Mean power of a heat content released over a duration, W."""
    return np.asarray(energy_j, dtype=float) / np.asarray(duration_s, dtype=float)


def lava_energy_curve(
    durations_s: Iterable[float], area_m2: float, emplacement_s: float, n: int = N_TIME
) -> np.ndarray:
    """Heat delivered by conductive lava cooling at each eruption duration, J."""
    return np.array(
        [
            energy_delivered(
                lambda tt: sources.phi_lava_cooling(tt, area_m2, emplacement_s), tau, n=n
            )
            for tau in durations_s
        ]
    )


def lava_energy_map(
    durations_s, areas_m2, emplacement_s: float, area_ref_m2: float, n: int = N_TIME
) -> np.ndarray:
    """Heat delivered by lava cooling, shape ``(len(areas), len(durations))``, J.

    Computed at ``area_ref_m2`` and scaled linearly to each area, since
    ``sources.phi_lava_cooling`` is proportional to the flow area.
    """
    curve = lava_energy_curve(durations_s, area_ref_m2, emplacement_s, n=n)
    return np.outer(np.asarray(areas_m2, dtype=float) / area_ref_m2, curve)


def dyke_energy_map(
    lengths_m,
    heights_m,
    duration_s: float,
    length_ref_m: float,
    height_ref_m: float,
    n: int = N_TIME,
) -> np.ndarray:
    """Heat delivered by a dyke over ``duration_s``, shape ``(len(heights), len(lengths))``, J.

    Computed at the reference dyke and scaled by the face area ``L H``, since
    ``sources.phi_dyke`` is proportional to it.
    """
    e_ref = energy_delivered(
        lambda tt: sources.phi_dyke(tt, length_ref_m, height_ref_m), duration_s, n=n
    )
    lh = np.outer(np.asarray(heights_m, dtype=float), np.asarray(lengths_m, dtype=float))
    return e_ref * lh / (length_ref_m * height_ref_m)


def crossing_duration(
    phi_of_t: Callable[[np.ndarray], np.ndarray],
    target_j: float,
    lo_s: float,
    hi_s: float,
    n: int = N_TIME,
    xtol: float = 1.0,
) -> float:
    """Eruption duration at which the delivered heat first equals ``target_j``, s."""

    def f(tau):
        return energy_delivered(phi_of_t, tau, n=n) - target_j

    return float(brentq(f, lo_s, hi_s, xtol=xtol))


def verified_entries(context: dict, ids: Iterable[str] | None = None) -> list[dict]:
    """Entries of ``results/heat_flux_context.json`` marked verified, optionally by id."""
    rows = [e for e in context["entries"] if e.get("verified") is True]
    if ids is None:
        return rows
    ids = list(ids)
    by_id = {e["id"]: e for e in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"not verified or absent: {missing}")
    return [by_id[i] for i in ids]


def _finite(values):
    return [float(v) for v in values if v is not None and np.isfinite(v)]


def envelope_W(entries: Iterable[dict]) -> tuple[float, float]:
    """Smallest and largest heat flux over entries (low_W, high_W, value_W), W."""
    lo, hi = [], []
    for e in entries:
        vals = _finite([e.get("low_W"), e.get("value_W"), e.get("high_W")])
        if vals:
            lo.append(min(vals))
            hi.append(max(vals))
    if not lo:
        raise ValueError("no heat-flux values in these entries")
    return min(lo), max(hi)


def envelope_J(entries: Iterable[dict]) -> tuple[float, float]:
    """Smallest and largest heat content over entries (low_J, energy_J, high_J), J."""
    lo, hi = [], []
    for e in entries:
        vals = _finite([e.get("low_J"), e.get("energy_J"), e.get("high_J")])
        if vals:
            lo.append(min(vals))
            hi.append(max(vals))
    if not lo:
        raise ValueError("no heat-content values in these entries")
    return min(lo), max(hi)
