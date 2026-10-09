"""Loaders for the NESCA push-core tephra data, with provenance checks.

Source: Clague, Paduan & Davis (2009), *J. Volcanol. Geotherm. Res.* **180**,
171-188, Appendix A supplementary workbooks, in ``data/raw/nesca_clague2009/``.
See that directory's README for the download URLs and checksums, and for the two
traps this module exists to handle.

The two traps
-------------
1. **Two core areas.** The footnote quotes one core diameter (6.9 cm), but the
   published ``g/m2`` column implies 6.90 cm for dives T887/T889/T890/T891 and
   7.08 cm for T888 -- a 5.3 % step that falls exactly where a spatial gradient
   would be looked for.  The area is therefore taken per dive, recovered from
   the data itself, and cross-checked against the two nominal diameters.
2. **Qualitative percentages.** ``%Glass`` is "visually estimated" and its
   vocabulary includes ``<1`` and ``<<1``, which the authors state they treated
   as 0.3 % and 0.1 %.  Reconstructing ``Total Glass (g)`` from the fractions
   under that convention reproduces the published total to better than 5 % for
   115 of 132 cores; the failures are concentrated in cores whose total is
   dominated by a trace percentage applied to a large sieved mass, which is
   exactly where a visual estimate is least reliable.  Every row therefore
   carries its reconstruction residual so the likelihood can down-weight them.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "FRACTIONS",
    "PCT_BELOW_ONE",
    "PCT_WELL_BELOW_ONE",
    "RAW_DIR",
    "NominalCoreArea",
    "load_nesca",
    "local_xy",
]

#: Default location of the Appendix A workbooks (fetched by ``scripts/fetch_data.py``).
RAW_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw" / "nesca_clague2009"

#: Sieve fractions as published: (name, lower bound m, upper bound m, mass col, pct col).
#: The top bin is open; ``d_upper`` is the modelling truncation, recorded here so
#: that every downstream number can be traced to the choice.
FRACTIONS = [
    ("63-125um", 63e-6, 125e-6, "63-125m", "Unnamed: 6"),
    ("125-250um", 125e-6, 250e-6, "125-250m", "Unnamed: 8"),
    ("250-500um", 250e-6, 500e-6, "250-500m", "Unnamed: 10"),
    (">500um", 500e-6, 1000e-6, ">500m", "Unnamed: 12"),
]

PCT_BELOW_ONE = 0.3
"""Value used for the literal '<1': the authors' stated convention."""

PCT_WELL_BELOW_ONE = 0.1
"""Value used for the literal '<<1': the authors' stated convention."""

EARTH_R = 6371000.0


@dataclass(frozen=True)
class NominalCoreArea:
    """The two push-corer geometries implied by the published ``g/m2``."""

    small_diameter_m: float = 0.069
    """The diameter the workbook's footnote quotes: 6.9 cm, area 3.739e-3 m^2."""

    large_diameter_m: float = 0.070826
    """Recovered from the g/m2 column for dive T888: area 3.939e-3 m^2.

    The footnote does not mention it.  6.9 cm would give 3.739e-3 m^2, so the
    T888 rows are 5.3 % apart from the rest; the implied diameter is 7.083 cm.
    """

    large_dives: tuple[str, ...] = ("T888",)
    tolerance: float = 1e-3
    """Relative tolerance when matching an implied area to a nominal one."""

    def area(self, diameter: float) -> float:
        return np.pi * (diameter / 2.0) ** 2


CORE_AREA = NominalCoreArea()


def _percent(value) -> float:
    """Map a published %Glass cell to a number, or NaN if it is not a datum."""
    s = str(value).strip()
    if s in ("", "nan", "None"):
        return np.nan
    if s == "lost":
        return np.nan  # sieve loss; a missing measurement, not a zero
    if s == "<<1":
        return PCT_WELL_BELOW_ONE
    if s == "<1":
        return PCT_BELOW_ONE
    try:
        return float(s)
    except ValueError:
        return np.nan


def _is_footnote(name: str) -> bool:
    return any(
        k in name
        for k in ("Indicates", "Core samples", "size fractions", "calculation", "except for")
    )


def local_xy(lat, lon, lat0: float, lon0: float):
    """Local tangent-plane coordinates in metres, east-north, about (lat0, lon0).

    A flat-Earth projection is accurate to much better than a metre over the
    ~15 km span of this survey, which is far below the vent-location uncertainty.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return x, y


def load_nesca(raw_dir: str | pathlib.Path | None = None) -> pd.DataFrame:
    """Load and join the NESCA push-core data.

    Returns one row per core with, for each sieve fraction, the sieved mass, the
    estimated glass percentage, the glass mass and the glass mass per unit area;
    plus coordinates, depth, dive, the per-dive core area, and data-quality
    columns.

    Raises
    ------
    FileNotFoundError, ValueError
        If the raw workbooks are missing or fail their provenance checks.  The
        checks are deliberately strict: this is the only path from the published
        data into the inversion.
    """
    raw = pathlib.Path(raw_dir) if raw_dir else RAW_DIR
    f_tephra, f_coords = raw / "mmc3.xls", raw / "mmc1.xls"
    for f in (f_tephra, f_coords):
        if not f.exists():
            raise FileNotFoundError(
                f"{f} is missing.  Run `python scripts/fetch_data.py` from the "
                f"repository root, or see {raw / 'README.rst'} for the download "
                "URLs and checksums."
            )

    t = pd.ExcelFile(f_tephra).parse("Sheet1")
    t = t[t["Sample"].notna()].copy()
    t = t[~t["Sample"].astype(str).map(_is_footnote)].reset_index(drop=True)
    t["Sample"] = t["Sample"].astype(str).str.strip()

    c = pd.ExcelFile(f_coords).parse("LimuSamples")
    c["Sample"] = c["Sample"].astype(str).str.strip()
    c = c[["Sample", "Latitude", "Longitude", "Depth", "Equipment", "Dive"]]

    df = t.merge(c, on="Sample", how="left", validate="one_to_one")
    missing = df["Latitude"].isna().sum()
    if missing:
        raise ValueError(f"{missing} cores have no coordinate row in mmc1.xls")

    df["dive"] = df["Sample"].str.split("-").str[0]
    df["published_total_glass_g"] = pd.to_numeric(df["Total Glass (g)"], errors="coerce")
    df["published_g_per_m2"] = pd.to_numeric(df["g/m2"], errors="coerce")

    # --- core area, recovered from the data and checked against the nominals ---
    implied = df["published_total_glass_g"] / df["published_g_per_m2"]
    df["core_area_m2"] = implied.round(6)
    nominal = [
        CORE_AREA.area(CORE_AREA.small_diameter_m),
        CORE_AREA.area(CORE_AREA.large_diameter_m),
    ]
    found = sorted(df["core_area_m2"].dropna().unique())
    if len(found) > len(nominal):
        raise ValueError(
            f"g/m2 implies {len(found)} distinct core areas ({found}); only "
            f"{len(nominal)} push-corer geometries are documented"
        )
    for a in found:
        if not any(abs(a - n) / n < CORE_AREA.tolerance for n in nominal):
            raise ValueError(
                f"core area {a} m^2 implied by g/m2 matches neither documented "
                f"push-corer geometry {nominal}"
            )
    # Which dives use the larger corer must match what the README records.
    big = set(df.loc[df["core_area_m2"] > np.mean(nominal), "dive"].dropna().unique())
    if big != set(CORE_AREA.large_dives):
        raise ValueError(
            f"the larger core area is used by dives {sorted(big)}, but "
            f"{sorted(CORE_AREA.large_dives)} was expected"
        )
    df["core_diameter_m"] = 2.0 * np.sqrt(df["core_area_m2"] / np.pi)

    # --- per-fraction masses ------------------------------------------------
    recon = np.zeros(len(df))
    for name, _d_lo, _d_hi, mcol, pcol in FRACTIONS:
        mass = pd.to_numeric(df[mcol], errors="coerce")
        pct = df[pcol].map(_percent)
        glass = mass * pct / 100.0
        df[f"sieved_g[{name}]"] = mass
        df[f"pct_glass[{name}]"] = pct
        df[f"pct_glass_qualitative[{name}]"] = df[pcol].astype(str).str.strip().isin(["<1", "<<1"])
        df[f"glass_g[{name}]"] = glass
        df[f"glass_g_per_m2[{name}]"] = glass / df["core_area_m2"]
        recon = recon + np.nan_to_num(glass.values)
    df["reconstructed_total_glass_g"] = recon

    with np.errstate(divide="ignore", invalid="ignore"):
        df["total_reconstruction_rel_error"] = (
            np.abs(df["reconstructed_total_glass_g"] - df["published_total_glass_g"])
            / df["published_total_glass_g"]
        )

    df["n_fractions_measured"] = sum(
        df[f"glass_g[{n}]"].notna().astype(int) for n, *_ in [(f[0],) for f in FRACTIONS]
    )
    df["complete_four_fractions"] = df["n_fractions_measured"] == len(FRACTIONS)
    df["any_qualitative_pct"] = np.logical_or.reduce(
        [df[f"pct_glass_qualitative[{n}]"].values for n, *_ in [(f[0],) for f in FRACTIONS]]
    )

    # --- geometry -----------------------------------------------------------
    lat0 = float(df["Latitude"].mean())
    lon0 = float(df["Longitude"].mean())
    x, y = local_xy(df["Latitude"], df["Longitude"], lat0, lon0)
    df["x_m"], df["y_m"] = x, y
    df.attrs["origin_lat"] = lat0
    df.attrs["origin_lon"] = lon0
    df.attrs["source"] = (
        "Clague, Paduan & Davis (2009) JVGR 180, 171-188, Appendix A "
        "(mmc3.xls Sheet1, mmc1.xls LimuSamples); "
        "see data/raw/nesca_clague2009/README.rst"
    )
    df.attrs["pct_convention"] = {"<1": PCT_BELOW_ONE, "<<1": PCT_WELL_BELOW_ONE}
    return df
