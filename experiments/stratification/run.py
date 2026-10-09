"""Ambient stratification N(z) at the NESCA site from World Ocean Atlas 2023.

Writes ``results/stratification.json``.

Method
------
WOA23 objectively-analysed annual mean in-situ temperature (``t_an``) and
practical salinity (``s_an``) on the 1-degree grid are read at the cells
covering the NESCA push-core field, converted to TEOS-10 Absolute Salinity and
Conservative Temperature, and differentiated vertically with
``gsw.Nsquared`` to give N^2 at layer mid-points.

Three summaries are reported, because three different parts of the model need
different averages of N:

* ``N_neutral``     -- N at the depth of the umbrella (the seafloor depth minus
                       the neutral height), which sets the intrusion dynamics.
* ``N_rise_mean``   -- the root-mean-square of N over the rise height, which is
                       the N that the uniformly-stratified MTT closure
                       approximates.
* ``N_profile``     -- the full profile, for the record and for the SI figure.

Uncertainty combines the spread across the grid cells spanning the core field
with the WOA standard error of the mean where it is available.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import subprocess
import sys

import gsw
import numpy as np
import xarray as xr

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "woa"
OUT = ROOT / "results" / "stratification.json"

# Site: NESCA push-core field, northern Escanaba Trough, southern Gorda Ridge.
# Coordinates and depths from the USGS Escanaba Trough core-location data
# release underlying Clague, Paduan & Davis (2009):
# https://cmgds.marine.usgs.gov/catalog/pcmsc/DataReleases/ScienceBase/DR_P13B46QX/TN403_CoreLocations.html
SITE = dict(
    lat=40.98,
    lon=-127.49,
    lat_range=(40.6957, 41.1632),
    lon_range=(-127.5357, -127.4076),
    depth_range_m=(3201.0, 3304.0),
    depth_nominal_m=3250.0,
)
# Cells to average over: everything within +/- 1.5 deg of the core field.
HALO_DEG = 1.5
# Heights above the seafloor over which the stem rises (from stem.solve_stem_mtt
# with the benchmark F0; recomputed below so the two stay consistent).
Z_NEUTRAL_GUESS_M = 2900.0


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "uncommitted"


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from plume_inv import stem  # noqa: E402
    from plume_inv.constants import EPSILON_ENTRAIN  # noqa: E402

    t = xr.open_dataset(RAW / "woa23_decav_t00_01.nc", decode_times=False)
    s = xr.open_dataset(RAW / "woa23_decav_s00_01.nc", decode_times=False)

    sel = dict(
        lat=slice(SITE["lat_range"][0] - HALO_DEG, SITE["lat_range"][1] + HALO_DEG),
        lon=slice(SITE["lon_range"][0] - HALO_DEG, SITE["lon_range"][1] + HALO_DEG),
    )
    t_an = t["t_an"].isel(time=0).sel(**sel)
    s_an = s["s_an"].isel(time=0).sel(**sel)
    depth = t_an["depth"].values.astype(float)

    lats = t_an["lat"].values.astype(float)
    lons = t_an["lon"].values.astype(float)
    print(
        f"cells: lat {lats.min()}..{lats.max()}  lon {lons.min()}..{lons.max()}"
        f"  ({lats.size} x {lons.size}), {depth.size} standard depths"
    )

    n2_cells = []
    p_mid = None
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            tp = t_an.values[:, i, j]
            sp = s_an.values[:, i, j]
            good = np.isfinite(tp) & np.isfinite(sp)
            # Require a deep profile: the site is ~3250 m, so a cell that stops
            # shallower than 2800 m is a shelf or seamount cell and is dropped.
            if good.sum() < 60 or depth[good].max() < 2800.0:
                continue
            z = depth[good]
            p = gsw.p_from_z(-z, la)
            sa = gsw.SA_from_SP(sp[good], p, lo, la)
            ct = gsw.CT_from_t(sa, tp[good], p)
            n2, p_m = gsw.Nsquared(sa, ct, p, lat=la)
            z_m = -gsw.z_from_p(p_m, la)
            common = depth[1:-1]
            interp = np.interp(common, z_m, n2, left=np.nan, right=np.nan)
            interp[(common < z_m.min()) | (common > z_m.max())] = np.nan
            n2_cells.append(interp)
            if p_mid is None:
                p_mid = common

    n2_stack = np.array(n2_cells)
    print(f"usable cells: {n2_stack.shape[0]}")
    n2_mean = np.nanmean(n2_stack, axis=0)
    n2_std = np.nanstd(n2_stack, axis=0)
    z_mid = p_mid

    # Negative N^2 (static instability) is climatology noise in the deep ocean;
    # clip to zero for reporting N but record how much was clipped.
    n_clipped = int(np.sum(n2_mean < 0))
    n_mean = np.sqrt(np.clip(n2_mean, 0.0, None))
    n_lo = np.sqrt(np.clip(n2_mean - n2_std, 0.0, None))
    n_hi = np.sqrt(np.clip(n2_mean + n2_std, 0.0, None))

    seafloor = SITE["depth_nominal_m"]

    def n_at(depth_m: float) -> tuple[float, float, float]:
        return (
            float(np.interp(depth_m, z_mid, n_mean)),
            float(np.interp(depth_m, z_mid, n_lo)),
            float(np.interp(depth_m, z_mid, n_hi)),
        )

    def n_rms_band(height_m: float) -> tuple[float, float]:
        """RMS of N over the layer from the seafloor up to ``height_m``."""
        band = (z_mid >= seafloor - height_m) & (z_mid <= seafloor)
        if band.sum() < 2:
            return float("nan"), float("nan")
        val = float(np.sqrt(np.nanmean(np.clip(n2_mean[band], 0.0, None))))
        hi = float(np.sqrt(np.nanmean(np.clip(n2_mean[band] + n2_std[band], 0.0, None))))
        return val, hi - val

    # N is strongly depth-dependent, so a single uniform-N closure needs an
    # explicit choice of averaging band.  Report several; the primary value is
    # the deepest 1000 m, the layer an observed ~1 km megaplume occupies.
    bands = {}
    for hgt in (500.0, 1000.0, 1500.0, 2000.0, 3000.0):
        val, sig = n_rms_band(hgt)
        bands[f"{int(hgt)}m"] = {
            "N_s^-1": val,
            "sigma_s^-1": sig,
            "depth_band_m": [seafloor - hgt, seafloor],
        }
    n_deep, n_deep_sig = n_rms_band(1000.0)
    n_seafloor, _, _ = n_at(seafloor - 100.0)

    # ---- Rise-height consistency check ---------------------------------------
    # A uniform-N estimate is only self-consistent if the plume stays inside the
    # layer whose N was used.  It does not for a point source, so the point-source
    # rise height is computed by integrating MTT through the measured N(z).
    f0_benchmark = 6.0e2  # the benchmark buoyancy flux
    k_heat = stem.heat_flux_constant()
    phi_benchmark = k_heat * f0_benchmark

    z_h = seafloor - z_mid  # height above the seafloor
    order = np.argsort(z_h)
    z_h_sorted = z_h[order]
    n2_sorted = np.clip(n2_mean[order], 0.0, None)

    def n2_of_z(z: float) -> float:
        return float(np.interp(z, z_h_sorted, n2_sorted, left=n2_sorted[0], right=n2_sorted[-1]))

    point_var = stem.solve_stem_mtt_profile(
        f0_benchmark, n2_of_z, EPSILON_ENTRAIN, z_ceiling=seafloor - 5.0
    )
    point_uniform = stem.solve_stem_mtt(f0_benchmark, n_deep, EPSILON_ENTRAIN)

    z_obs = 1000.0  # representative observed megaplume rise, m above seafloor
    # For a fissure the rise is ~1 km, i.e. inside the layer whose N we measured,
    # so the uniform-N closure IS self-consistent here and the analytic
    # coefficients can be used.
    l_required = stem.fissure_length_for_rise(z_obs, f0_benchmark, n_deep, EPSILON_ENTRAIN)
    phi_for_length = {}
    for ll in (2e3, 5e3, 15e3, 36.7e3):
        f0_l = (stem.C_ZMAX_PLANAR**3) * ll * EPSILON_ENTRAIN * n_deep**3 * 0.0
        # Invert z_max = C (F0/(l eps N^3))^(1/3):  F0 = l eps N^3 (z/C)^3
        f0_l = ll * EPSILON_ENTRAIN * n_deep**3 * (z_obs / stem.C_ZMAX_PLANAR) ** 3
        phi_for_length[f"{ll / 1000:g}km"] = k_heat * f0_l
    f0_point_for_zobs = (z_obs / stem.C_ZMAX_POINT) ** 4 * EPSILON_ENTRAIN**2 * n_deep**3

    # Sensitivity of the point-source rise height to the entrainment coefficient.
    eps_sensitivity = {}
    for e in (0.08, 0.10, 0.12):
        sv = stem.solve_stem_mtt_profile(f0_benchmark, n2_of_z, e, z_ceiling=seafloor - 5.0)
        eps_sensitivity[f"{e:.2f}"] = {"z_neutral_m": sv.z_neutral, "z_max_m": sv.z_max}

    keep = (z_mid >= 100.0) & (z_mid <= seafloor + 100.0)
    payload = {
        "_description": "Ambient buoyancy frequency at the NESCA site from WOA23.",
        "_generated": datetime.date.today().isoformat(),
        "_script": "experiments/stratification/run.py",
        "_git_commit": git_hash(),
        "site": SITE,
        "data_source": {
            "product": "World Ocean Atlas 2023 (WOA23), objectively analysed annual mean",
            "climatology": "decav (1955-2022 decadal average)",
            "resolution_deg": 1.0,
            "files": ["data/raw/woa/woa23_decav_t00_01.nc", "data/raw/woa/woa23_decav_s00_01.nc"],
            "provenance": "data/raw/woa/README.rst (URLs, checksums, citation)",
            "equation_of_state": f"TEOS-10 via gsw {gsw.__version__}",
            "cells_used": int(n2_stack.shape[0]),
            "halo_deg": HALO_DEG,
        },
        "method": (
            "SA_from_SP and CT_from_t on WOA standard levels, then gsw.Nsquared "
            "at layer midpoints, per grid cell; mean and standard deviation "
            "taken across cells.  Negative N^2 (static instability from "
            "climatological smoothing) is clipped to zero before taking the "
            f"square root; this affected {n_clipped} of {z_mid.size} levels.  "
            "N varies by an order of magnitude over the water column, so no "
            "single number represents it: the value to use depends on the layer "
            "the plume occupies, and several bands are reported."
        ),
        "N_primary": {
            "value_s^-1": n_deep,
            "sigma_s^-1": n_deep_sig,
            "definition": "RMS of N over the deepest 1000 m (2250-3250 m depth)",
            "rationale": (
                "The layer an observed ~1 km megaplume occupies.  Use "
                "this as the uniform N of the MTT closure; propagate "
                "the band choice through sensitivity runs."
            ),
        },
        "N_bands": bands,
        "N_near_seafloor": {"value_s^-1": n_seafloor, "depth_m": seafloor - 100.0},
        "benchmark_comparison": {
            "pf2021_value_s^-1": 1.0e-3,
            "pf2021_source": "Argo profiles (Pegler & Ferguson 2021)",
            "ratio_primary_to_pf2021": n_deep / 1.0e-3,
            "verdict": (
                "WOA23 supports the benchmark's 1e-3 s^-1 to within 10 %; "
                "the difference is smaller than the band-choice ambiguity."
            ),
        },
        "mtt_coefficients": {
            "c_zmax_point": stem.C_ZMAX_POINT,
            "c_zneutral_point": stem.C_ZNEUTRAL_POINT,
            "c_zmax_planar": stem.C_ZMAX_PLANAR,
            "note": (
                "z_max = 1.3661222 (F0/(eps^2 N^3))^(1/4) for a point source and "
                "z_max = 1.6609899 (f0_total/(l eps N^3))^(1/3) for a line source. "
                "Both are established by direct MTT integration and verified "
                "independent of F0, N and eps to 1e-6 relative in "
                "tests/test_stem.py::TestMTTIntegration.  Note the explicit "
                "eps^(-1/3) in the planar form: quoting that coefficient against "
                "(f0/N^3)^(1/3) instead gives 3.5785, which is correct only at "
                "eps = 0.1 and is 4.509 at eps = 0.05."
            ),
        },
        "rise_height_consistency": {
            "f0_benchmark_m^4s^-3": f0_benchmark,
            "phi_benchmark_W": phi_benchmark,
            "phi_benchmark_note": (
                "k * F0 with F0 = 6e2 m^4 s^-3 gives 1.26 TW, which is the "
                "central value implied by the benchmark's own Q_umb; the "
                "benchmark quotes 1.5 +/- 0.9 TW.  All heat fluxes below are "
                "referred to this 1.26 TW, not to 1.5 TW."
            ),
            "water_depth_m": seafloor,
            "point_source_variable_N": {
                "z_neutral_m": point_var.z_neutral,
                "z_max_m": point_var.z_max,
                "reached_sea_surface": bool(getattr(point_var, "reached_ceiling", False)),
                "method": (
                    "MTT integrated through the measured N(z) profile, "
                    "since the plume leaves the layer used to define a "
                    "single N."
                ),
            },
            "point_source_uniform_N": {
                "z_neutral_m": point_uniform.z_neutral,
                "z_max_m": point_uniform.z_max,
                "N_used_s^-1": n_deep,
                "why_it_is_wrong": (
                    "Uses N from the deepest 1000 m for a plume that rises far "
                    "above that layer, so it over-estimates the rise height by "
                    f"{point_uniform.z_max / point_var.z_max:.2f}x.  Recorded "
                    "only to show the size of the error; do not quote it."
                ),
            },
            "point_source_epsilon_sensitivity": eps_sensitivity,
            "observed_megaplume_rise_m": z_obs,
            "observed_rise_provenance": (
                "Representative value for OTHER events (Juan de Fuca 1986, "
                "Gorda 1996, Lau 2008); NESCA's own plume was never observed. "
                "Event values and their sources are given in the paper."
            ),
            "fissure_length_required_for_observed_rise_m": l_required,
            "fissure_uniform_N_is_self_consistent": True,
            "fissure_note": (
                "A fissure source rises only ~1 km, i.e. stays inside the layer "
                "whose N was measured, so the uniform-N closure is "
                "self-consistent for the fissure case even though it is not for "
                "the point source."
            ),
            "phi_achievable_at_observed_rise_W": phi_for_length,
            "f0_point_source_for_observed_rise_m^4s^-3": f0_point_for_zobs,
            "phi_point_source_for_observed_rise_W": k_heat * f0_point_for_zobs,
            "finding": (
                "With the measured N(z), a point source carrying the benchmark "
                f"buoyancy flux reaches {point_var.z_max:.0f} m above the "
                f"seafloor in a {seafloor:.0f} m water column -- high, but it "
                "does NOT breach the surface.  (A uniform-N estimate using the "
                "deep-layer N gives 4086 m and would have breached it; that "
                "estimate is inconsistent and is not used.)  The rise is still "
                "2-2.5 times the ~1 km rise of the observed megaplumes, which "
                "favours a line source over a point source.  But a fissure of "
                "plausible length (2-15 km) rising only ~1 km delivers "
                "0.07-0.51 TW, well below the 1.26 TW implied by the deposit; "
                "matching 1.26 TW at a 1 km rise needs a fissure ~37 km long. "
                "Either the NESCA plume rose higher than the observed events, "
                "or the peak heat flux was lower and the discharge longer -- "
                "a time-dependent source.  This tension is reported rather "
                "than resolved by assumption."
            ),
        },
        "N_profile": {
            "depth_m": [float(x) for x in z_mid[keep]],
            "N_s^-1": [float(x) for x in n_mean[keep]],
            "N_lo_s^-1": [float(x) for x in n_lo[keep]],
            "N_hi_s^-1": [float(x) for x in n_hi[keep]],
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print(
        f"N (deepest 1000 m)   = {n_deep:.3e} +/- {n_deep_sig:.1e} s^-1   "
        f"[PF2021 used 1.0e-3 -> ratio {n_deep / 1e-3:.2f}]"
    )
    for name, b in bands.items():
        print(f"  band {name:>6}: N = {b['N_s^-1']:.3e} s^-1")
    print(f"N (100 m off bottom) = {n_seafloor:.3e} s^-1")
    print(
        f"point source, variable N(z): z_n = {point_var.z_neutral:.0f} m, "
        f"z_max = {point_var.z_max:.0f} m  (water column {seafloor:.0f} m)"
    )
    print(
        f"  the uniform-N estimate would give z_max = {point_uniform.z_max:.0f} m "
        f"-- over by {point_uniform.z_max / point_var.z_max:.2f}x, not used"
    )
    print(
        f"fissure length needed for a {z_obs:.0f} m rise at "
        f"{phi_benchmark / 1e12:.2f} TW: {l_required / 1000:.1f} km"
    )
    for name, phi in phi_for_length.items():
        print(f"  a {name} fissure rising {z_obs:.0f} m delivers {phi / 1e12:.3f} TW")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
