"""Error budget of the heat flux from the plume-stem closure and the setting.

Writes ``results/heat_budget.json``.

The heat flux is Phi = k c_f (N^5 <Q>^4 / eps^2)^(1/3) (point source, closure at
the top of rise, N the RMS over the deepest kilometre, seawater constants from
``constants.py``).  This script evaluates, for the blocky posterior median
<Q>, how Phi moves when each of those choices is changed:

* the closure evaluated at the neutral level instead of the top of rise;
* N averaged over thinner or thicker bottom layers;
* a point source integrated through the measured N(z) profile, with F0 found
  so that the plume delivers <Q> at its top of rise;
* a line source of several lengths, with the planar closure;
* the thermal expansion coefficient at the in-situ temperature, salinity and
  pressure of the deep layer (TEOS-10 on WOA23), in place of the adopted value;
* the lateral drift of each sieve fraction by an ambient current while it
  falls from the neutral level, per cm/s of current.

It also records where the 15 h reference duration comes from.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import gsw
import numpy as np
import xarray as xr
from scipy.optimize import brentq

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from plume_inv import constants as C  # noqa: E402
from plume_inv import settling, stem  # noqa: E402
from plume_inv.data import FRACTIONS  # noqa: E402

OUT = ROOT / "results" / "heat_budget.json"
LINE_LENGTHS_M = (2e3, 5e3, 10e3)


def main() -> int:
    strat = json.loads((ROOT / "results" / "stratification.json").read_text())
    vs = json.loads((ROOT / "results" / "vent_shape.json").read_text())
    q = float(vs["by_shape"]["blocky"]["q_mass_weighted_mean_m^3s^-1"]["median"])
    phi_post = float(vs["by_shape"]["blocky"]["phi_at_mass_weighted_mean_W"]["median"])
    n_ref = float(strat["N_primary"]["value_s^-1"])
    eps = C.EPSILON_ENTRAIN
    k = stem.heat_flux_constant()

    def phi_point_uniform(n, qq=q, c_f=C.C_F_POINT):
        return k * float(stem.f0_point(qq, n, eps, c_f))

    phi_ref = phi_point_uniform(n_ref)

    # closure at the neutral level: Q(z_n) = c_n (eps^2 F0^3/N^5)^(1/4)
    c_n = stem.solve_stem_mtt(600.0, n_ref, eps).c_q_neutral
    c_top = stem.solve_stem_mtt(600.0, n_ref, eps).c_q_max
    phi_neutral = phi_point_uniform(n_ref, c_f=c_n ** (-4.0 / 3.0))

    # N bands
    bands = {b: phi_point_uniform(v["N_s^-1"]) for b, v in strat["N_bands"].items()}

    # point source through the measured N(z)
    prof = strat["N_profile"]
    depth = np.asarray(prof["depth_m"], float)
    nn = np.asarray(prof["N_s^-1"], float)
    seafloor = float(strat["site"]["depth_nominal_m"])
    z_h = seafloor - depth
    ok = np.isfinite(nn) & (z_h >= 0)
    order = np.argsort(z_h[ok])
    zs, n2s = z_h[ok][order], np.clip(nn[ok][order], 0, None) ** 2

    def n2_of_z(z):
        return float(np.interp(z, zs, n2s, left=n2s[0], right=n2s[-1]))

    check = stem.solve_stem_mtt_profile(600.0, n2_of_z, eps, z_ceiling=seafloor - 5.0)
    z_check = strat["rise_height_consistency"]["point_source_variable_N"]["z_max_m"]

    def q_top(f0):
        return stem.solve_stem_mtt_profile(f0, n2_of_z, eps, z_ceiling=seafloor - 5.0).q_max

    f0_prof = brentq(lambda f: q_top(f) - q, 50.0, 1e6, xtol=1e-3)
    sol_prof = stem.solve_stem_mtt_profile(f0_prof, n2_of_z, eps, z_ceiling=seafloor - 5.0)
    phi_prof = k * f0_prof

    # line sources, uniform N (a line plume rises about a kilometre)
    lines = {}
    for ll in LINE_LENGTHS_M:
        f0 = float(stem.f0_planar(q, n_ref, ll, eps))
        lines[f"{ll / 1e3:g}km"] = {
            "phi_W": k * f0,
            "ratio_to_reference": k * f0 / phi_ref,
            "rise_height_m": float(stem.planar_rise_height(f0, n_ref, ll, eps)),
        }
    l_star = float(stem.transition_length(q, n_ref, eps))

    # in-situ thermal expansion over the deepest kilometre, WOA23 + TEOS-10
    lat0, lon0 = strat["site"]["lat"], strat["site"]["lon"]
    ds_t = xr.open_dataset(ROOT / strat["data_source"]["files"][0], decode_times=False)
    ds_s = xr.open_dataset(ROOT / strat["data_source"]["files"][1], decode_times=False)
    t = ds_t["t_an"].isel(time=0).sel(lat=lat0, lon=lon0, method="nearest")
    s = ds_s["s_an"].isel(time=0).sel(lat=lat0, lon=lon0, method="nearest")
    z = t["depth"].values
    band = (
        (z >= seafloor - 1000.0) & (z <= seafloor) & np.isfinite(t.values) & np.isfinite(s.values)
    )
    if band.sum() == 0:  # the nearest cell may be shallower; use its deepest valid levels
        valid = np.isfinite(t.values) & np.isfinite(s.values)
        band = valid & (z >= z[valid].max() - 1000.0)
    p = gsw.p_from_z(-z[band], lat0)
    sa = gsw.SA_from_SP(s.values[band], p, lon0, lat0)
    ct = gsw.CT_from_t(sa, t.values[band], p)
    alpha = gsw.alpha(sa, ct, p)
    alpha_mean = float(np.mean(alpha))
    phi_alpha = phi_ref * C.ALPHA_T / alpha_mean

    # drift by an ambient current while falling from the neutral level
    z_n = strat["rise_height_consistency"]["point_source_variable_N"]["z_neutral_m"]
    drift = {}
    for name, d_lo, d_hi, _m, _p in FRACTIONS:
        w = float(settling.settling_velocity(settling.sieve_midpoint(d_lo, d_hi)))
        drift[name] = {
            "w_s_m_s": w,
            "fall_time_h": z_n / w / 3600.0,
            "displacement_per_cm_s_m": 0.01 * z_n / w,
        }

    res = {
        "_description": (
            "Heat flux at the blocky posterior median <Q> under alternative "
            "closures and settings, the in-situ thermal expansion, and the "
            "drift of each fraction per cm/s of ambient current."
        ),
        "_generated": "2026-09-23",
        "_script": "experiments/heat_budget/run.py",
        "_git_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "q_mass_weighted_mean_m3s": q,
        "phi_reference_W": phi_ref,
        "phi_reference_matches_posterior_median_rel": abs(phi_ref / phi_post - 1.0),
        "closure_level": {
            "c_q_top_of_rise": c_top,
            "c_q_neutral_level": c_n,
            "phi_neutral_level_W": phi_neutral,
            "ratio_neutral_to_top": phi_neutral / phi_ref,
        },
        "n_bands": {
            b: {
                "N_s^-1": strat["N_bands"][b]["N_s^-1"],
                "phi_W": v,
                "ratio_to_reference": v / phi_ref,
            }
            for b, v in bands.items()
        },
        "point_source_measured_profile": {
            "reproduces_stratification_rise_m": [check.z_max, z_check],
            "f0_m4s3": f0_prof,
            "phi_W": phi_prof,
            "ratio_to_reference": phi_prof / phi_ref,
            "z_max_m": sol_prof.z_max,
            "z_neutral_m": sol_prof.z_neutral,
        },
        "line_source_uniform_N": lines,
        "transition_length_m": l_star,
        "thermal_expansion": {
            "adopted_K^-1": C.ALPHA_T,
            "in_situ_mean_K^-1": alpha_mean,
            "in_situ_range_K^-1": [float(alpha.min()), float(alpha.max())],
            "depths_m": [float(x) for x in z[band]],
            "phi_in_situ_W": phi_alpha,
            "ratio_to_reference": phi_alpha / phi_ref,
        },
        "drift_per_cm_s": {"z_neutral_m": z_n, "by_fraction": drift},
        "reference_duration": {
            "value_h": 15.0,
            "origin": (
                "midpoint of the 10-20 h duration range of the steady "
                "inversion (constants.BenchmarkPF2021.duration_range_h)"
            ),
            "range_h": list(C.BenchmarkPF2021().duration_range_h),
        },
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps({k_: v for k_, v in res.items() if not k_.startswith("_")}, indent=1)[:3500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
