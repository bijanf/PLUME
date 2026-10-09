"""The NESCA megaplume as an extreme heat flux, and the energy maps of two sources.

Writes ``results/heat_context.json``.

1. A ladder of heat fluxes, from a single black smoker to the heat loss of the
   whole Earth, built from the verified entries of
   ``results/heat_flux_context.json`` (``verified: true`` only), with the NESCA
   inference placed on it (``results/vent_shape.json``: blocky settling and
   shape-marginalised).  The megaplume heat range adopted by the source test
   (``results/model_comparison.json``) is converted to a mean power over
   eruption durations of 1 h to 100 h, the duration sweep of that test.
2. Ratios that follow from the ladder: the NESCA heat flux as a fraction of the
   global hydrothermal and ridge-axis fluxes, as a multiple of a large vent
   field and of an average black smoker, and the time a large vent field needs
   to release one megaplume's heat.
3. The heat delivered by conductive lava cooling over (eruption duration, flow
   area), at the top of the prior range exactly as in
   ``experiments/source_discrimination/run.py`` (emplacement over 1800 s), with
   the crossing of the megaplume floor at the mapped area and the area factor
   needed at the reference duration.
4. The heat delivered by a dyke over (length, height) at the reference
   duration, over the prior box of the source test.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from plume_inv import heat_context as HC  # noqa: E402
from plume_inv import sources  # noqa: E402

OUT = ROOT / "results" / "heat_context.json"

# As in experiments/source_discrimination/run.py: the lava flow at the top of
# its prior (mapped area, fastest emplacement) and the dyke at the top of its
# prior box.
AREA_MAPPED_M2 = 15e6
EMPLACEMENT_S = 1800.0
DYKE_L_MAX_M, DYKE_H_MAX_M = 3e4, 5e3
DYKE_L_MIN_M, DYKE_H_MIN_M = 1e3, 1e2
PLAUSIBLE_DYKE_M = (5e3, 1e3)  # length, height of a representative dyke
YEAR_S = 365.25 * 86400.0

DUR_H = np.geomspace(1.0, 100.0, 49)
AREA_KM2 = np.geomspace(1.0, 200.0, 61)
DYKE_L_KM = np.geomspace(DYKE_L_MIN_M / 1e3, DYKE_L_MAX_M / 1e3, 61)
DYKE_H_KM = np.geomspace(DYKE_H_MIN_M / 1e3, DYKE_H_MAX_M / 1e3, 61)

# Ladder rows: (key, label, group, verified entry ids).  Rows are listed in
# increasing heat flux.
LADDER = [
    ("single_black_smoker", "single black smoker", "steady", ["single_black_smoker"]),
    ("vent_field", "vent field", "steady", ["vent_field_TAG", "vent_field_lucky_strike"]),
    ("ridge_segment", "ridge segment", "steady", ["ridge_segment_southern_jdfr"]),
    ("megaplume_ep86", "1986 megaplume", "megaplume", ["megaplume_ep86_heat_flux"]),
    ("global_ridge_axis", "global ridge axis", "global", ["global_axial_hydrothermal"]),
    (
        "global_hydrothermal",
        "global hydrothermal",
        "global",
        ["global_hydrothermal_elderfield", "global_hydrothermal_stein"],
    ),
    ("ocean_floor", "ocean floor", "global", ["ocean_heat_loss_davies"]),
    ("whole_earth", "whole Earth", "global", ["earth_heat_loss_davies"]),
]


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


def ratio_block(phi: float, lo: float, hi: float) -> dict:
    """phi / reference for a reference known as a range [lo, hi]."""
    return {"low": float(phi / hi), "high": float(phi / lo)}


def main() -> int:
    ctx = json.loads((ROOT / "results" / "heat_flux_context.json").read_text())
    vs = json.loads((ROOT / "results" / "vent_shape.json").read_text())
    mc = json.loads((ROOT / "results" / "model_comparison.json").read_text())
    dt = mc["delivery_test"]
    floor_j, ceil_j = (float(x) for x in dt["_megaplume_heat_range_J"])
    tau_ref = float(dt["_duration_reference_s"])

    # ---- 1. NESCA and the ladder ------------------------------------------
    nesca = {}
    for key, src in (
        ("blocky", vs["by_shape"]["blocky"]["phi_at_mass_weighted_mean_W"]),
        ("long", vs["by_shape"]["long"]["phi_at_mass_weighted_mean_W"]),
        ("sheet", vs["by_shape"]["sheet"]["phi_at_mass_weighted_mean_W"]),
    ):
        nesca[key] = {"median_W": float(src["median"]), "ci95_W": [float(x) for x in src["ci95"]]}
    nesca["_source"] = "results/vent_shape.json#by_shape.<shape>.phi_at_mass_weighted_mean_W"

    ladder = []
    for key, label, group, ids in LADDER:
        rows = HC.verified_entries(ctx, ids)
        lo, hi = HC.envelope_W(rows)
        vals = [r["value_W"] for r in rows if r.get("value_W") is not None]
        ladder.append(
            {
                "key": key,
                "label": label,
                "group": group,
                "low_W": lo,
                "high_W": hi,
                "value_W": float(vals[0]) if vals else None,
                "entry_ids": ids,
                "bibkeys": sorted({r["bibkey"] for r in rows}),
            }
        )

    dur_lo_s, dur_hi_s = 3600.0 * DUR_H[0], 3600.0 * DUR_H[-1]
    megaplume_power = {
        "heat_J": [floor_j, ceil_j],
        "heat_source": "results/model_comparison.json#delivery_test._megaplume_heat_range_J",
        "duration_h": [float(DUR_H[0]), float(DUR_H[-1])],
        "mean_power_W": [
            float(HC.mean_power(floor_j, dur_hi_s)),
            float(HC.mean_power(ceil_j, dur_lo_s)),
        ],
        "reference_duration_h": tau_ref / 3600.0,
        "mean_power_at_reference_W": [
            float(HC.mean_power(floor_j, tau_ref)),
            float(HC.mean_power(ceil_j, tau_ref)),
        ],
        "_note": "Heat range of the source test, divided by the eruption duration. "
        "The lowest power is the floor released over 100 h; the highest is "
        "the ceiling released over 1 h.",
    }

    # verified heat contents of observed events
    mp_ids = ["megaplume_ep86_heat_content", "megaplume_ep87_heat_content", "event_plume_carlsberg"]
    ev_ids = mp_ids + ["event_plumes_coaxial_1993"]
    events = {
        "megaplumes_J": list(HC.envelope_J(HC.verified_entries(ctx, mp_ids))),
        "megaplume_ids": mp_ids,
        "all_event_plumes_J": list(HC.envelope_J(HC.verified_entries(ctx, ev_ids))),
        "all_event_plume_ids": ev_ids,
    }

    # ---- 2. ratios ----------------------------------------------------------
    by_key = {r["key"]: r for r in ladder}
    vf_hi = by_key["vent_field"]["high_W"]
    tag = HC.verified_entries(ctx, ["vent_field_TAG"])[0]
    smoker_mean = by_key["single_black_smoker"]["value_W"]
    gh = by_key["global_hydrothermal"]
    ax = by_key["global_ridge_axis"]
    seg = by_key["ridge_segment"]
    ep86 = by_key["megaplume_ep86"]
    earth = by_key["whole_earth"]
    ocean = by_key["ocean_floor"]
    ratios = {}
    for key in ("blocky", "long", "sheet"):
        phi = nesca[key]["median_W"]
        ratios[key] = {
            "over_global_hydrothermal_central": float(phi / gh["value_W"]),
            "over_global_hydrothermal": ratio_block(phi, gh["low_W"], gh["high_W"]),
            "over_global_ridge_axis": ratio_block(phi, ax["low_W"], ax["high_W"]),
            "over_ocean_floor": float(phi / ocean["value_W"]),
            "over_whole_earth": float(phi / earth["value_W"]),
            "over_large_vent_field": float(phi / vf_hi),
            "over_tag_field": ratio_block(phi, tag["low_W"], tag["high_W"]),
            "over_ridge_segment": ratio_block(phi, seg["low_W"], seg["high_W"]),
            "over_ep86_megaplume": ratio_block(phi, ep86["low_W"], ep86["high_W"]),
            "mean_black_smokers": float(phi / smoker_mean),
            "hours_to_release_megaplume_range": [
                float(floor_j / phi / 3600.0),
                float(ceil_j / phi / 3600.0),
            ],
            "heat_in_reference_duration_J": float(phi * tau_ref),
            "heat_in_reference_duration_over_floor": float(phi * tau_ref / floor_j),
        }
    ratios["_definitions"] = {
        "large_vent_field_W": vf_hi,
        "large_vent_field_basis": "upper end of the verified vent-field range "
        "(Lucky Strike, barreyre2012structure)",
        "mean_black_smoker_W": smoker_mean,
        "global_hydrothermal_central_W": gh["value_W"],
    }
    vent_field_time = {
        "vent_field_W": vf_hi,
        "floor_days": float(floor_j / vf_hi / 86400.0),
        "floor_years": float(floor_j / vf_hi / YEAR_S),
        "ceiling_years": float(ceil_j / vf_hi / YEAR_S),
        "tag_field_years": [
            float(floor_j / tag["high_W"] / YEAR_S),
            float(ceil_j / tag["low_W"] / YEAR_S),
        ],
        "_note": "Time a vent field discharging steadily at the given power needs to "
        "release the megaplume floor and ceiling of results/model_comparison.json.",
    }
    # Cross-check against Baker et al. (1987): EP86 equalled the annual output of
    # 200-2000 high-temperature chimneys.
    ep86_j = HC.verified_entries(ctx, ["megaplume_ep86_heat_content"])[0]
    chimney_years = [
        float(ep86_j["low_J"] / (smoker_mean * YEAR_S)),
        float(ep86_j["high_J"] / (smoker_mean * YEAR_S)),
    ]
    cross = {
        "ep86_chimney_years_at_mean_smoker": chimney_years,
        "baker1987_chimney_years": [200.0, 2000.0],
        "consistent": bool(200.0 <= chimney_years[0] and chimney_years[1] <= 2000.0),
    }

    # ---- 3. lava cooling map -------------------------------------------------
    dur_s = DUR_H * 3600.0
    e_lava_ref = HC.lava_energy_curve(dur_s, AREA_MAPPED_M2, EMPLACEMENT_S)
    e_lava = np.outer(AREA_KM2 * 1e6 / AREA_MAPPED_M2, e_lava_ref)
    # linearity in area, checked directly at two grid points
    lin = []
    for ia, it in ((0, 10), (len(AREA_KM2) - 1, len(DUR_H) - 5)):
        direct = HC.energy_delivered(
            lambda tt, a=AREA_KM2[ia] * 1e6: sources.phi_lava_cooling(tt, a, EMPLACEMENT_S),
            dur_s[it],
        )
        lin.append(abs(e_lava[ia, it] / direct - 1.0))
    lava_fn = lambda tt: sources.phi_lava_cooling(tt, AREA_MAPPED_M2, EMPLACEMENT_S)  # noqa: E731
    e_ref_15h = HC.energy_delivered(lava_fn, tau_ref)
    reproduce = abs(e_ref_15h / dt["lava_cooling"]["energy_delivered_J"] - 1.0)
    tau_cross = HC.crossing_duration(lava_fn, floor_j, 60.0, 100 * 3600.0)
    area_factor = floor_j / e_ref_15h
    # the duration at which the area factor needed would be seven
    tau_seven = HC.crossing_duration(lava_fn, floor_j / 7.0, 60.0, 100 * 3600.0)
    lava_map = {
        "duration_h": DUR_H.tolist(),
        "area_km2": AREA_KM2.tolist(),
        "energy_J": [[float(f"{v:.6e}") for v in row] for row in e_lava],
        "_axes": "energy_J[i_area][i_duration]",
        "emplacement_s": EMPLACEMENT_S,
        "mapped_area_km2": AREA_MAPPED_M2 / 1e6,
        "reference_duration_h": tau_ref / 3600.0,
        "floor_J": floor_j,
        "ceiling_J": ceil_j,
        "energy_at_mapped_area_reference_J": e_ref_15h,
        "reproduces_source_test_rel_err": float(reproduce),
        "linearity_in_area_max_rel_err": float(max(lin)),
        "crossing_duration_h_at_mapped_area": tau_cross / 3600.0,
        "crossing_duration_h_power_law_source_test": mc["duration_sensitivity"][
            "_lava_crossing_duration_h"
        ],
        "area_factor_needed_at_reference_duration": float(area_factor),
        "area_needed_at_reference_duration_km2": float(area_factor * AREA_MAPPED_M2 / 1e6),
        "duration_h_where_area_factor_is_seven": tau_seven / 3600.0,
        "fraction_of_map_below_floor": float(np.mean(e_lava < floor_j)),
    }

    # ---- 4. dyke map ------------------------------------------------------------
    e_dyke = HC.dyke_energy_map(
        DYKE_L_KM * 1e3, DYKE_H_KM * 1e3, tau_ref, DYKE_L_MAX_M, DYKE_H_MAX_M
    )
    e_ceiling = float(e_dyke[-1, -1])
    lp, hp = PLAUSIBLE_DYKE_M
    dyke_fn = lambda tt: sources.phi_dyke(tt, lp, hp)  # noqa: E731
    e_plaus = HC.energy_delivered(dyke_fn, tau_ref)
    lh_needed = floor_j / (e_ceiling / (DYKE_L_MAX_M * DYKE_H_MAX_M))
    tau_plaus = HC.crossing_duration(dyke_fn, floor_j, 60.0, 5000 * 3600.0)
    dyke_map = {
        "length_km": DYKE_L_KM.tolist(),
        "height_km": DYKE_H_KM.tolist(),
        "energy_J": [[float(f"{v:.6e}") for v in row] for row in e_dyke],
        "_axes": "energy_J[i_height][i_length]",
        "duration_h": tau_ref / 3600.0,
        "prior_ceiling_energy_J": e_ceiling,
        "reproduces_source_test_rel_err": float(
            abs(e_ceiling / dt["dyke_heating"]["energy_delivered_J"] - 1.0)
        ),
        "face_area_product_needed_m2": float(lh_needed),
        "plausible_dyke": {
            "length_m": lp,
            "height_m": hp,
            "energy_J": e_plaus,
            "energy_over_floor": float(e_plaus / floor_j),
            "phi_at_reference_duration_W": float(dyke_fn(np.array([tau_ref]))[0]),
            "mean_power_over_reference_W": float(e_plaus / tau_ref),
            "height_needed_at_this_length_m": float(lh_needed / lp),
            "duration_needed_for_floor_h": tau_plaus / 3600.0,
        },
        "fraction_of_prior_box_below_floor": float(np.mean(e_dyke < floor_j)),
    }

    res = {
        "_description": (
            "The NESCA megaplume on a ladder of verified heat fluxes, the "
            "ratios that follow, and the heat that lava cooling and dyke "
            "heating deliver over two of their parameters."
        ),
        "_generated": datetime.date.today().isoformat(),
        "_script": "experiments/heat_context/run.py",
        "_git_commit": git_hash(),
        "_inputs": [
            "results/heat_flux_context.json",
            "results/vent_shape.json",
            "results/model_comparison.json",
        ],
        "nesca": nesca,
        "ladder": ladder,
        "megaplume_mean_power": megaplume_power,
        "observed_event_heat": events,
        "ratios": ratios,
        "vent_field_time_for_one_megaplume": vent_field_time,
        "megaplume_share_of_one_year_axial": {
            "low": float(floor_j / (ax["high_W"] * YEAR_S)),
            "high": float(ceil_j / (ax["low_W"] * YEAR_S)),
            "_note": (
                "Heat of one megaplume (floor, ceiling) as a fraction of one "
                "year of hydrothermal output of the global ridge axis "
                "(2-4 TW, Elderfield and Schultz 1996)."
            ),
        },
        "cross_checks": cross,
        "lava_map": lava_map,
        "dyke_map": dyke_map,
    }
    OUT.write_text(json.dumps(res, indent=1))

    b, m = ratios["blocky"], ratios["sheet"]
    print(
        f"NESCA blocky {nesca['blocky']['median_W'] / 1e12:.2f} TW, "
        f"sheet {nesca['sheet']['median_W'] / 1e12:.2f} TW"
    )
    print(
        f"  / global hydrothermal: {b['over_global_hydrothermal_central']:.3f} "
        f"(sheet {m['over_global_hydrothermal_central']:.3f})"
    )
    print(
        f"  / global ridge axis: {b['over_global_ridge_axis']} "
        f"(sheet {m['over_global_ridge_axis']})"
    )
    print(
        f"  / large vent field: {b['over_large_vent_field']:.0f} "
        f"(sheet {m['over_large_vent_field']:.0f})"
    )
    print(f"  mean smokers: {b['mean_black_smokers']:.3g}")
    print(
        f"  vent field time for floor: {vent_field_time['floor_days']:.0f} d, "
        f"ceiling {vent_field_time['ceiling_years']:.1f} yr"
    )
    print(
        f"  megaplume mean power at 15 h: "
        f"{[x / 1e12 for x in megaplume_power['mean_power_at_reference_W']]} TW"
    )
    print(f"  chimney-years EP86: {chimney_years}, consistent {cross['consistent']}")
    print(
        f"LAVA: E(15 h, 15 km2) = {e_ref_15h:.3e} J (rel err {reproduce:.1e}); "
        f"crossing {tau_cross / 3600:.1f} h; area factor {area_factor:.2f} "
        f"({area_factor * 15:.1f} km2); factor 7 at {tau_seven / 3600:.2f} h; "
        f"linearity {max(lin):.1e}"
    )
    print(
        f"DYKE: ceiling {e_ceiling:.3e} J; 5x1 km {e_plaus:.3e} J "
        f"({e_plaus / floor_j:.2f} of floor); needs {tau_plaus / 3600:.0f} h; "
        f"H needed {lh_needed / lp:.0f} m"
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
