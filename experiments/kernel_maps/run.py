"""Space-time and settling-speed maps of the front-limited deposit (Figure 1e-g).

Writes ``results/kernel_maps.json`` (every quotable number) and
``results/kernel_maps_grids.npz`` (the gridded fields the figure draws).

Reference case, identical to Figure 1a-d: a constant umbrella flux
Q = 7.6e5 m^3 s^-1 for 15 h, 1e9 kg released per size class, N from
``results/stratification.json``, front coefficient LAMBDA_FRONT, blocky
settling law.  Radii of the cores are measured from the sampled vent of
``results/vent_shape.json`` (blocky median), the vent the paper uses.

Three products:

1. Timing.  When the front passes the innermost, median and outermost cored
   radius; what fraction of each class is on the seafloor by then and by the
   source shut-off; when half of the local deposit has landed at each radius.
2. Space-time field.  The deposition rate D(r, t) of Eq. (S1) in
   ``plume_inv.spacetime`` and the cumulative fraction of the local deposit, for
   the four sieve fractions.
3. Settling-speed map.  The front-limited deposit and its ratio to the
   quasi-steady deposit over radius and settling speed, 1 mm s^-1 to
   15 cm s^-1, with the four fractions on the grid; the radii where the ratio
   crosses one.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from plume_inv import forward as fwd  # noqa: E402
from plume_inv import kernel as K  # noqa: E402
from plume_inv import settling, umbrella  # noqa: E402
from plume_inv import spacetime as S  # noqa: E402
from plume_inv.constants import LAMBDA_FRONT  # noqa: E402
from plume_inv.data import FRACTIONS, load_nesca  # noqa: E402

OUT = ROOT / "results" / "kernel_maps.json"
GRIDS = ROOT / "results" / "kernel_maps_grids.npz"

Q0 = 7.6e5
MASS = 1.0e9
TAU = 15 * 3600.0
FLOOR = 1e-2  # kg m^-2, the mask of Figure 1c
N_QUAD = 4096
N_EXTRA = 2500
W_MIN, W_MAX, N_W = 1.0e-3, 0.15, 64
R_MAP = np.arange(50.0, 9500.0 + 1.0, 50.0)
R_FINE = np.arange(10.0, 9500.0 + 1.0, 10.0)
T_PLOT_H = np.linspace(0.0, 40.0, 481)
N_T_RATE = 6000

CLASSES = [
    ("63-125", "63-125um"),
    ("125-250", "125-250um"),
    ("250-500", "250-500um"),
    ("500-1000", ">500um"),
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


def f_or_none(x) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def class_speeds() -> dict[str, float]:
    out = {}
    for label, name in CLASSES:
        row = next(f for f in FRACTIONS if f[0] == name)
        out[label] = float(settling.settling_velocity(settling.sieve_midpoint(row[1], row[2])))
    return out


def cored_radii() -> dict:
    vs = json.loads((ROOT / "results" / "vent_shape.json").read_text())
    vx = vs["by_shape"]["blocky"]["vent_x_m"]["median"]
    vy = vs["by_shape"]["blocky"]["vent_y_m"]["median"]
    df = load_nesca()
    on_flow = df["Location"].astype(str).str.contains("on flow", case=False, na=False).values
    r = np.hypot(df["x_m"].values - vx, df["y_m"].values - vy)[~on_flow]
    return {
        "vent_source": (
            "results/vent_shape.json#by_shape.blocky.vent_x_m.median, "
            "#by_shape.blocky.vent_y_m.median"
        ),
        "vent_xy_m": [float(vx), float(vy)],
        "n_cores": int(r.size),
        "cores": "all cores not logged 'on flow' (the polydisperse set)",
        "r_min_m": float(r.min()),
        "r_median_m": float(np.median(r)),
        "r_max_m": float(r.max()),
        "r_p05_m": float(np.percentile(r, 5)),
        "r_p95_m": float(np.percentile(r, 95)),
    }


def record(w: float, n_buoy: float, q0: float = Q0):
    """Extended constant-flux record and its umbrella, as in fig_kernels.unsteady."""
    src = fwd.constant_history(q0, MASS, TAU, n_t=3001)
    t_end = K.settling_window(float(src.t[-1]), float(np.mean(src.q)), n_buoy, w, LAMBDA_FRONT)
    t, q, m = K.extend_history(src.t, src.q, src.mdot, t_end, n_extra=N_EXTRA)
    umb = umbrella.solve_umbrella(lambda tt: float(np.interp(tt, t, q)), t, n_buoy, LAMBDA_FRONT)
    return src, t, q, m, umb, t_end


def ratio_pair(r, w, n_buoy, q0=Q0):
    src, t, q, m, umb, t_end = record(w, n_buoy, q0)
    u = K.deposit_unsteady(r, umb, t, q, m, w, 1.0, n_quad=N_QUAD)
    g = K.deposit_quasi_steady(r, src.t, src.q, src.mdot, w, 1.0)
    return u, g, t_end


def masked_ratio(u, g):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(g > FLOOR, u / g, np.nan)


def main() -> int:
    strat = json.loads((ROOT / "results" / "stratification.json").read_text())
    n_buoy = strat["N_primary"]["value_s^-1"]
    w_cls = class_speeds()
    cores = cored_radii()
    r_lo, r_md, r_hi = cores["r_min_m"], cores["r_median_m"], cores["r_max_m"]

    res: dict = {
        "_description": (
            "Figure 1e-g: space-time and settling-speed maps of the "
            "front-limited deposit at the reference constant-flux case."
        ),
        "_generated": "2026-09-23",
        "_script": "experiments/kernel_maps/run.py",
        "_git_commit": git_hash(),
        "_grids_file": "results/kernel_maps_grids.npz",
        "parameters": {
            "q_umb_m^3s^-1": Q0,
            "mass_per_class_kg": MASS,
            "duration_s": TAU,
            "N_s^-1": n_buoy,
            "N_source": "results/stratification.json#N_primary.value_s^-1",
            "lambda_front": LAMBDA_FRONT,
            "shape": settling.DEFAULT_SHAPE,
            "mask_floor_kg_m^-2": FLOOR,
            "w_s_by_fraction_m s^-1": w_cls,
            "w_grid_m s^-1": [W_MIN, W_MAX, N_W],
            "r_grid_map_m": [float(R_MAP[0]), float(R_MAP[-1]), int(R_MAP.size)],
            "r_grid_fine_m": [float(R_FINE[0]), float(R_FINE[-1]), int(R_FINE.size)],
            "n_quad": N_QUAD,
            "settling_residual": 1e-5,
        },
        "cored_radii": cores,
    }

    # ---- 1. front timing -------------------------------------------------
    t_long = np.linspace(0.0, 60 * 3600.0, 12001)
    umb_ref = umbrella.solve_umbrella(
        lambda tt: Q0 if tt <= TAU else 0.0, t_long, n_buoy, LAMBDA_FRONT
    )
    t_at = {
        k: float(umb_ref.time_front_reaches(v))
        for k, v in (("r_min", r_lo), ("r_median", r_md), ("r_max", r_hi))
    }
    res["front"] = {
        "front_radius_at_shutoff_m": float(np.interp(TAU, t_long, umb_ref.front_radius)),
        "thickness_at_shutoff_m": float(np.interp(TAU, t_long, umb_ref.thickness)),
        "t_reaches_r_min_h": t_at["r_min"] / 3600.0,
        "t_reaches_r_median_h": t_at["r_median"] / 3600.0,
        "t_reaches_r_max_h": t_at["r_max"] / 3600.0,
        "t_reaches_r_max_over_duration": t_at["r_max"] / TAU,
        "crossing_time_r_min_to_r_max_h": (t_at["r_max"] - t_at["r_min"]) / 3600.0,
        "crossing_time_over_duration": (t_at["r_max"] - t_at["r_min"]) / TAU,
    }

    # ---- 2. space-time field and deposition timing, four fractions -----------
    rate_plot, cum_plot, timing = {}, {}, {}
    t_plot = T_PLOT_H * 3600.0
    for label, _name in CLASSES:
        w = w_cls[label]
        src, t, q, m, umb, t_end = record(w, n_buoy)
        u_final = K.deposit_unsteady(R_MAP, umb, t, q, m, w, 1.0, n_quad=N_QUAD)
        t_obs = np.linspace(0.0, t[-1], N_T_RATE)
        rate = S.deposition_rate_unsteady(R_MAP, t_obs, umb, t, q, m, w, 1.0)
        cum = np.concatenate(
            (
                np.zeros((R_MAP.size, 1)),
                np.cumsum(0.5 * np.diff(t_obs)[None, :] * (rate[:, 1:] + rate[:, :-1]), axis=1),
            ),
            axis=1,
        )
        total = cum[:, -1]
        ok = u_final > 1e-6 * u_final.max()
        rate_vs_kernel = float(np.max(np.abs(total[ok] / u_final[ok] - 1.0)))
        with np.errstate(divide="ignore", invalid="ignore"):
            frac_local = np.where(total[:, None] > 0, cum / total[:, None], np.nan)

        def t_half_local(r0, frac_local=frac_local, t_obs=t_obs):
            j = int(np.argmin(np.abs(R_MAP - r0)))
            f = frac_local[j]
            return float(np.interp(0.5, f, t_obs)) / 3600.0, float(R_MAP[j])

        def after_shutoff_local(r0, frac_local=frac_local, t_obs=t_obs):
            j = int(np.argmin(np.abs(R_MAP - r0)))
            return 1.0 - float(np.interp(TAU, t_obs, frac_local[j]))

        f_glob = S.deposited_fraction_unsteady(t_obs, umb, t, m, w)
        # Area integral of the rate field against the global bookkeeping at shut-off.
        k_off = int(np.searchsorted(t_obs, TAU))
        area_int = float(np.trapezoid(2 * np.pi * R_MAP * cum[:, k_off], R_MAP)) / MASS
        th_lo, rj_lo = t_half_local(r_lo)
        th_md, rj_md = t_half_local(r_md)
        th_hi, rj_hi = t_half_local(r_hi)
        timing[label] = {
            "w_s_m s^-1": w,
            "settling_window_over_duration": t_end / TAU,
            "fraction_on_seafloor_at_shutoff": float(np.interp(TAU, t_obs, f_glob)),
            "fraction_on_seafloor_when_front_reaches_r_max": float(
                np.interp(t_at["r_max"], t_obs, f_glob)
            ),
            "fraction_on_seafloor_when_front_reaches_r_median": float(
                np.interp(t_at["r_median"], t_obs, f_glob)
            ),
            "t_half_of_class_mass_on_seafloor_h": float(np.interp(0.5, f_glob, t_obs)) / 3600.0,
            "t_90pc_of_class_mass_on_seafloor_h": float(np.interp(0.9, f_glob, t_obs)) / 3600.0,
            "t_half_local_deposit_h": {"r_min": th_lo, "r_median": th_md, "r_max": th_hi},
            "t_half_local_grid_radius_m": {"r_min": rj_lo, "r_median": rj_md, "r_max": rj_hi},
            "local_deposit_fraction_landing_after_shutoff": {
                "r_min": after_shutoff_local(r_lo),
                "r_median": after_shutoff_local(r_md),
                "r_max": after_shutoff_local(r_hi),
            },
            "check_rate_integral_vs_kernel_max_rel": rate_vs_kernel,
            "check_area_integral_at_shutoff_vs_bookkeeping": {
                "area_integral_fraction": area_int,
                "bookkeeping_fraction": float(np.interp(TAU, t_obs, f_glob)),
            },
        }
        rate_plot[label] = np.array(
            [np.interp(t_plot, t_obs, row) for row in rate], dtype=np.float32
        )
        cum_plot[label] = np.array(
            [np.interp(t_plot, t_obs, row) for row in frac_local], dtype=np.float32
        )
        print(f"{label}: window {t_end / TAU:.2f} tau, rate-vs-kernel {rate_vs_kernel:.1e}")
    res["deposition_timing"] = timing

    # ---- 3. settling-speed map ---------------------------------------------
    w_grid = np.unique(
        np.concatenate([np.geomspace(W_MIN, W_MAX, N_W), np.array(list(w_cls.values()))])
    )
    log_u = np.full((w_grid.size, R_MAP.size), np.nan)
    log_g = np.full_like(log_u, np.nan)
    ratio = np.full_like(log_u, np.nan)
    r_cross_map = []
    for i, w in enumerate(w_grid):
        u, g, _te = ratio_pair(R_MAP, w, n_buoy)
        with np.errstate(divide="ignore"):
            log_u[i] = np.log10(np.maximum(u, 1e-300))
            log_g[i] = np.log10(np.maximum(g, 1e-300))
        ratio[i] = masked_ratio(u, g)
        r_cross_map.append(S.unit_crossings(R_MAP, ratio[i]))
    first_cross = np.array([c[0] if c else np.nan for c in r_cross_map])

    per_class = {}
    for label, _name in CLASSES:
        w = w_cls[label]
        u, g, _te = ratio_pair(R_FINE, w, n_buoy)
        rat = masked_ratio(u, g)
        vis = np.where(g > FLOOR)[0]
        annulus = (R_FINE >= r_lo) & (R_FINE <= r_hi) & np.isfinite(rat)
        length = float(np.sqrt(Q0 / w))
        cross = S.unit_crossings(R_FINE, rat)
        per_class[label] = {
            "w_s_m s^-1": w,
            "gaussian_length_L_m": length,
            "ratio_one_crossings_m": cross,
            "ratio_one_crossings_over_L": [c / length for c in cross],
            "mask_edge_m": float(R_FINE[vis[-1]]) if vis.size else None,
            "ratio_at_r_min": f_or_none(np.interp(r_lo, R_FINE, rat)),
            "ratio_at_r_max": f_or_none(np.interp(r_hi, R_FINE, rat)),
            "ratio_min_in_cored_annulus": (
                f_or_none(np.min(rat[annulus])) if annulus.any() else None
            ),
            "ratio_max_in_cored_annulus": (
                f_or_none(np.max(rat[annulus])) if annulus.any() else None
            ),
            "ratio_at_1km": f_or_none(np.interp(1000.0, R_FINE, rat)),
            "ratio_at_8km": f_or_none(np.interp(8000.0, R_FINE, rat)),
            "front_limited_deposit_at_r_max_kg_m^-2": float(np.interp(r_hi, R_FINE, u)),
            "quasi_steady_deposit_at_r_max_kg_m^-2": float(np.interp(r_hi, R_FINE, g)),
        }
    in_map = np.isfinite(ratio)
    has_cross = np.isfinite(first_cross)
    cross_over_l = first_cross[has_cross] / np.sqrt(Q0 / w_grid[has_cross])
    res["ratio_map"] = {
        "per_fraction": per_class,
        "ratio_min_in_map": float(np.nanmin(ratio)),
        "ratio_max_in_map": float(np.nanmax(ratio)),
        "fraction_of_unmasked_map_below_one": float(np.mean(ratio[in_map] < 1.0)),
        "first_crossing_m_at_w_min": f_or_none(first_cross[0]),
        "first_crossing_m_at_w_max": f_or_none(first_cross[-1]),
        "first_crossing_over_L_min": float(cross_over_l.min()),
        "first_crossing_over_L_max": float(cross_over_l.max()),
        "w_range_with_crossing_inside_map_m s^-1": [
            float(w_grid[has_cross].min()),
            float(w_grid[has_cross].max()),
        ],
        "note": (
            "Ratio masked where the quasi-steady deposit is below the floor, as in "
            "Figure 1c.  Crossings listed from the vent outwards."
        ),
    }

    # ---- 4. reproduction check against forward_checks ------------------------
    fc = json.loads((ROOT / "results" / "forward_checks.json").read_text())
    ref = fc["unsteady_kernel"]["ratio_unsteady_to_quasi_steady"]
    r_chk = np.array([1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3])
    u, g, _te = ratio_pair(r_chk, 0.03, 1.0e-3)
    mine = u / g
    theirs = np.array([ref[f"{k}km"] for k in range(1, 9)])
    res["reproduction_check"] = {
        "against": "results/forward_checks.json#unsteady_kernel.ratio_unsteady_to_quasi_steady",
        "case": "N = 1e-3 s^-1, w = 0.03 m s^-1, same Q, mass and duration",
        "ratio_here": {f"{k}km": float(v) for k, v in zip(range(1, 9), mine, strict=True)},
        "max_abs_rel_difference": float(np.max(np.abs(mine / theirs - 1.0))),
    }
    print("reproduction max rel diff:", res["reproduction_check"]["max_abs_rel_difference"])

    np.savez_compressed(
        GRIDS,
        r_km=R_MAP / 1e3,
        t_h=T_PLOT_H,
        front_t_h=t_long / 3600.0,
        front_r_km=umb_ref.front_radius / 1e3,
        w_grid=w_grid,
        log10_omega_front=log_u,
        log10_omega_quasi=log_g,
        ratio=ratio,
        ratio_first_crossing_m=first_cross,
        class_labels=np.array([c[0] for c in CLASSES]),
        class_w=np.array([w_cls[c[0]] for c in CLASSES]),
        **{f"rate_{c[0]}": rate_plot[c[0]] for c in CLASSES},
        **{f"cumfrac_{c[0]}": cum_plot[c[0]] for c in CLASSES},
    )
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} and {GRIDS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
