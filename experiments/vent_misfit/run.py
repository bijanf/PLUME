"""The misfit of the NESCA deposit over trial vent positions.

Writes ``results/vent_misfit.json``.

For each trial vent on a regular grid about the survey centroid, the negative
log likelihood of the four-fraction deposit is evaluated with the shape of nu
held at its posterior median (from ``results/nesca_maps.json``), and with the
per-fraction amplitude and the common noise scale profiled out in closed form
(``plume_inv.maps.vent_misfit_grid``).  The surface is reported as the
difference from its minimum.

Adding the Gaussian vent prior gives the conditional posterior of the vent on
the grid, whose spread is compared with the marginal posterior from the sampler,
in which nu varies with the vent.

The widths of the 95 % heat-flux intervals drawn in Fig. 4b are recorded here
too, so every number on that figure has a results entry.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from plume_inv import settling  # noqa: E402
from plume_inv.data import FRACTIONS, load_nesca  # noqa: E402
from plume_inv.maps import vent_misfit_grid  # noqa: E402

HALF_WIDTH_M = 3000.0
STEP_M = 50.0
ZOOM_HALF_WIDTH_M = 700.0
ZOOM_STEP_M = 10.0


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


def class_xy(df, keep, shape):
    """As in experiments/vent_shape/run.py."""
    xy_all = np.c_[df["x_m"].values, df["y_m"].values]
    core_xy, obs, w_s = [], [], []
    for name, d_lo, d_hi, _m, _p in FRACTIONS:
        v = df[f"glass_g_per_m2[{name}]"].values / 1000.0
        good = keep & np.isfinite(v) & (v > 0)
        core_xy.append(xy_all[good])
        obs.append(v[good])
        w_s.append(
            float(settling.settling_velocity(settling.sieve_midpoint(d_lo, d_hi), shape=shape))
        )
    return core_xy, obs, w_s


def grid_moments(gx, gy, logp):
    """Mean, standard deviation and correlation of a density given on a grid."""
    p = np.exp(logp - logp.max())
    p /= p.sum()
    X, Y = np.meshgrid(gx, gy)
    mx, my = (p * X).sum(), (p * Y).sum()
    sx = np.sqrt((p * (X - mx) ** 2).sum())
    sy = np.sqrt((p * (Y - my) ** 2).sum())
    rho = (p * (X - mx) * (Y - my)).sum() / (sx * sy)
    return {
        "mean_xy_m": [float(mx), float(my)],
        "sd_xy_m": [float(sx), float(sy)],
        "corr_xy": float(rho),
    }


def bilinear(gx, gy, z, x, y):
    """Bilinear interpolation of a grid value at (x, y)."""
    i = int(np.clip(np.searchsorted(gx, x) - 1, 0, gx.size - 2))
    j = int(np.clip(np.searchsorted(gy, y) - 1, 0, gy.size - 2))
    tx = (x - gx[i]) / (gx[i + 1] - gx[i])
    ty = (y - gy[j]) / (gy[j + 1] - gy[j])
    return float(
        (1 - tx) * (1 - ty) * z[j, i]
        + tx * (1 - ty) * z[j, i + 1]
        + (1 - tx) * ty * z[j + 1, i]
        + tx * ty * z[j + 1, i + 1]
    )


def main() -> int:
    M = json.loads((ROOT / "results" / "nesca_maps.json").read_text())
    V = json.loads((ROOT / "results" / "vent_shape.json").read_text())
    U = json.loads((ROOT / "results" / "nesca_unsteady.json").read_text())
    shape = M["shape_used"]
    q_nodes = np.array(M["q_nodes"])
    nu_med = np.array(M["nu_median"])
    prior_scale = float(M["vent_prior"]["scale_m"])

    df = load_nesca()
    on_flow = df["Location"].astype(str).str.contains("on flow", case=False, na=False).values
    core_xy, obs, w_s = class_xy(df, ~on_flow, shape)

    gx = np.arange(-HALF_WIDTH_M, HALF_WIDTH_M + 1e-9, STEP_M)
    gy = np.arange(-HALF_WIDTH_M, HALF_WIDTH_M + 1e-9, STEP_M)
    wide = vent_misfit_grid(core_xy, obs, w_s, q_nodes, nu_med, gx, gy)
    d = wide["delta_nll"]
    print(
        f"wide grid {gx.size}x{gy.size}: min at {wide['argmin_xy']}, max dNLL {d.max():.1f}",
        flush=True,
    )

    # Fine grid about the minimum for the conditional posterior moments.
    cx, cy = wide["argmin_xy"]
    zx = np.arange(cx - ZOOM_HALF_WIDTH_M, cx + ZOOM_HALF_WIDTH_M + 1e-9, ZOOM_STEP_M)
    zy = np.arange(cy - ZOOM_HALF_WIDTH_M, cy + ZOOM_HALF_WIDTH_M + 1e-9, ZOOM_STEP_M)
    zoom = vent_misfit_grid(core_xy, obs, w_s, q_nodes, nu_med, zx, zy)
    ZX, ZY = np.meshgrid(zx, zy)
    zlp = -zoom["delta_nll"] - 0.5 * (ZX**2 + ZY**2) / prior_scale**2
    cond = grid_moments(zx, zy, zlp)
    k = np.unravel_index(int(np.argmax(zlp)), zlp.shape)
    cond["mode_xy_m"] = [float(ZX[k]), float(ZY[k])]

    # Marginal spread from the blocky run that the conditional surface is built on.
    Ms = M["samples_thinned"]
    mx, my = np.array(Ms["vent_x_m"]), np.array(Ms["vent_y_m"])
    marg = {
        "median_xy_m": [float(np.median(mx)), float(np.median(my))],
        "sd_xy_m": [float(mx.std()), float(my.std())],
        "corr_xy": float(np.corrcoef(mx, my)[0, 1]),
    }
    ratio = [cond["sd_xy_m"][0] / marg["sd_xy_m"][0], cond["sd_xy_m"][1] / marg["sd_xy_m"][1]]
    print(
        f"conditional sd {cond['sd_xy_m']}, marginal sd {marg['sd_xy_m']}, ratio {ratio}",
        flush=True,
    )
    print(f"minimum of the misfit at {zoom['argmin_xy']}", flush=True)

    # Misfit at reference points.
    prof_xy = U["vent_xy_m"]
    at = {
        "survey_centroid": bilinear(gx, gy, d, 0.0, 0.0),
        "posterior_median_vent": bilinear(gx, gy, d, *marg["median_xy_m"]),
        "least_squares_profiled_vent": bilinear(gx, gy, d, *prof_xy),
    }
    print(
        f"dNLL at centroid {at['survey_centroid']:.1f}, at posterior median "
        f"{at['posterior_median_vent']:.2f}, at profiled {at['least_squares_profiled_vent']:.2f}"
    )

    # Heat-flux interval widths drawn in Fig. 4b.
    prof = U["posterior"]["phi_at_mass_weighted_mean_W"]
    samp = V["by_shape"]["blocky"]["phi_at_mass_weighted_mean_W"]
    wp = prof["ci95"][1] - prof["ci95"][0]
    wsd = samp["ci95"][1] - samp["ci95"][0]
    widths = {
        "profiled_W": float(wp),
        "sampled_W": float(wsd),
        "ratio_sampled_over_profiled": float(wsd / wp),
        "_source": (
            "results/nesca_unsteady.json#posterior."
            "phi_at_mass_weighted_mean_W.ci95 and "
            "results/vent_shape.json#by_shape.blocky."
            "phi_at_mass_weighted_mean_W.ci95"
        ),
    }
    print(
        f"Phi 95% widths: profiled {wp / 1e12:.2f} TW, sampled {wsd / 1e12:.2f} TW, "
        f"ratio {wsd / wp:.2f}"
    )

    res = {
        "_description": (
            "Negative log likelihood of the NESCA deposit over trial vent "
            "positions, nu shape at its posterior median, per-fraction "
            "amplitudes and noise scale profiled; conditional vent "
            "posterior against the sampled marginal; widths of the "
            "heat-flux intervals in Fig. 4b."
        ),
        "_generated": "2026-09-23",
        "_script": "experiments/vent_misfit/run.py",
        "_git_commit": git_hash(),
        "shape_used": shape,
        "held_fixed": "shape of nu at its posterior median, results/nesca_maps.json#nu_median",
        "profiled": [
            "per-fraction amplitude (fraction weight times total mass)",
            "common log-space noise scale",
        ],
        "n_obs": wide["n_obs"],
        "grid": {
            "x_m": [float(v) for v in gx],
            "y_m": [float(v) for v in gy],
            "delta_nll": np.round(d, 3).tolist(),
            "argmin_xy_m": wide["argmin_xy"],
            "sigma_at_min": wide["sigma_at_min"],
        },
        "zoom_grid": {
            "half_width_m": ZOOM_HALF_WIDTH_M,
            "step_m": ZOOM_STEP_M,
            "x_m": [float(v) for v in zx],
            "y_m": [float(v) for v in zy],
            "delta_nll": np.round(zoom["delta_nll"], 4).tolist(),
            "log_conditional_posterior": np.round(zlp - zlp.max(), 4).tolist(),
            "argmin_xy_m": zoom["argmin_xy"],
            "sigma_at_min": zoom["sigma_at_min"],
        },
        "delta_nll_at": at,
        "reference_points_xy_m": {
            "survey_centroid": [0.0, 0.0],
            "posterior_median_vent": marg["median_xy_m"],
            "least_squares_profiled_vent": prof_xy,
        },
        "conditional_posterior_vent": cond,
        "marginal_posterior_vent": marg,
        "conditional_over_marginal_sd": ratio,
        "vent_prior_scale_m": prior_scale,
        "phi_ci95_widths": widths,
    }
    out = ROOT / "results" / "vent_misfit.json"
    out.write_text(json.dumps(res))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
