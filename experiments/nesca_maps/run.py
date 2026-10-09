"""The NESCA posterior in plan view, and the joint posterior of its two calibrated modes.

Writes ``results/nesca_maps.json``.

The vent-sampled polydisperse inversion under blocky settling is rerun with the
same model, prior, sampler settings and random key as
``experiments/vent_shape/run.py``, and this time its draws are kept:

* thinned posterior samples of the total mass, the mass-weighted mean flux, the
  heat flux at that flux, the vent coordinates and the fraction weights;
* the posterior median and 95 % interval of nu on the flux nodes;
* the posterior predictive deposit against radius from the vent, per fraction;
* the posterior predictive deposit in plan view on a regular grid, per fraction,
  as the pointwise posterior median over a subsample of draws, each draw with
  its own vent;
* a posterior predictive check of the radial residual trend, computed as in
  ``experiments/nesca/run.py`` but with the vent sampled;
* the correlation of the two calibrated modes.

The run also compares its summaries with ``results/vent_shape.json`` so any
Monte Carlo drift is recorded next to the numbers.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "nesca"))

import jax  # noqa: E402
import numpyro  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402
from run import Q_NODES, load_all  # noqa: E402  (experiments/nesca/run.py)
from scipy import stats  # noqa: E402

from plume_inv import settling, stem  # noqa: E402
from plume_inv.constants import EPSILON_ENTRAIN  # noqa: E402
from plume_inv.data import FRACTIONS  # noqa: E402
from plume_inv.inverse import InversionData, nu_model_vent  # noqa: E402
from plume_inv.maps import deposit_xy  # noqa: E402

# experiments/nesca/run.py asks for four host devices on import.  One device
# keeps this run to the thread budget; the four chains are then drawn one after
# another from the same split keys, so the draws match the parallel run.
if os.environ.get("PLUME_PARALLEL_CHAINS", "0") != "1":
    numpyro.set_host_device_count(1)

SHAPE = "blocky"
N_THIN = 2000  # stored posterior samples
N_MAP_DRAWS = 400  # draws pushed through the plan-view predictive
GRID_X_KM = (-6.0, 6.0)
GRID_Y_KM = (-8.0, 7.5)
GRID_STEP_KM = 0.1


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
    """Core coordinates, deposit and settling speed per fraction.

    Identical to ``class_xy`` in experiments/vent_shape/run.py.
    """
    xy_all = np.c_[df["x_m"].values, df["y_m"].values]
    core_xy, obs, w_s, names = [], [], [], []
    for name, d_lo, d_hi, _m, _p in FRACTIONS:
        v = df[f"glass_g_per_m2[{name}]"].values / 1000.0
        good = keep & np.isfinite(v) & (v > 0)
        core_xy.append(xy_all[good])
        obs.append(v[good])
        w_s.append(
            float(settling.settling_velocity(settling.sieve_midpoint(d_lo, d_hi), shape=shape))
        )
        names.append(name)
    return core_xy, obs, w_s, names


def ci(v):
    v = np.asarray(v, dtype=float)
    return {
        "median": float(np.median(v)),
        "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
    }


def rel(a, b):
    return float((a - b) / b)


def main() -> int:
    df, on_flow, sigma_log, n_buoy = load_all()
    keep = ~on_flow
    k_heat = stem.heat_flux_constant()
    pref = k_heat * 0.187 * (n_buoy**5 / EPSILON_ENTRAIN**2) ** (1 / 3)
    lava_area_m2 = 15.0e6
    vent_prior_scale = float(np.sqrt(lava_area_m2) / 2.0)

    core_xy, obs, w_s, names = class_xy(df, keep, SHAPE)
    data = InversionData(
        q_nodes=Q_NODES,
        design=[],
        obs=obs,
        class_names=names,
        w_s=w_s,
        radii=[],
        sigma_log=sigma_log,
    )
    mcmc = MCMC(
        NUTS(nu_model_vent, target_accept_prob=0.9),
        num_warmup=1000,
        num_samples=1000,
        num_chains=4,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(0), data, core_xy, vent_prior_scale)
    s = mcmc.get_samples()
    summ = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True), prob=0.95)
    rhat = float(max(np.nanmax(v["r_hat"]) for v in summ.values()))
    n_div = int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))

    nu = np.asarray(s["nu"], dtype=float)
    vent = np.asarray(s["vent"], dtype=float)
    p_frac = np.asarray(s["p_frac"], dtype=float)
    sigma = np.asarray(s["sigma"], dtype=float)
    tot = nu.sum(axis=1)
    wgt = nu / tot[:, None]
    q_mean = (wgt * Q_NODES).sum(axis=1)
    phi = pref * q_mean ** (4 / 3)
    lq = np.log(Q_NODES)
    lq_sd = np.sqrt((wgt * (lq - (wgt * lq).sum(axis=1)[:, None]) ** 2).sum(axis=1))
    n_draw = tot.size
    print(f"draws={n_draw}  rhat={rhat:.4f}  divergences={n_div}", flush=True)

    # ---- reproduction against results/vent_shape.json -----------------------
    ref = json.loads((ROOT / "results" / "vent_shape.json").read_text())["by_shape"][SHAPE]
    post = {
        "mass_total_kg": ci(tot),
        "q_mass_weighted_mean_m^3s^-1": ci(q_mean),
        "phi_at_mass_weighted_mean_W": ci(phi),
        "vent_x_m": ci(vent[:, 0]),
        "vent_y_m": ci(vent[:, 1]),
        "vent_offset_from_centroid_m": ci(np.hypot(vent[:, 0], vent[:, 1])),
        "p_fractions": [ci(p_frac[:, i]) for i in range(p_frac.shape[1])],
        "sigma_log": ci(sigma),
        "log_q_sd": ci(lq_sd),
    }
    repro = {}
    for key in (
        "mass_total_kg",
        "q_mass_weighted_mean_m^3s^-1",
        "phi_at_mass_weighted_mean_W",
        "vent_x_m",
        "vent_y_m",
    ):
        a, b = post[key], ref[key]
        repro[key] = {
            "this_run_median": a["median"],
            "vent_shape_median": b["median"],
            "median_rel_diff": rel(a["median"], b["median"]),
            "this_run_ci95": a["ci95"],
            "vent_shape_ci95": b["ci95"],
            "ci95_rel_diff": [rel(a["ci95"][j], b["ci95"][j]) for j in (0, 1)],
        }
    # Monte Carlo standard error of each median from the effective sample size.
    mcse = {}
    ess_vent = np.asarray(summ["vent"]["n_eff"], dtype=float)
    for key, v, ess in (
        ("vent_x_m", vent[:, 0], ess_vent[0]),
        ("vent_y_m", vent[:, 1], ess_vent[1]),
    ):
        mcse[key] = float(1.2533 * np.std(v) / np.sqrt(ess))
    repro["_mcse_of_median_m"] = mcse
    repro["_ess_vent"] = [float(v) for v in ess_vent]
    # Chain-to-chain spread of each summary: the Monte Carlo error of the
    # interval ends is the standard deviation across chains over sqrt(chains).
    g_nu = np.asarray(mcmc.get_samples(group_by_chain=True)["nu"], dtype=float)
    g_tot = g_nu.sum(axis=2)
    g_q = (g_nu / g_tot[..., None] * Q_NODES).sum(axis=2)
    per_chain = {}
    for key, arr in (
        ("mass_total_kg", g_tot),
        ("q_mass_weighted_mean_m^3s^-1", g_q),
        ("phi_at_mass_weighted_mean_W", pref * g_q ** (4 / 3)),
    ):
        qs = np.percentile(arr, [2.5, 50.0, 97.5], axis=1)  # (3, chains)
        per_chain[key] = {
            "q025_by_chain": [float(v) for v in qs[0]],
            "median_by_chain": [float(v) for v in qs[1]],
            "q975_by_chain": [float(v) for v in qs[2]],
            "mcse_q025": float(qs[0].std(ddof=1) / np.sqrt(qs.shape[1])),
            "mcse_median": float(qs[1].std(ddof=1) / np.sqrt(qs.shape[1])),
            "mcse_q975": float(qs[2].std(ddof=1) / np.sqrt(qs.shape[1])),
        }
    repro["_per_chain_quantiles"] = per_chain
    q975 = per_chain["phi_at_mass_weighted_mean_W"]["q975_by_chain"]
    print(f"  Phi q97.5 by chain (TW): {[round(v / 1e12, 3) for v in q975]}", flush=True)
    for key, v in repro.items():
        if not key.startswith("_"):
            print(
                f"  {key:32s} this {v['this_run_median']:.4g}  "
                f"ref {v['vent_shape_median']:.4g}  "
                f"rel {v['median_rel_diff']:+.2e}",
                flush=True,
            )

    # ---- thinned samples ----------------------------------------------------
    step = max(1, n_draw // N_THIN)
    idx = np.arange(0, n_draw, step)[:N_THIN]

    def g6(a):
        return [float(f"{v:.6g}") for v in np.asarray(a, dtype=float)]

    samples = {
        "mass_total_kg": g6(tot[idx]),
        "q_mass_weighted_mean_m^3s^-1": g6(q_mean[idx]),
        "phi_at_mass_weighted_mean_W": g6(phi[idx]),
        "vent_x_m": g6(vent[idx, 0]),
        "vent_y_m": g6(vent[idx, 1]),
        "p_fractions": [g6(p_frac[idx, i]) for i in range(len(names))],
        "sigma_log": g6(sigma[idx]),
    }

    # ---- joint posterior of the two calibrated modes -----------------------
    pear = stats.pearsonr(np.log(tot), np.log(q_mean))
    spear = stats.spearmanr(tot, q_mean)
    joint = {
        "pearson_r_log_mass_log_qmean": float(pear.statistic),
        "spearman_rho_mass_qmean": float(spear.statistic),
        "n_draws": int(n_draw),
        "_note": (
            "Correlation over all posterior draws of the total mass and the "
            "mass-weighted mean flux."
        ),
    }
    print(
        f"  joint: pearson(log) = {joint['pearson_r_log_mass_log_qmean']:+.3f}  "
        f"spearman = {joint['spearman_rho_mass_qmean']:+.3f}",
        flush=True,
    )

    # ---- posterior predictive against radius --------------------------------
    rng = np.random.default_rng(20260923)
    sub = rng.choice(n_draw, size=1000, replace=False)
    rr = np.linspace(0.0, 9000.0, 181)
    radial = {"r_m": [float(v) for v in rr], "by_class": {}}
    for i, name in enumerate(names):
        g = (w_s[i] / Q_NODES)[None, :] * np.exp(
            -np.pi * w_s[i] * rr[:, None] ** 2 / Q_NODES[None, :]
        )  # (nr, K)
        pred = p_frac[sub, i][:, None] * (nu[sub] @ g.T)  # (ns, nr)
        radial["by_class"][name] = {
            "median": [float(v) for v in np.median(pred, axis=0)],
            "lo95": [float(v) for v in np.percentile(pred, 2.5, axis=0)],
            "hi95": [float(v) for v in np.percentile(pred, 97.5, axis=0)],
        }

    # ---- posterior predictive check with the vent sampled -------------------
    vent_med = np.median(vent, axis=0)
    sig_med = float(np.median(sigma))
    ppc = {}
    for i, name in enumerate(names):
        xy = core_xy[i]
        pred = np.empty((sub.size, xy.shape[0]))
        for j, d in enumerate(sub):
            pred[j] = deposit_xy(xy[:, 0], xy[:, 1], vent[d], Q_NODES, nu[d], w_s[i], p_frac[d, i])
        pmed = np.median(pred, axis=0)
        z = (np.log(obs[i]) - np.log(pmed)) / sig_med
        rkm = np.hypot(xy[:, 0] - vent_med[0], xy[:, 1] - vent_med[1]) / 1e3
        slope, intercept = np.polyfit(rkm, z, 1)
        n = z.size
        se = float(
            np.sqrt(np.sum((z - (slope * rkm + intercept)) ** 2) / (n - 2))
            / np.sqrt(np.sum((rkm - rkm.mean()) ** 2))
        )
        ppc[name] = {
            "n": int(n),
            "mean_standardised_residual": float(np.mean(z)),
            "sd_standardised_residual": float(np.std(z, ddof=1)),
            "radial_slope_per_km": float(slope),
            "radial_slope_se": se,
            "radial_slope_z": float(slope / se),
            "systematic_radial_trend": bool(abs(slope / se) > 3.0),
        }
    n_trend = int(sum(v["systematic_radial_trend"] for v in ppc.values()))
    print(
        f"  ppc: classes with |z| > 3 radial trend = {n_trend}; "
        f"max |z| = {max(abs(v['radial_slope_z']) for v in ppc.values()):.2f}",
        flush=True,
    )

    # ---- plan-view posterior predictive -------------------------------------
    gx = np.arange(GRID_X_KM[0], GRID_X_KM[1] + 1e-9, GRID_STEP_KM) * 1e3
    gy = np.arange(GRID_Y_KM[0], GRID_Y_KM[1] + 1e-9, GRID_STEP_KM) * 1e3
    X, Y = np.meshgrid(gx, gy)
    msub = sub[:N_MAP_DRAWS]
    maps = {
        "x_m": [float(v) for v in gx],
        "y_m": [float(v) for v in gy],
        "n_draws": int(msub.size),
        "statistic": "pointwise posterior median of Omega, stored as log10(kg m^-2)",
        "log10_omega": {},
    }
    for i, name in enumerate(names):
        stack = np.empty((msub.size,) + X.shape, dtype=np.float32)
        for j, d in enumerate(msub):
            stack[j] = deposit_xy(X, Y, vent[d], Q_NODES, nu[d], w_s[i], p_frac[d, i])
        med = np.median(stack, axis=0).astype(np.float64)
        maps["log10_omega"][name] = np.round(np.log10(med), 3).tolist()
        print(
            f"  map {name}: log10 range {np.log10(med).min():.2f} to {np.log10(med).max():.2f}",
            flush=True,
        )

    res = {
        "_description": (
            "NESCA polydisperse inversion with the vent sampled, blocky "
            "settling: posterior samples, predictive deposit in plan "
            "view and against radius, and the joint posterior of total "
            "mass and mass-weighted mean flux."
        ),
        "_generated": "2026-09-23",
        "_script": "experiments/nesca_maps/run.py",
        "_git_commit": git_hash(),
        "shape_used": SHAPE,
        "classes": names,
        "w_s_m s^-1": w_s,
        "q_nodes": [float(v) for v in Q_NODES],
        "phi_prefactor_W_per_Q^(4/3)": float(pref),
        "vent_prior": {"centre": "survey centroid", "scale_m": vent_prior_scale},
        "sampler": {
            "kernel": "NUTS",
            "target_accept_prob": 0.9,
            "num_warmup": 1000,
            "num_samples": 1000,
            "num_chains": 4,
            "rng_key": 0,
            "model": "plume_inv.inverse.nu_model_vent",
        },
        "diagnostics": {"max_rhat": rhat, "n_divergences": n_div, "n_draws": int(n_draw)},
        "n_cores_used": int(keep.sum()),
        "n_observations": int(sum(o.size for o in obs)),
        "posterior": post,
        "reproduction_vs_vent_shape": repro,
        "joint_mass_qmean": joint,
        "nu_median": [float(v) for v in np.median(nu, axis=0)],
        "nu_ci95": [
            [float(v) for v in np.percentile(nu, 2.5, axis=0)],
            [float(v) for v in np.percentile(nu, 97.5, axis=0)],
        ],
        "samples_thinned": samples,
        "samples_thin_step": int(step),
        "posterior_predictive_radial": radial,
        "posterior_predictive_by_class": ppc,
        "ppc_classes_with_systematic_radial_trend": n_trend,
        "vent_median_xy_m": [float(vent_med[0]), float(vent_med[1])],
        "maps": maps,
    }
    out = ROOT / "results" / "nesca_maps.json"
    out.write_text(json.dumps(res))
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
