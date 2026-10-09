"""The vent position as a parameter, and a shape-marginalised heat flux.

Writes ``results/vent_shape.json``.

Two things a fixed-vent, fixed-shape inversion leaves open:

1. The vent position was profiled out by least squares and then held fixed while
   the posterior was sampled.  The quasi-steady kernel depends on the core radius
   squared, so the vent enters the operator itself and fixing it understates the
   width of every posterior downstream of it.  Here the vent is sampled, under a
   prior centred on the survey centroid whose scale is set by the mapped flow.

2. The heat flux was reported at two end-member clast shapes with no statement of
   which the deposit prefers.  Each volcaniclast shape is inverted separately and
   the three posteriors are combined into one mixture, weighted by the marginal
   likelihood of each shape from the four-fraction chi-squared fit.
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
sys.path.insert(0, str(ROOT / "experiments" / "nesca"))

import jax  # noqa: E402
import numpyro  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402
from run import Q_NODES, load_all  # noqa: E402  (experiments/nesca/run.py)

from plume_inv import settling, stem  # noqa: E402
from plume_inv.constants import EPSILON_ENTRAIN  # noqa: E402
from plume_inv.data import FRACTIONS  # noqa: E402
from plume_inv.inverse import InversionData, nu_model_vent  # noqa: E402

numpyro.set_host_device_count(4)
SHAPES = ("blocky", "long", "sheet")


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
    """Core coordinates, deposit and settling speed for each sieve fraction."""
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
    return {
        "median": float(np.median(v)),
        "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
    }


def main() -> int:
    df, on_flow, sigma_log, n_buoy = load_all()
    keep = ~on_flow
    k_heat = stem.heat_flux_constant()
    pref = k_heat * 0.187 * (n_buoy**5 / EPSILON_ENTRAIN**2) ** (1 / 3)

    # Prior on the vent: centred on the survey centroid, which is the origin of
    # the local frame, with a scale of half the characteristic dimension of the
    # mapped flow, sqrt(A)/2 with A the mapped lava area.  Nothing here is fitted
    # to the deposit.
    lava_area_m2 = 15.0e6
    vent_prior_scale = float(np.sqrt(lava_area_m2) / 2.0)

    shape_fits = json.loads((ROOT / "results" / "clast_shape.json").read_text())
    by_shape, chi2 = {}, {}
    for shape in SHAPES:
        core_xy, obs, w_s, names = class_xy(df, keep, shape)
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

        nu = np.asarray(s["nu"])
        vent = np.asarray(s["vent"])
        tot = nu.sum(axis=1)
        wgt = nu / tot[:, None]
        q_mean = (wgt * Q_NODES).sum(axis=1)
        phi = pref * q_mean ** (4 / 3)
        chi2[shape] = float(shape_fits["by_shape"][shape]["chi2"])
        by_shape[shape] = {
            "w_s_m s^-1": w_s,
            "chi2_four_fraction": chi2[shape],
            "max_rhat": rhat,
            "vent_x_m": ci(vent[:, 0]),
            "vent_y_m": ci(vent[:, 1]),
            "vent_offset_from_centroid_m": ci(np.hypot(vent[:, 0], vent[:, 1])),
            "mass_total_kg": ci(tot),
            "q_mass_weighted_mean_m^3s^-1": ci(q_mean),
            "phi_at_mass_weighted_mean_W": ci(phi),
            "_phi_samples_n": int(phi.size),
        }
        by_shape[shape]["_phi"] = phi
        by_shape[shape]["_vent"] = vent
        print(
            f"  {shape:7s} vent=({np.median(vent[:, 0]):7.0f},{np.median(vent[:, 1]):7.0f}) m "
            f"+/-({vent[:, 0].std():5.0f},{vent[:, 1].std():5.0f})  "
            f"Phi={np.median(phi) / 1e12:5.2f} TW  chi2={chi2[shape]:5.2f}  "
            f"rhat={rhat:.3f}",
            flush=True,
        )

    # Shape weights from the four-fraction chi-squared fit, w ~ exp(-chi2/2),
    # normalised over the three volcaniclast shapes.  This is the marginal
    # likelihood of the shape under a flat prior across the three.
    c = np.array([chi2[s] for s in SHAPES])
    w = np.exp(-(c - c.min()) / 2.0)
    w = w / w.sum()
    weights = {s: float(wi) for s, wi in zip(SHAPES, w, strict=True)}

    rng = np.random.default_rng(0)
    n_draw = 4000
    counts = rng.multinomial(n_draw, w)
    phi_mix = np.concatenate(
        [
            rng.choice(by_shape[s]["_phi"], size=int(n), replace=True)
            for s, n in zip(SHAPES, counts, strict=True)
        ]
    )
    vent_mix = np.concatenate(
        [
            by_shape[s]["_vent"][
                rng.choice(by_shape[s]["_vent"].shape[0], size=int(n), replace=True)
            ]
            for s, n in zip(SHAPES, counts, strict=True)
        ]
    )

    phi_by_shape = {s: [float(v) for v in by_shape[s]["_phi"][::4]] for s in SHAPES}
    for s in SHAPES:
        by_shape[s].pop("_phi")
        by_shape[s].pop("_vent")

    res = {
        "_description": (
            "Vent position sampled as a parameter, and a heat flux marginalised over clast shape."
        ),
        "_generated": datetime.date.today().isoformat(),
        "_script": "experiments/vent_shape/run.py",
        "_git_commit": git_hash(),
        "vent_prior": {
            "centre": "survey centroid (origin of the local frame)",
            "scale_m": vent_prior_scale,
            "basis": (
                "sqrt(A)/2 with A the mapped lava area, results/manifest.json#sources.lava_area_m2"
            ),
        },
        "by_shape": by_shape,
        "shape_weights": weights,
        "shape_marginal": {
            "phi_W": ci(phi_mix),
            "vent_x_m": ci(vent_mix[:, 0]),
            "vent_y_m": ci(vent_mix[:, 1]),
            "vent_offset_from_centroid_m": ci(np.hypot(vent_mix[:, 0], vent_mix[:, 1])),
        },
        "phi_samples_W": {
            "by_shape": phi_by_shape,
            "shape_marginal": [float(v) for v in phi_mix[::2]],
        },
        "vent_posterior_samples": {
            "x_m": [float(v) for v in vent_mix[::8, 0]],
            "y_m": [float(v) for v in vent_mix[::8, 1]],
        },
    }
    out = ROOT / "results" / "vent_shape.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nshape weights: {weights}")
    pm = res["shape_marginal"]["phi_W"]
    print(
        f"shape-marginalised Phi = {pm['median'] / 1e12:.2f} TW "
        f"[{pm['ci95'][0] / 1e12:.2f}, {pm['ci95'][1] / 1e12:.2f}]"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
