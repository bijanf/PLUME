"""Simulation-based calibration of the nu(Q) model.

Draws parameters from the prior, simulates a deposit on the real NESCA geometry,
refits, and records the rank of each true value within its posterior draws.  If
the sampler and model are self-consistent the ranks are uniform; departures say
which quantities are being reported dishonestly.

Writes ``results/sbc.json``.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
from numpyro.infer import MCMC, NUTS, Predictive  # noqa: E402

from plume_inv import kernel as K  # noqa: E402
from plume_inv.inverse import InversionData, nu_model  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "synthetic"))
from run import load_geometry, observed_fractions  # noqa: E402

OUT = ROOT / "results" / "sbc.json"
Q_NODES = np.geomspace(3e4, 3e7, 24)
N_REP = 48
N_POST = 400


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
    radii, w_s, names, sigma_log, n_buoy, _centre = load_geometry()
    design = [K.design_matrix(r, Q_NODES, w, 1.0) for r, w in zip(radii, w_s, strict=True)]
    observed_fractions()  # ensure the geometry loader and data agree
    dummy_obs = [np.ones(r.size) for r in radii]
    base = InversionData(
        q_nodes=Q_NODES,
        design=design,
        obs=dummy_obs,
        class_names=names,
        w_s=w_s,
        radii=radii,
        sigma_log=sigma_log,
    )

    prior = Predictive(nu_model, num_samples=N_REP)(jax.random.PRNGKey(0), base)
    ranks = {"mass_total": [], "q_mass_weighted_mean": [], "log_q_sd": [], "sigma": []}
    n_div = 0

    for rep in range(N_REP):
        nu_t = np.asarray(prior["nu"][rep])
        p_t = np.asarray(prior["p_frac"][rep])
        sig_t = float(np.asarray(prior["sigma"][rep]))
        rng = np.random.default_rng(1000 + rep)
        obs = []
        for a, p in zip(design, p_t, strict=True):
            clean = p * (a @ nu_t)
            obs.append(np.maximum(clean, 1e-30) * np.exp(rng.normal(0.0, sig_t, size=clean.size)))
        data = InversionData(
            q_nodes=Q_NODES,
            design=design,
            obs=obs,
            class_names=names,
            w_s=w_s,
            radii=radii,
            sigma_log=sigma_log,
        )
        mcmc = MCMC(
            NUTS(nu_model, target_accept_prob=0.9),
            num_warmup=400,
            num_samples=N_POST,
            num_chains=1,
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(5000 + rep), data)
        post = mcmc.get_samples()
        extra = mcmc.get_extra_fields()
        if extra and "diverging" in extra:
            n_div += int(np.sum(np.asarray(extra["diverging"])))

        nu_p = np.asarray(post["nu"])
        sig_p = np.asarray(post["sigma"])

        def stat(nu, key):
            tot = nu.sum(axis=-1)
            if key == "mass_total":
                return tot
            wgt = nu / tot[..., None]
            lq = np.log(Q_NODES)
            mean_lq = (wgt * lq).sum(axis=-1)
            if key == "q_mass_weighted_mean":
                return (wgt * Q_NODES).sum(axis=-1)
            return np.sqrt((wgt * (lq - mean_lq[..., None]) ** 2).sum(axis=-1))

        for key in ("mass_total", "q_mass_weighted_mean", "log_q_sd"):
            t = float(stat(nu_t[None, :], key)[0])
            ranks[key].append(int(np.sum(stat(nu_p, key) < t)))
        ranks["sigma"].append(int(np.sum(sig_p < sig_t)))
        print(f"  rep {rep + 1}/{N_REP}", flush=True)

    out = {
        "_description": "Simulation-based calibration of the nonparametric nu(Q) model.",
        "_generated": "2026-09-17",
        "_script": "experiments/synthetic/sbc.py",
        "_git_commit": git_hash(),
        "n_replicates": N_REP,
        "n_posterior_draws": N_POST,
        "n_divergences_total": n_div,
        "ranks": {k: v for k, v in ranks.items()},
        "uniformity": {},
    }
    from scipy import stats

    for k, v in ranks.items():
        u = (np.array(v) + 0.5) / (N_POST + 1)
        ks = stats.kstest(u, "uniform")
        out["uniformity"][k] = {
            "ks_statistic": float(ks.statistic),
            "p_value": float(ks.pvalue),
            "uniform_at_5pct": bool(ks.pvalue > 0.05),
            "mean_rank_fraction": float(np.mean(u)),
        }
    OUT.write_text(json.dumps(out, indent=2))
    for k, v in out["uniformity"].items():
        print(
            f"  {k:22s} KS={v['ks_statistic']:.3f} p={v['p_value']:.3f} "
            f"{'uniform' if v['uniform_at_5pct'] else 'NOT UNIFORM'}"
        )
    print(f"divergences: {n_div}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
