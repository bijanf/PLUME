"""The NESCA inversion.

Writes ``results/nesca_steady.json`` and ``results/nesca_unsteady.json``.

Order matters.  The steady, single-class inversion must reproduce Pegler &
Ferguson (2021) within their stated uncertainty BEFORE anything unsteady is run;
that gate is enforced here in code.

What is compared, and why it is L and not Q
-------------------------------------------
The deposit constrains the dispersal length scale L.  Q = w_s L^2 and
Phi ~ Q^(4/3) inherit the settling speed multiplicatively, and the settling speed
depends on clast shape by a factor 2.2 across the measured classes
(``results/forward_checks.json#settling_law``).  So the gate is on L, and the
Q and Phi consequences are reported for each shape hypothesis separately.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import subprocess
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
import numpyro  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402

from plume_inv import kernel as K  # noqa: E402
from plume_inv import settling, stem  # noqa: E402
from plume_inv.constants import EPSILON_ENTRAIN  # noqa: E402
from plume_inv.data import FRACTIONS, load_nesca  # noqa: E402
from plume_inv.inverse import InversionData, fit_steady_lsq, nu_model  # noqa: E402

numpyro.set_host_device_count(4)
Q_NODES = np.geomspace(3e4, 3e7, 24)
BENCH = {
    "L_m": 4.9e3,
    "L_sigma_m": 0.4e3,
    "q_umb": 7.6e5,
    "q_sigma": 3.6e5,
    "phi_W": 1.5e12,
    "phi_sigma_W": 0.9e12,
    "w_s": 0.03,
}


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


def load_all():
    n_buoy = json.loads((ROOT / "results" / "stratification.json").read_text())["N_primary"][
        "value_s^-1"
    ]
    ident = json.loads((ROOT / "results" / "identifiability.json").read_text())
    sigma_log = ident["noise_model"]["sigma_log_used"]
    df = load_nesca()
    on_flow = df["Location"].astype(str).str.contains("on flow", case=False, na=False)
    return df, on_flow.values, sigma_log, n_buoy


def class_arrays(df, keep, centre, shape):
    cx, cy = centre
    r_all = np.hypot(df["x_m"].values - cx, df["y_m"].values - cy)
    radii, obs, w_s, names = [], [], [], []
    for name, d_lo, d_hi, _m, _p in FRACTIONS:
        v = df[f"glass_g_per_m2[{name}]"].values / 1000.0
        good = keep & np.isfinite(v) & (v > 0)
        radii.append(r_all[good])
        obs.append(v[good])
        w_s.append(
            float(settling.settling_velocity(settling.sieve_midpoint(d_lo, d_hi), shape=shape))
        )
        names.append(name)
    return radii, obs, w_s, names


def fit_centre(df, keep, shape, only_class=None):
    """Vent position and steady fit, minimising the log-space residual."""

    def misfit(centre, full=False):
        radii, obs, w_s, _n = class_arrays(df, keep, centre, shape)
        if only_class is not None:
            i = [f[0] for f in FRACTIONS].index(only_class)
            radii, obs, w_s = [radii[i]], [obs[i]], [w_s[i]]
        fit = fit_steady_lsq(radii, obs, w_s)
        if full:
            return fit, radii, obs, w_s
        return fit["residual_sd_log"] ** 2 * fit["n_points"]

    best = minimize(
        misfit,
        x0=np.array([0.0, -500.0]),
        method="Nelder-Mead",
        options={"xatol": 5.0, "fatol": 1e-6, "maxiter": 3000},
    )
    fit, radii, obs, w_s = misfit(best.x, full=True)
    return {
        "centre_xy_m": [float(best.x[0]), float(best.x[1])],
        "fit": fit,
        "radii": radii,
        "obs": obs,
        "w_s": w_s,
    }


def main() -> int:
    df, on_flow, sigma_log, n_buoy = load_all()
    keep = ~on_flow
    k_heat = stem.heat_flux_constant()

    # =================== the gate: steady, single class ===================
    gate = {}
    for shape in ("blocky", "long", "sheet"):
        out = fit_centre(df, keep, shape, only_class="250-500um")
        q = out["fit"]["q_umb"]
        w = out["w_s"][0]
        L = float(np.sqrt(q / w))
        f0 = float(stem.f0_point(q, n_buoy, EPSILON_ENTRAIN))
        gate[shape] = {
            "centre_xy_m": out["centre_xy_m"],
            "w_s_m s^-1": w,
            "L_m": L,
            "L_sigma_published_m": BENCH["L_sigma_m"],
            "L_z_score_vs_published": float((L - BENCH["L_m"]) / BENCH["L_sigma_m"]),
            "q_umb_m^3s^-1": float(q),
            "q_z_score_vs_published": float((q - BENCH["q_umb"]) / BENCH["q_sigma"]),
            "f0_m^4s^-3": f0,
            "phi_W": float(stem.heat_flux(f0)),
            "phi_z_score_vs_published": float(
                (stem.heat_flux(f0) - BENCH["phi_W"]) / BENCH["phi_sigma_W"]
            ),
            "residual_sd_log": out["fit"]["residual_sd_log"],
            "n_cores": out["fit"]["n_points"],
        }
    # The gate is on L, which is what the deposit constrains.
    passed = {s: abs(v["L_z_score_vs_published"]) < 2.0 for s, v in gate.items()}

    steady = {
        "_description": (
            "Steady single-class reproduction of the benchmark "
            "(the gate before the unsteady inversion)."
        ),
        "_generated": datetime.date.today().isoformat(),
        "_script": "experiments/nesca/run.py",
        "_git_commit": git_hash(),
        "_data": df.attrs["source"],
        "published_benchmark": BENCH,
        "gate_criterion": (
            "|z| < 2 on the DISPERSAL LENGTH SCALE L, which is what the deposit "
            "constrains.  Q = w_s L^2 and Phi ~ Q^(4/3) inherit the settling "
            "speed, so they are reported per shape hypothesis rather than gated."
        ),
        "excluded": {
            "on_flow_cores": int(on_flow.sum()),
            "why": (
                "Cores logged 'on flow' or 'on flow lobe' sit on the lava and "
                "carry up to 1.2e4 g m^-2, an order of magnitude above the "
                "rest; they are plausibly primary lava fragments, not "
                "fallout.  Retained as a sensitivity below."
            ),
        },
        "by_shape": gate,
        "gate_passed": passed,
        "gate_passed_any": bool(any(passed.values())),
    }

    # sensitivity: keep the on-flow cores
    alt = fit_centre(df, np.ones(len(df), bool), "blocky", only_class="250-500um")
    steady["sensitivity_including_on_flow_cores"] = {
        "L_m": float(np.sqrt(alt["fit"]["q_umb"] / alt["w_s"][0])),
        "centre_xy_m": alt["centre_xy_m"],
        "residual_sd_log": alt["fit"]["residual_sd_log"],
    }
    (ROOT / "results" / "nesca_steady.json").write_text(json.dumps(steady, indent=2))

    print("STEADY GATE (single 250-500 um class, benchmark L = 4900 +/- 400 m):")
    for s, v in gate.items():
        print(
            f"  {s:7s} w={100 * v['w_s_m s^-1']:.2f} cm/s  L={v['L_m']:6.0f} m "
            f"(z={v['L_z_score_vs_published']:+.2f})  Q={v['q_umb_m^3s^-1']:.3g} "
            f"Phi={v['phi_W'] / 1e12:.2f} TW  {'PASS' if passed[s] else 'fail'}"
        )

    if not steady["gate_passed_any"]:
        print(
            "\nGATE NOT PASSED under any shape hypothesis -- stopping before the "
            "unsteady inversion."
        )
        return 1

    # =================== polydisperse nu(Q) inversion =====================
    shape = next(s for s in ("blocky", "long", "sheet") if passed[s])
    out = fit_centre(df, keep, shape)
    centre = out["centre_xy_m"]
    radii, obs, w_s, names = class_arrays(df, keep, centre, shape)
    design = [K.design_matrix(r, Q_NODES, w, 1.0) for r, w in zip(radii, w_s, strict=True)]
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
        num_warmup=1000,
        num_samples=1000,
        num_chains=4,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(20260917), data)
    post = mcmc.get_samples()
    nu = np.asarray(post["nu"])
    p_frac = np.asarray(post["p_frac"])
    sigma = np.asarray(post["sigma"])

    import arviz as az

    idata = az.from_numpyro(mcmc)
    rh = az.rhat(idata)
    max_rhat = float(max(float(np.nanmax(rh[v].values)) for v in rh.data_vars))

    pref = k_heat * 0.187 * (n_buoy**5 / EPSILON_ENTRAIN**2) ** (1 / 3)
    tot = nu.sum(axis=1)
    wgt = nu / tot[:, None]
    q_mean = (wgt * Q_NODES).sum(axis=1)
    lq = np.log(Q_NODES)
    lq_sd = np.sqrt((wgt * (lq - (wgt * lq).sum(axis=1)[:, None]) ** 2).sum(axis=1))

    def ci(v):
        return {
            "median": float(np.median(v)),
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
        }

    unsteady = {
        "_description": "Polydisperse nu(Q) inversion of the NESCA deposit.",
        "_generated": datetime.date.today().isoformat(),
        "_script": "experiments/nesca/run.py",
        "_git_commit": git_hash(),
        "shape_used": shape,
        "vent_xy_m": centre,
        "sigma_log_prior_scale": sigma_log,
        "n_cores_used": int(keep.sum()),
        "classes": names,
        "w_s_m s^-1": w_s,
        "diagnostics": {
            "max_rhat": max_rhat,
            "n_divergences": int(np.sum(np.asarray(mcmc.get_extra_fields().get("diverging", 0))))
            if mcmc.get_extra_fields()
            else 0,
        },
        "posterior": {
            "mass_total_kg": ci(tot),
            "q_mass_weighted_mean_m^3s^-1": ci(q_mean),
            "log_q_sd": ci(lq_sd),
            "sigma_log": ci(sigma),
            "p_fractions": [ci(p_frac[:, i]) for i in range(p_frac.shape[1])],
            "phi_at_mass_weighted_mean_W": ci(pref * q_mean ** (4 / 3)),
            "energy_per_unit_c_J_m_4_3": ci(pref * tot),
            "energy_per_unit_c_J_m_1": ci(
                np.array([pref * np.sum(d * Q_NODES ** (1 / 3)) for d in nu])
            ),
        },
        "nu_median": [float(v) for v in np.median(nu, axis=0)],
        "nu_ci95": [
            [float(v) for v in np.percentile(nu, 2.5, axis=0)],
            [float(v) for v in np.percentile(nu, 97.5, axis=0)],
        ],
        "q_nodes": [float(v) for v in Q_NODES],
        "caveat": (
            "The identifiability analysis found the deposit resolves 2-6 modes "
            "of nu(Q).  The shape of "
            "nu between those modes is prior, not evidence.  Quote the integrated "
            "quantities; show nu with its interval."
        ),
    }
    # ---- posterior predictive check: is ONE nu consistent with all classes? ----
    # Second half of the polydispersity test.  If a single nu cannot fit every sieve
    # fraction, either p_i varies during the eruption (each class then has its
    # own nu_i) or the settling model is wrong.  Either is a result.
    ppc = {}
    nu_med = np.median(nu, axis=0)
    p_med = np.median(p_frac, axis=0)
    for i, name in enumerate(names):
        pred = p_med[i] * (design[i] @ nu_med)
        resid = np.log(obs[i]) - np.log(pred)
        z = resid / float(np.median(sigma))
        # Is there a systematic trend with radius?  A shared nu that is wrong for
        # this class shows up as a slope, not as scatter.
        rr = radii[i]
        slope, intercept = np.polyfit(rr / 1e3, z, 1)
        n = z.size
        se = float(
            np.sqrt(np.sum((z - (slope * rr / 1e3 + intercept)) ** 2) / (n - 2))
            / np.sqrt(np.sum((rr / 1e3 - np.mean(rr / 1e3)) ** 2))
        )
        ppc[name] = {
            "n": int(n),
            "mean_standardised_residual": float(np.mean(z)),
            "sd_standardised_residual": float(np.std(z, ddof=1)),
            "radial_slope_per_km": float(slope),
            "radial_slope_se": se,
            "radial_slope_z": float(slope / se) if se > 0 else float("nan"),
            "systematic_radial_trend": bool(abs(slope / se) > 3.0) if se > 0 else None,
        }
    trends = [v["systematic_radial_trend"] for v in ppc.values()]
    unsteady["posterior_predictive_by_class"] = ppc
    unsteady["H2_single_nu_verdict"] = {
        "classes_with_systematic_radial_trend": int(sum(bool(t) for t in trends)),
        "statement": (
            "A single nu(Q) shared by every sieve fraction is the model's central "
            "assumption (Theorem 1 requires the mass fractions p_i to be constant "
            "in time).  A class for which that assumption fails shows a "
            "SYSTEMATIC RADIAL TREND in its standardised residuals, not merely "
            "extra scatter.  The per-class slopes and their significance are "
            "above; "
            + (
                "no class shows one at |z| > 3, so one nu is adequate for all four fractions."
                if not any(trends)
                else "at least one class does, which means either p_i varied during the "
                "eruption -- each class then needs its own nu_i -- or the settling "
                "law is wrong for that fraction.  This is a testable finding, not "
                "a fit failure to be tuned away."
            )
        ),
    }

    (ROOT / "results" / "nesca_unsteady.json").write_text(json.dumps(unsteady, indent=2))
    pp = unsteady["posterior"]
    print(f"\nPOLYDISPERSE nu(Q) INVERSION (shape={shape}, rhat={max_rhat:.3f}):")
    print(
        f"  total mass  {pp['mass_total_kg']['median']:.3g} kg "
        f"{[float(f'{v:.3g}') for v in pp['mass_total_kg']['ci95']]}"
    )
    print(f"  <Q>         {pp['q_mass_weighted_mean_m^3s^-1']['median']:.3g} m^3/s")
    phi = pp["phi_at_mass_weighted_mean_W"]
    print(
        f"  Phi(<Q>)    {phi['median'] / 1e12:.2f} TW {[round(v / 1e12, 2) for v in phi['ci95']]}"
    )
    print(f"  log-Q sd    {pp['log_q_sd']['median']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
