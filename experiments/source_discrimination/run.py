"""Source discrimination by Bayesian evidence.

Writes ``results/model_comparison.json``.

Each mechanism predicts Phi(t; theta).  That maps to the umbrella flux through
the stem closure, to nu(Q) by pushforward, and to the deposit through the same
kernel and the same likelihood used for the nonparametric inversion -- so the
four mechanisms and the nonparametric reference are compared on exactly the same
data with exactly the same error model.

Two things are reported, deliberately separately:

* a **rate test**, which asks whether a mechanism can deliver the inferred heat
  flux at all within its physically bounded parameters.  It needs no priors and
  is therefore not prior-sensitive.
* the **evidence**, with Bayes factors and a prior-sensitivity sweep.  These
  mechanisms carry parameters spanning many orders of magnitude (crustal
  permeability above all), so the marginal likelihood is partly a statement
  about prior volume.  That is reported, not hidden.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import subprocess
import sys

import numpy as np
from dynesty import NestedSampler
from scipy.special import logsumexp  # noqa: F401

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from plume_inv import kernel as K  # noqa: E402
from plume_inv import settling, sources, stem  # noqa: E402
from plume_inv.constants import EPSILON_ENTRAIN, MEGAPLUME_HEAT_J  # noqa: E402
from plume_inv.data import FRACTIONS, load_nesca  # noqa: E402

OUT = ROOT / "results" / "model_comparison.json"
Q_NODES = np.geomspace(3e4, 3e7, 24)
Q_EDGES = np.concatenate(
    [[Q_NODES[0] * 0.5], np.sqrt(Q_NODES[1:] * Q_NODES[:-1]), [Q_NODES[-1] * 2.0]]
)
N_T = 240
M_EXP = 4.0 / 3.0
NLIVE = 250

# Physically bounded priors.  (lo, hi) in the natural units given; log-uniform
# where the range spans orders of magnitude.
PRIORS = {
    "lava_cooling": {
        "log10_area_m2": (5.0, np.log10(15e6)),  # <= the 15 km^2 mapped flow
        "log10_duration_s": (np.log10(1800.0), np.log10(3 * 24 * 3600.0)),
    },
    "dyke_heating": {
        "log10_length_m": (3.0, np.log10(3e4)),
        "log10_height_m": (2.0, np.log10(5e3)),
    },
    "volatile_exsolution": {
        "log10_volume_m3": (6.0, np.log10(4.5e7)),  # <= the mapped erupted volume
        "log10_co2_wt": (-3.3, -1.7),  # 0.05 - 2 wt %
        "log10_duration_s": (np.log10(1800.0), np.log10(3 * 24 * 3600.0)),
    },
    "hydrothermal_evacuation": {
        "log10_permeability_m2": (-16.0, -11.0),
        "log10_area_m2": (5.0, 8.0),
        "log10_thickness_m": (1.5, 3.2),
        "log10_delta_p_pa": (5.5, 7.5),
        "reservoir_temp_C": (150.0, 400.0),
    },
}
COMMON = {"log10_mass_kg": (5.0, 9.0), "log10_sigma": (-1.0, 0.5)}


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


def load_setup():
    n_buoy = json.loads((ROOT / "results" / "stratification.json").read_text())["N_primary"][
        "value_s^-1"
    ]
    u = json.loads((ROOT / "results" / "nesca_unsteady.json").read_text())
    shape, centre = u["shape_used"], u["vent_xy_m"]
    df = load_nesca()
    keep = ~df["Location"].astype(str).str.contains("on flow", case=False, na=False).values
    r_all = np.hypot(df["x_m"].values - centre[0], df["y_m"].values - centre[1])
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
    design = [K.design_matrix(r, Q_NODES, w, 1.0) for r, w in zip(radii, w_s, strict=True)]
    return df, radii, obs, w_s, names, design, n_buoy, shape, u


def phi_of(model: str, p: dict, t: np.ndarray):
    """Heat-flux history for a mechanism, W."""
    if model == "lava_cooling":
        return sources.phi_lava_cooling(t, 10 ** p["log10_area_m2"], 10 ** p["log10_duration_s"])
    if model == "dyke_heating":
        return sources.phi_dyke(t, 10 ** p["log10_length_m"], 10 ** p["log10_height_m"])
    if model == "volatile_exsolution":
        return sources.phi_volatile(
            t, 10 ** p["log10_volume_m3"], 10 ** p["log10_co2_wt"], 10 ** p["log10_duration_s"]
        )
    if model == "hydrothermal_evacuation":
        phi, _tau, _e = sources.phi_hydrothermal(
            t,
            10 ** p["log10_permeability_m2"],
            10 ** p["log10_area_m2"],
            10 ** p["log10_thickness_m"],
            10 ** p["log10_delta_p_pa"],
            p["reservoir_temp_C"],
        )
        return phi
    raise ValueError(model)


def nu_from_phi(phi: np.ndarray, t: np.ndarray, mass: float, n_buoy: float):
    """Push a heat-flux history through the stem closure onto the Q grid."""
    phi = np.maximum(phi, 1.0)
    f0 = phi / stem.heat_flux_constant()
    q = np.asarray(stem.q_umb_point(f0, n_buoy, EPSILON_ENTRAIN))
    shape_t = q**M_EXP
    integ = np.trapezoid(shape_t, t)
    if not np.isfinite(integ) or integ <= 0:
        return None
    mdot = shape_t * (mass / integ)
    return K.nu_from_history(t, q, mdot, Q_EDGES)


def make_loglike(model, design, obs, n_buoy, t_grid):
    keys = list(PRIORS[model]) + list(COMMON)
    n_p = len(obs)

    def loglike(u):
        p = dict(zip(keys, u[: len(keys)], strict=True))
        # simplex coordinates for the size fractions, via a stick-breaking map
        sticks = u[len(keys) :]
        rem, frac = 1.0, []
        for s in sticks:
            take = rem * s
            frac.append(take)
            rem -= take
        frac.append(rem)
        frac = np.array(frac)
        if np.any(frac <= 1e-9):
            return -1e30
        nu = nu_from_phi(phi_of(model, p, t_grid), t_grid, 10 ** p["log10_mass_kg"], n_buoy)
        if nu is None or not np.isfinite(nu).all() or nu.sum() <= 0:
            return -1e30
        sigma = 10 ** p["log10_sigma"]
        ll = 0.0
        for i in range(n_p):
            pred = frac[i] * (design[i] @ nu)
            if not np.all(pred > 0):
                return -1e30
            resid = np.log(obs[i]) - np.log(pred)
            ll += float(
                -0.5 * np.sum((resid / sigma) ** 2)
                - resid.size * np.log(sigma * np.sqrt(2 * np.pi))
            )
        return ll if np.isfinite(ll) else -1e30

    return loglike, keys


def make_ptform(model, width_scale: float = 1.0):
    bounds = {**PRIORS[model], **COMMON}
    keys = list(bounds)
    lo = np.array([bounds[k][0] for k in keys])
    hi = np.array([bounds[k][1] for k in keys])
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * width_scale

    def ptform(u):
        out = np.empty_like(u)
        out[: len(keys)] = (mid - half) + u[: len(keys)] * (2 * half)
        out[len(keys) :] = u[len(keys) :]  # stick-breaking coords stay in [0,1]
        return out

    return ptform, len(keys)


def run_model(model, design, obs, n_buoy, t_grid, width_scale=1.0, seed=0):
    loglike, keys = make_loglike(model, design, obs, n_buoy, t_grid)
    ptform, n_named = make_ptform(model, width_scale)
    ndim = n_named + (len(obs) - 1)
    # 'rwalk' with bootstrap disabled: the default uniform sampler proposes from
    # bootstrapped ellipsoids, which for this posterior needs a huge enlargement
    # factor and becomes prohibitively inefficient (dynesty warns about exactly
    # this).  Random-walk sampling is the recommended alternative at this
    # dimensionality.
    sampler = NestedSampler(
        loglike,
        ptform,
        ndim,
        nlive=NLIVE,
        sample="rwalk",
        bootstrap=0,
        walks=12,
        rstate=np.random.default_rng(seed),
    )
    sampler.run_nested(print_progress=False, dlogz=1.0, maxiter=40000)
    r = sampler.results
    return float(r.logz[-1]), float(r.logzerr[-1]), r, keys


def main() -> int:
    df, radii, obs, w_s, names, design, n_buoy, shape, u = load_setup()
    t_grid = np.linspace(0.0, 3 * 24 * 3600.0, N_T)
    phi_obs = u["posterior"]["phi_at_mass_weighted_mean_W"]

    res = {
        "_description": "Source discrimination for the NESCA megaplume.",
        "_generated": datetime.date.today().isoformat(),
        "_script": "experiments/source_discrimination/run.py",
        "_git_commit": git_hash(),
        "clast_shape_used": shape,
        "inferred_phi_W": phi_obs,
        "material_properties_provenance": sources.NEEDS_PRIMARY_SOURCE,
        "priors": {m: {k: list(v) for k, v in PRIORS[m].items()} for m in PRIORS},
        "common_priors": {k: list(v) for k, v in COMMON.items()},
    }

    # ---- 1. the delivery test, which needs no priors ---------------------
    # Two axes, because they exclude different things:
    #   * peak heat flux -- can the mechanism reach the inferred Phi at all?
    #   * ENERGY DELIVERED over the eruption -- can it supply a megaplume?
    # The peak is reported but is NOT the discriminator: both conduction models
    # have a 1/sqrt(t) singularity at t = 0, so their "peak" is whatever time you
    # choose to evaluate first, which is an artefact.  The integral over the
    # eruption window has no such sensitivity and is what the verdict rests on.
    # The deposit does not constrain the eruption duration, so no single value
    # is asserted here.  The verdict is reported as a function of duration over
    # DURATION_SWEEP_S below.  TAU_REF is a reference point for the tables and
    # is not an inferred quantity.
    TAU_REF = 15 * 3600.0
    DURATION_SWEEP_S = np.geomspace(3600.0, 100 * 3600.0, 40)
    tau_erupt = TAU_REF
    t_int = np.linspace(1.0, tau_erupt, 20000)
    MEGAPLUME_J = MEGAPLUME_HEAT_J  # Baker et al. (1989); see constants.py

    def deliver(phi_series):
        return float(np.trapezoid(phi_series, t_int)), float(np.max(phi_series))

    rate = {}
    e, pk = deliver(sources.phi_lava_cooling(t_int, 15e6, 1800.0))
    rate["lava_cooling"] = {
        "energy_delivered_J": e,
        "peak_phi_W": pk,
        "reservoir_available_J": sources.lava_heat_budget(4.5e7),
        "bound": "area <= 15 km^2 mapped, emplaced as fast as the prior allows "
        "(1800 s); volume <= 4.5e7 m^3",
        "limited_by": "RATE -- the heat is there (2.1e17 J) but conduction "
        "through a 15 km^2 surface cannot deliver it in hours",
    }
    e, pk = deliver(sources.phi_dyke(t_int, 3e4, 5e3))
    rate["dyke_heating"] = {
        "energy_delivered_J": e,
        "peak_phi_W": pk,
        "reservoir_available_J": None,
        "bound": "length <= 30 km, height <= 5 km -- a very large dyke",
        "limited_by": "neither, at the top of its range",
    }
    e_vol = sources.volatile_heat_budget(4.5e7, 0.02)
    rate["volatile_exsolution"] = {
        "energy_delivered_J": e_vol,
        "peak_phi_W": float(e_vol / 1800.0),
        "reservoir_available_J": e_vol,
        "bound": "volume <= 4.5e7 m^3, CO2 <= 2 wt % (generous for MORB), released over >= 1800 s",
        "limited_by": "BUDGET -- it can reach the flux if released fast enough, "
        "but the total heat available is an order of magnitude "
        "below the megaplume range",
    }
    phi_h, tau_h, e_h = sources.phi_hydrothermal(t_int, 1e-12, 1e8, 1000.0, 3e7, 400.0)
    e, pk = deliver(phi_h)
    rate["hydrothermal_evacuation"] = {
        "energy_delivered_J": e,
        "peak_phi_W": pk,
        "reservoir_available_J": e_h,
        "decay_timescale_h": tau_h / 3600.0,
        "bound": "k <= 1e-11 m^2, area <= 1e8 m^2, thickness <= 1.6 km, dP <= 30 MPa",
        "limited_by": "neither",
    }
    target = phi_obs["median"]
    for v in rate.values():
        v["can_reach_inferred_phi"] = bool(v["peak_phi_W"] >= target)
        v["peak_over_inferred"] = float(v["peak_phi_W"] / target)
        v["supplies_a_megaplume"] = bool(v["energy_delivered_J"] >= MEGAPLUME_J[0])
        v["energy_over_megaplume_minimum"] = float(v["energy_delivered_J"] / MEGAPLUME_J[0])

    res["delivery_test"] = rate
    res["delivery_test"]["_inferred_phi_W"] = target
    res["delivery_test"]["_megaplume_heat_range_J"] = list(MEGAPLUME_J)
    res["delivery_test"]["_megaplume_heat_source"] = (
        "Baker et al. (1989) abstract, 10^16-10^17 J; constants.MEGAPLUME_HEAT_J"
    )
    res["delivery_test"]["_eruption_window_s"] = tau_erupt
    res["delivery_test"]["_duration_reference_s"] = TAU_REF

    # ---- 1b. duration sensitivity ----------------------------------------
    # Conduction delivers energy as ~sqrt(t), so the exclusion of lava cooling
    # must be shown to hold across the plausible range of eruption durations
    # and not at one chosen duration.
    def energy_to(tau, phi_of_t):
        tt = np.linspace(1.0, float(tau), 20000)
        return float(np.trapezoid(phi_of_t(tt), tt))

    mech = {
        "lava_cooling": lambda tt: sources.phi_lava_cooling(tt, 15e6, 1800.0),
        "dyke_heating": lambda tt: sources.phi_dyke(tt, 3e4, 5e3),
        "hydrothermal_evacuation": lambda tt: sources.phi_hydrothermal(
            tt, 1e-12, 1e8, 1000.0, 3e7, 400.0
        )[0],
    }
    sweep = {"duration_h": (DURATION_SWEEP_S / 3600.0).tolist()}
    for name, fn in mech.items():
        e_tau = np.array([energy_to(tau, fn) for tau in DURATION_SWEEP_S])
        sweep[name] = {
            "energy_delivered_J": e_tau.tolist(),
            "energy_over_megaplume_minimum": (e_tau / MEGAPLUME_J[0]).tolist(),
            "supplies_a_megaplume_anywhere": bool(np.any(e_tau >= MEGAPLUME_J[0])),
        }
    # Volatile exsolution is budget limited, so its delivered energy does not
    # grow with duration.
    sweep["volatile_exsolution"] = {
        "energy_delivered_J": [e_vol] * len(DURATION_SWEEP_S),
        "energy_over_megaplume_minimum": [e_vol / MEGAPLUME_J[0]] * len(DURATION_SWEEP_S),
        "supplies_a_megaplume_anywhere": bool(e_vol >= MEGAPLUME_J[0]),
    }

    # Duration at which conductive cooling of the lava would first reach the
    # megaplume floor, from a power law fitted to the sweep.  Reported so the
    # exclusion can be checked against any duration a reader prefers.
    e_lava = np.array(sweep["lava_cooling"]["energy_delivered_J"])
    slope, icept = np.polyfit(np.log(DURATION_SWEEP_S), np.log(e_lava), 1)
    tau_cross = float(np.exp((np.log(MEGAPLUME_J[0]) - icept) / slope))
    sweep["_lava_power_law_exponent"] = float(slope)
    sweep["_lava_crossing_duration_s"] = tau_cross
    sweep["_lava_crossing_duration_h"] = tau_cross / 3600.0
    sweep["_note"] = (
        "Energy delivered as a function of eruption duration, 1 h to 100 h.  "
        "The deposit does not constrain duration; this sweep replaces the "
        "single 15 h window used earlier so the source verdict can be read at "
        "any duration.  The lava crossing duration is extrapolated from a "
        "power law fitted to the sweep."
    )
    res["duration_sensitivity"] = sweep

    res["delivery_test"]["_note"] = (
        "Each mechanism is pushed to the TOP of its bounded range.  The verdict "
        "rests on ENERGY DELIVERED over a 15 h eruption, not on peak flux: both "
        "conduction models diverge as 1/sqrt(t) at t = 0, so their peak is an "
        "artefact of the first time evaluated.  An earlier version of this test "
        "used peak flux alone and also fixed the volatile release at 15 h when "
        "the prior allows 1800 s; that version wrongly excluded volatile "
        "exsolution on rate.  It is excluded on BUDGET instead."
    )

    print(
        f"DELIVERY TEST -- inferred Phi = {target / 1e12:.2f} TW (shape: {shape}), "
        f"megaplume heat {MEGAPLUME_J[0]:.0e}-{MEGAPLUME_J[1]:.0e} J",
        flush=True,
    )
    for m, v in rate.items():
        if m.startswith("_"):
            continue
        print(
            f"  {m:24s} E(15h) = {v['energy_delivered_J']:8.2e} J "
            f"({v['energy_over_megaplume_minimum']:6.2f}x the {MEGAPLUME_J[0]:.0e} J floor)  "
            f"{'supplies a megaplume' if v['supplies_a_megaplume'] else 'DOES NOT'}",
            flush=True,
        )

    # Does the clast-shape uncertainty change anything?  Phi ~ Q^(4/3) with
    # Q = w_s L^2, so the INFERRED flux scales as w^(4/3) and w spans a factor
    # 2.2 across the measured shape classes.  The energy verdict does not depend
    # on it at all -- that comparison is against the observed megaplume heat
    # range, which is external to this inversion -- but the peak-flux comparison
    # does, so it is repeated at each shape.
    st = json.loads((ROOT / "results" / "nesca_steady.json").read_text())
    w_by_shape = {k: st["by_shape"][k]["w_s_m s^-1"] for k in ("blocky", "long", "sheet")}
    shape_rows = {}
    for sh, w in w_by_shape.items():
        phi_sh = target * (w / w_by_shape[shape]) ** (4.0 / 3.0)
        shape_rows[sh] = {
            "w_s_m s^-1": w,
            "inferred_phi_W": float(phi_sh),
            "mechanisms_reaching_it": sorted(
                m for m, v in rate.items() if not m.startswith("_") and v["peak_phi_W"] >= phi_sh
            ),
        }
    res["clast_shape_robustness"] = {
        "_why": (
            "The settling speed spans a factor 2.2 across the shape classes of "
            "Barreyre et al. (2011), and Phi ~ w^(4/3), so the inferred flux "
            "spans a factor 2.8.  If the verdict flipped between shapes it would "
            "be an artefact of an unresolved assumption."
        ),
        "by_shape": shape_rows,
        "peak_verdict_is_shape_robust": bool(
            len({tuple(r["mechanisms_reaching_it"]) for r in shape_rows.values()}) == 1
        ),
        "energy_verdict_depends_on_shape": False,
        "_energy_note": (
            "The energy comparison is against the observed megaplume heat range "
            "(1e16-1e17 J, Baker et al. 1989), which is independent of this "
            "inversion, so the clast shape cannot affect it."
        ),
    }

    # ---- 2. evidence, with a prior-width sensitivity sweep ---------------
    # The evidence does not depend on the megaplume heat range, which enters the
    # delivery test only.  PLUME_REUSE_EVIDENCE=1 keeps the evidence section of
    # the existing results file and recomputes the delivery test alone; the
    # prior-width sweep for every mechanism and every clast shape is in
    # results/model_comparison_by_shape.json.
    import os

    if os.environ.get("PLUME_REUSE_EVIDENCE") == "1" and OUT.exists():
        old = json.loads(OUT.read_text())
        for key in ("evidence", "bayes_factors_log10_vs_best"):
            if key in old:
                res[key] = old[key]
        res["delivery_test"]["_survivors"] = [
            m for m in sources.MECHANISMS if rate[m]["supplies_a_megaplume"]
        ]
        res["_evidence_reused_from"] = old.get("_git_commit")
        OUT.write_text(json.dumps(res, indent=2))
        print(f"wrote {OUT} (evidence reused)")
        return 0
    ev = {}
    survivors = [m for m in sources.MECHANISMS if rate[m]["supplies_a_megaplume"]]
    res["delivery_test"]["_survivors"] = survivors
    for m in sources.MECHANISMS:
        ev[m] = {}
        # Every mechanism gets a nominal evidence; the prior-width sweep, which
        # is the expensive part, is run only for those the rate test did not
        # already exclude.  A mechanism excluded on rate cannot be rescued by a
        # prior width, so sweeping it would buy nothing.
        sweeps = (
            ((1.0, "nominal"), (0.5, "half_width"), (2.0, "double_width"))
            if m in survivors
            else ((1.0, "nominal"),)
        )
        for ws, label in sweeps:
            try:
                logz, logzerr, _r, keys = run_model(
                    m, design, obs, n_buoy, t_grid, width_scale=ws, seed=7
                )
                ev[m][label] = {
                    "logz": logz,
                    "logz_err": logzerr,
                    "n_params": len(keys) + len(obs) - 1,
                }
                print(
                    f"  evidence {m:24s} {label:13s} logZ = {logz:10.2f} +/- {logzerr:.2f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                ev[m][label] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  evidence {m:24s} {label:13s} FAILED: {exc}")
    res["evidence"] = ev

    best = max(
        (m for m in ev if "logz" in ev[m].get("nominal", {})),
        key=lambda m: ev[m]["nominal"]["logz"],
        default=None,
    )
    if best:
        res["bayes_factors_log10_vs_best"] = {
            "reference": best,
            "nominal": {
                m: (ev[m]["nominal"]["logz"] - ev[best]["nominal"]["logz"]) / np.log(10)
                for m in ev
                if "logz" in ev[m].get("nominal", {})
            },
            "prior_sensitivity_range": {
                m: [
                    min(
                        (ev[m][lab]["logz"] - ev[best][lab]["logz"]) / np.log(10)
                        for lab in ("nominal", "half_width", "double_width")
                        if "logz" in ev[m].get(lab, {}) and "logz" in ev[best].get(lab, {})
                    ),
                    max(
                        (ev[m][lab]["logz"] - ev[best][lab]["logz"]) / np.log(10)
                        for lab in ("nominal", "half_width", "double_width")
                        if "logz" in ev[m].get(lab, {}) and "logz" in ev[best].get(lab, {})
                    ),
                ]
                for m in ev
                if "logz" in ev[m].get("nominal", {})
            },
        }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
