"""Acceptance checks for the forward model.

Every number the paper may quote about the forward model is produced here
and written to ``results/forward_checks.json``.  Nothing in this script fits
anything; it evaluates the model at known parameters and records the outcome.
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

from plume_inv import forward as fwd  # noqa: E402
from plume_inv import kernel as K  # noqa: E402
from plume_inv import settling, stem, umbrella  # noqa: E402
from plume_inv.constants import (  # noqa: E402
    C_F_PLANAR,
    C_Q_POINT,
    EPSILON_ENTRAIN,
    LAMBDA_FRONT,
)

OUT = ROOT / "results" / "forward_checks.json"

Q0 = 7.6e5
W = 0.03
MASS = 1.0e9
TAU = 15 * 3600.0


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


def stratification_n() -> tuple[float, str]:
    f = ROOT / "results" / "stratification.json"
    if f.exists():
        d = json.loads(f.read_text())
        return d["N_primary"]["value_s^-1"], "results/stratification.json#N_primary.value_s^-1"
    return 1.0e-3, "fallback: Pegler & Ferguson (2021)"


def main() -> int:
    n_woa, n_src = stratification_n()
    n_pf = 1.0e-3
    res: dict = {
        "_description": "Forward-model acceptance checks.",
        "_generated": datetime.date.today().isoformat(),
        "_script": "experiments/forward_checks/run.py",
        "_git_commit": git_hash(),
        "_stratification_source": n_src,
    }

    # ---- 1. MTT coefficients ------------------------------------------------
    point = stem.solve_stem_mtt(6e2, n_pf, EPSILON_ENTRAIN)
    planar = stem.solve_stem_mtt_planar(10.0, n_pf, EPSILON_ENTRAIN)
    res["mtt_closure_verification"] = {
        "point_c_q_at_z_max": point.c_q_max,
        "point_c_q_at_z_neutral": point.c_q_neutral,
        "point_c_q_published": C_Q_POINT,
        "point_relative_error": abs(point.c_q_max - C_Q_POINT) / C_Q_POINT,
        "planar_c_q_at_z_max": planar.c_q_max,
        "planar_c_q_implied_by_published": C_F_PLANAR ** (-2.0 / 3.0),
        "planar_relative_error": abs(planar.c_q_max - C_F_PLANAR ** (-2.0 / 3.0))
        / C_F_PLANAR ** (-2.0 / 3.0),
        "z_max_over_scale_point": point.z_max / (6e2 / (EPSILON_ENTRAIN**2 * n_pf**3)) ** 0.25,
        "z_neutral_over_scale_point": point.z_neutral
        / (6e2 / (EPSILON_ENTRAIN**2 * n_pf**3)) ** 0.25,
        "z_max_over_scale_planar": planar.z_max / (10.0 / n_pf**3) ** (1.0 / 3.0),
        "finding": (
            "Direct integration of the MTT equations reproduces both published "
            "closure coefficients to better than 0.2 %, and identifies them as "
            "the volume flux at the TOP OF RISE, not at the neutral level "
            "(where the point-source coefficient is 2.489, not 3.520)."
        ),
    }

    # ---- 2. Benchmark reproduction -----------------------------------------
    f0_bench = float(stem.f0_point(Q0, n_pf, EPSILON_ENTRAIN))
    res["benchmark_pf2021"] = {
        "inputs": {
            "q_umb_m^3s^-1": Q0,
            "N_s^-1": n_pf,
            "epsilon": EPSILON_ENTRAIN,
            "w_s_m s^-1": W,
        },
        "L_m": float(stem.gaussian_length_scale(Q0, W)),
        "L_published_m": 4.9e3,
        "L_published_sigma_m": 0.4e3,
        "L_published_provenance": (
            "Pegler & Ferguson (2021), Results, 'Dispersal of the Northern Escanaba "
            "tephra', and the Fig. 2 caption: L = 4.9 +/- 0.4 km, sigma from "
            "bootstrapping 1e4 resampled datasets.  A value of 5.0 km is the "
            "back-of-envelope sqrt(7.6e5/0.03), not a published value."
        ),
        "L_is_not_an_independent_check": (
            "L = sqrt(Q/w) is the algebraic inverse of the relation the benchmark "
            "used to turn its fitted L into Q_umb, so recovering 5033 m from "
            "Q = 7.6e5 and w = 0.03 tests arithmetic, not science.  The real "
            "reproduction test is experiments/nesca: fit L to the published core data and "
            "recover 4.9 +/- 0.4 km.  What IS an independent check here is the MTT "
            "integration above, which derives the closure coefficients rather than "
            "assuming them."
        ),
        "F0_m^4s^-3": f0_bench,
        "F0_published_m^4s^-3": 6.0e2,
        "heat_flux_W": float(stem.heat_flux(f0_bench)),
        "heat_flux_published_W": 1.5e12,
        "heat_flux_constant_W s^3 m^-4": stem.heat_flux_constant(),
        "mass_fraction_within_L": 1.0 - float(np.exp(-np.pi)),
        "mass_fraction_within_L_as_quoted_by_benchmark": 0.93,
        "mass_fraction_note": (
            "The exact value for Omega = Omega_0 exp[-pi (r/L)^2] is "
            "1 - exp(-pi) = 0.9568.  The benchmark's ~93 % corresponds to r/L = 0.920; "
            "re-check the benchmark's definition of L before quoting either."
        ),
        "with_woa_stratification": {
            "N_s^-1": n_woa,
            "F0_m^4s^-3": float(stem.f0_point(Q0, n_woa, EPSILON_ENTRAIN)),
            "heat_flux_W": float(stem.heat_flux(stem.f0_point(Q0, n_woa, EPSILON_ENTRAIN))),
        },
    }

    # ---- 3. Kernel limits ---------------------------------------------------
    src = fwd.constant_history(Q0, MASS, TAU, n_t=4001)
    L = float(stem.gaussian_length_scale(Q0, W))
    r = np.linspace(0.0, 3 * L, 401)
    num = K.deposit_quasi_steady(r, src.t, src.q, src.mdot, W, 1.0)
    ana = MASS * (W / Q0) * np.exp(-np.pi * (r / L) ** 2)
    s_grid = np.concatenate(
        ([0.0], np.geomspace(1e-6 * Q0 / (np.pi * W), 40 * Q0 / (np.pi * W), 3000))
    )

    def deposited(t, q, mdot, w):
        d = K.deposit_quasi_steady(np.sqrt(s_grid), t, q, mdot, w, 1.0)
        return float(np.pi * np.trapezoid(d, s_grid))

    waning = fwd.waning_history(2e6, 0.3 * TAU, MASS, TAU, n_t=4001)
    pulses = fwd.two_pulse_history(3e5, 2e6, 0.3 * TAU, 0.5 * TAU, MASS, TAU, n_t=4001)
    res["kernel_limits"] = {
        "gaussian_limit_max_rel_error": float(np.max(np.abs(num - ana)) / ana.max()),
        "mass_conservation_rel_error": {
            "constant": abs(deposited(src.t, src.q, src.mdot, W) - MASS) / MASS,
            "waning": abs(deposited(waning.t, waning.q, waning.mdot, W) - MASS) / MASS,
            "two_pulse": abs(deposited(pulses.t, pulses.q, pulses.mdot, W) - MASS) / MASS,
        },
    }

    # ---- 4. Theorem 1 -------------------------------------------------------
    t = np.linspace(0.0, TAU, 8001)
    q_ramp = Q0 * (0.5 + 1.0 * t / TAU)
    mdot = np.full_like(t, MASS / TAU)
    rr = np.linspace(0.0, 3 * L, 201)
    d_fwd = K.deposit_quasi_steady(rr, t, q_ramp, mdot, W, 1.0)
    d_rev = K.deposit_quasi_steady(rr, t, q_ramp[::-1].copy(), mdot, W, 1.0)
    per_class = {}
    for w in (0.005, 0.012, 0.03, 0.055, 0.08):
        a = K.deposit_quasi_steady(rr, t, q_ramp, mdot, w, 1.0)
        b = K.deposit_quasi_steady(rr, t, q_ramp[::-1].copy(), mdot[::-1].copy(), w, 1.0)
        per_class[f"w={w}"] = float(np.max(np.abs(a - b)) / a.max())
    # A genuinely nonlinear relabeling t -> tau (t/tau)^2 that preserves nu:
    # reparametrise Q, and transform Mdot_p so the released mass in each
    # relabelled interval is unchanged.
    cum = np.concatenate(([0.0], np.cumsum(np.diff(t) * 0.5 * (mdot[1:] + mdot[:-1]))))
    t_new = TAU * (t / TAU) ** 2
    q_relabel = np.interp(t, t_new, q_ramp)
    mdot_relabel = np.gradient(np.interp(t, t_new, cum), t)
    d_relabel = K.deposit_quasi_steady(rr, t, q_relabel, np.maximum(mdot_relabel, 0.0), W, 1.0)
    # A cyclic permutation of three constant blocks: same nu, not a reflection.
    levels = np.array([1.0, 2.5, 0.6]) * Q0
    widths = np.array([0.2, 0.3, 0.5])

    def _blocks(order):
        edges = np.concatenate(([0.0], np.cumsum(widths[list(order)]))) * TAU
        out = np.empty_like(t)
        for k, idx in enumerate(order):
            out[(t >= edges[k]) & (t < edges[k + 1])] = levels[idx]
        out[t >= edges[-1]] = levels[order[-1]]
        return out

    d_abc = K.deposit_quasi_steady(rr, t, _blocks((0, 1, 2)), mdot, W, 1.0)
    d_bca = K.deposit_quasi_steady(rr, t, _blocks((1, 2, 0)), mdot, W, 1.0)

    res["theorem1"] = {
        "time_reversal_max_rel_difference": float(np.max(np.abs(d_fwd - d_rev)) / d_fwd.max()),
        "nonlinear_relabel_max_rel_difference": float(
            np.max(np.abs(d_fwd - d_relabel)) / d_fwd.max()
        ),
        "nonlinear_relabel_mass_check": {
            "original_kg": float(np.trapezoid(mdot, t)),
            "relabelled_kg": float(np.trapezoid(np.maximum(mdot_relabel, 0.0), t)),
        },
        "block_permutation_max_rel_difference": float(np.max(np.abs(d_abc - d_bca)) / d_abc.max()),
        "per_size_class_max_rel_difference": per_class,
        "non_degeneracy_check_rel_difference": float(
            np.max(
                np.abs(
                    K.deposit_quasi_steady(rr, t, np.full_like(t, Q0), mdot, W, 1.0)
                    - K.deposit_quasi_steady(
                        rr, t, Q0 * np.where(t < TAU / 2, 0.4, 1.6), mdot, W, 1.0
                    )
                )
            )
            / d_fwd.max()
        ),
        "finding": (
            "Deposits from nu-preserving relabelings agree to machine "
            "precision in every size class, while a history with a "
            "different nu differs by about 50 % of the peak deposit.  "
            "The invariance is real and not vacuous."
        ),
    }

    # ---- 5. Unsteady umbrella ----------------------------------------------
    for n_name, n_val in (("pf2021", n_pf), ("woa", n_woa)):
        tg = np.linspace(0.0, 20 * 3600.0, 4001)
        umb_c = umbrella.solve_umbrella(lambda _t: Q0, tg, n_val, LAMBDA_FRONT)
        ana_front = umbrella.front_radius_constant_q(tg[1:], Q0, n_val, LAMBDA_FRONT)
        res.setdefault("unsteady_umbrella", {})[n_name] = {
            "N_s^-1": n_val,
            "front_law_max_rel_error": float(
                np.max(np.abs(umb_c.front_radius[1:] - ana_front) / ana_front)
            ),
            "front_radius_m": {
                f"{h}h": float(
                    umbrella.front_radius_constant_q(h * 3600.0, Q0, n_val, LAMBDA_FRONT)
                )
                for h in (1, 5, 10, 20)
            },
            "thickness_m": {
                f"{h}h": float(umbrella.thickness_constant_q(h * 3600.0, Q0, n_val, LAMBDA_FRONT))
                for h in (1, 5, 10, 20)
            },
        }

    t_end = K.settling_window(TAU, Q0, n_pf, W, LAMBDA_FRONT)
    te, qe, me = K.extend_history(src.t, src.q, src.mdot, t_end, n_extra=3000)
    umb = umbrella.solve_umbrella(lambda tt: float(np.interp(tt, te, qe)), te, n_pf, LAMBDA_FRONT)
    r_core = np.array([1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3])
    d_uns = K.deposit_unsteady(r_core, umb, te, qe, me, W, 1.0, n_quad=8192)
    d_qs = K.deposit_quasi_steady(r_core, src.t, src.q, src.mdot, W, 1.0)
    r_fine = np.linspace(1.0, umb.max_front_radius * 0.9999, 4000)
    dep_fine = K.deposit_unsteady(r_fine, umb, te, qe, me, W, 1.0, n_quad=4096)

    long_src = fwd.constant_history(Q0, MASS, 5000 * 3600.0, n_t=6001)
    umb_long = umbrella.solve_umbrella(lambda _t: Q0, long_src.t, n_pf, LAMBDA_FRONT)
    r_l = np.array([1e3, 2e3, 3e3, 4e3])
    conv = np.max(
        np.abs(
            K.deposit_unsteady(
                r_l, umb_long, long_src.t, long_src.q, long_src.mdot, W, 1.0, n_quad=8192
            )
            - K.deposit_quasi_steady(r_l, long_src.t, long_src.q, long_src.mdot, W, 1.0)
        )
        / K.deposit_quasi_steady(r_l, long_src.t, long_src.q, long_src.mdot, W, 1.0)
    )

    res["unsteady_kernel"] = {
        "settling_window_over_eruption_duration": t_end / TAU,
        "settling_residual_tolerance": 1e-5,
        "mass_conservation_rel_error": abs(
            float(np.trapezoid(2 * np.pi * r_fine * dep_fine, r_fine)) - MASS
        )
        / MASS,
        "front_radius_at_end_of_record_m": umb.max_front_radius,
        "front_radius_note": (
            "NOT a maximum: with the source off the box model gives R_f ~ t^(1/3) "
            "without bound, so this is the radius at whichever time the record "
            "was stopped.  The deposit extent does not measure umbrella volume; "
            "the weaker bound V >= pi R^2 h_min does hold."
        ),
        "umbrella_volume_m^3": umb.total_volume,
        "ratio_unsteady_to_quasi_steady": {
            f"{int(rc / 1000)}km": float(u / s)
            for rc, u, s in zip(r_core, d_uns, d_qs, strict=True)
        },
        "convergence_to_gaussian_long_record_max_rel_error": float(conv),
        "finding": (
            "At NESCA timescales the front-limited kernel is steeper than the "
            "Gaussian: +13 % at 1 km, -84 % at 8 km.  A Gaussian fitted to such "
            "a deposit should return a shorter L, hence a smaller Q_umb = w L^2 "
            "and a smaller Phi ~ Q^(4/3).  Whether the INVERSION is biased low, "
            "and by how much, is measured in experiments/synthetic by fitting the steady model "
            "to synthetic front-limited deposits on the real core geometry: the "
            "near-field enhancement and the far-field suppression push L in "
            "opposite directions, so the sign does not follow from the shape "
            "alone and is not asserted here."
        ),
    }

    # ---- 6. Settling law ----------------------------------------------------
    fractions = [
        ("63-125um", 63e-6, 125e-6),
        ("125-250um", 125e-6, 250e-6),
        ("250-500um", 250e-6, 500e-6),
        ("500-1000um", 500e-6, 1e-3),
        ("1000-2000um", 1e-3, 2e-3),
    ]
    res["settling_law"] = {
        "law": "Ferguson & Church (2004), J. Sediment. Res. 74, 933-937, Eq. (4)",
        "coefficients": (
            "Barreyre, Soule & Sohn (2011), J. Volcanol. Geotherm. Res. 205, "
            "84-93, Table 2: non-linear maximum-likelihood fit of the same "
            "equation to tank measurements of real deep-sea basaltic "
            "volcaniclasts from the Gakkel Ridge, by clast shape.  "
            "doi:10.1016/j.jvolgeores.2011.05.006."
        ),
        "shape_coefficients_C1_C2": {k: list(v) for k, v in settling.SHAPE_CLASSES.items()},
        "rho_tephra_kg m^-3": 2600.0,
        "nu_seawater_m^2s^-1": 1.6e-6,
        "w_s_by_fraction_m s^-1": {
            name: float(settling.settling_velocity(settling.sieve_midpoint(a, b)))
            for name, a, b in fractions
        },
        "w_s_by_shape_and_fraction_m s^-1": {
            shape: {
                name: float(settling.settling_velocity(settling.sieve_midpoint(a, b), shape=shape))
                for name, a, b in fractions
            }
            for shape in settling.SHAPE_CLASSES
        },
        "benchmark_fraction_250_500um": {
            "computed_blocky_m s^-1": float(
                settling.settling_velocity(settling.sieve_midpoint(250e-6, 500e-6), shape="blocky")
            ),
            "computed_sheet_m s^-1": float(
                settling.settling_velocity(settling.sieve_midpoint(250e-6, 500e-6), shape="sheet")
            ),
            "published_m s^-1": 0.03,
            "published_sigma_m s^-1": 0.01,
        },
        "shape_is_first_order": {
            "w_sheet_over_assumed": float(
                settling.settling_velocity(settling.sieve_midpoint(250e-6, 500e-6), shape="sheet")
                / 0.03
            ),
            "q_factor": float(
                settling.settling_velocity(settling.sieve_midpoint(250e-6, 500e-6), shape="sheet")
                / 0.03
            ),
            "phi_factor": float(
                (
                    settling.settling_velocity(
                        settling.sieve_midpoint(250e-6, 500e-6), shape="sheet"
                    )
                    / 0.03
                )
                ** (4.0 / 3.0)
            ),
            "statement": (
                "The blocky coefficients reproduce the benchmark's assumed "
                "3 cm/s at 250-500 um almost exactly, which independently "
                "corroborates it from measurements the benchmark did not use.  "
                "But NESCA's tephra is limu o Pele -- bubble-wall fragments, i.e. "
                "SHEET clasts; the coordinate workbook is named 'LimuSamples'.  "
                "For sheet clasts w is 0.46x the assumed value, so Q = w L^2 "
                "falls by 0.46 and Phi ~ Q^(4/3) by 0.36: a factor 2.8 in the "
                "headline heat flux, which the benchmark did not propagate."
            ),
        },
        "shape_is_identifiable_from_the_deposit": {
            "observable": "L(fine)/L(coarse) = sqrt(w_coarse/w_fine)",
            "why": (
                "Independent of Q, of total erupted mass and of the eruption "
                "history, because L_i^2 = Q/w_i.  It depends only on the "
                "settling law, so a polydisperse deposit constrains the clast "
                "shape -- and a single-fraction inversion cannot."
            ),
            "ratio_63_125_to_500_1000_by_shape": {
                shape: settling.length_scale_ratio(
                    settling.sieve_midpoint(500e-6, 1e-3),
                    settling.sieve_midpoint(63e-6, 125e-6),
                    shape=shape,
                )
                for shape in settling.SHAPE_CLASSES
            },
        },
        "sensitivity_rho_tephra": {
            f"{rho:.0f}": float(
                settling.settling_velocity(settling.sieve_midpoint(250e-6, 500e-6), rho_s=rho)
            )
            for rho in (2400.0, 2600.0, 2900.0)
        },
        "sensitivity_viscosity": {
            "deep_2C_nu_1.6e-6": float(
                settling.settling_velocity(settling.sieve_midpoint(250e-6, 500e-6), nu=1.6e-6)
            ),
            "warm_20C_nu_1.05e-6": float(
                settling.settling_velocity(settling.sieve_midpoint(250e-6, 500e-6), nu=1.05e-6)
            ),
        },
        "open_discrepancy": (
            "Barreyre et al. quote 12, 8 and 4 cm/s at D = 1 mm for blocky, long "
            "and sheet, read off their Fig. 5.  Reproducing those from their own "
            "Table 2 coefficients requires rho_s of 3100-3700 kg/m^3, too dense "
            "for basaltic glass; at a physical 2700 kg/m^3 this code gives 9.3, "
            "6.7 and 3.6 cm/s.  The paper states neither the clast density nor, "
            "unambiguously, the viscosity behind the fit (1.83e-6 m^2/s in one "
            "place, 9e-7 in another, neither the standard 1.05e-6 at 20 degC), so "
            "their curves cannot be reproduced exactly from the paper alone.  "
            "rho_s, C1 and C2 are therefore sampled, not fixed."
        ),
        "laplace_dynamic_range_gain": {
            "_definition": (
                "w_max/w_min over the sieve fractions in play.  This is the factor "
                "by which polydispersity widens the range of Laplace abscissae "
                "pi w r^2 that the data sample; an equivalent widening in radius "
                "would be its square root."
            ),
            "four_nesca_fractions_500_1000um_top": float(
                settling.settling_velocity(settling.sieve_midpoint(500e-6, 1e-3))
                / settling.settling_velocity(settling.sieve_midpoint(63e-6, 125e-6))
            ),
            "four_nesca_fractions_500_2000um_top": float(
                settling.settling_velocity(settling.sieve_midpoint(500e-6, 2e-3))
                / settling.settling_velocity(settling.sieve_midpoint(63e-6, 125e-6))
            ),
            "_note": (
                "The NESCA top fraction is the OPEN bin '>500 um', so its "
                "representative diameter is a modelling choice, not a datum.  Both "
                "truncations are reported; the inversion must state which it uses and "
                "carry the other as a sensitivity."
            ),
        },
        "informative_flux_band_m^3s^-1": {
            "_definition": (
                "Q ~ pi w r^2: the umbrella fluxes the core geometry is sensitive "
                "to, using the actual fraction speeds and core radii 0.5-8 km."
            ),
            "q_min": float(
                np.pi
                * settling.settling_velocity(settling.sieve_midpoint(63e-6, 125e-6))
                * 500.0**2
            ),
            "q_max": float(
                np.pi
                * settling.settling_velocity(settling.sieve_midpoint(500e-6, 1e-3))
                * 8000.0**2
            ),
        },
        "finding": (
            "One physical settling law, with coefficients measured on real "
            "deep-sea volcaniclasts, reproduces the benchmark's 3 +/- 1 cm/s for "
            "the 250-500 um fraction under blocky clasts and supplies the other "
            "fractions consistently.  Across the four NESCA fractions the "
            "settling speed spans a factor of 22 (top bin truncated at 1 mm), "
            "which is the factor by which polydispersity widens the sampled range "
            "of the Laplace transform.  Clast shape moves the 250-500 um speed by "
            "a factor 2.2 across the measured classes and is identifiable from "
            "the deposit itself."
        ),
    }

    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:60])
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
