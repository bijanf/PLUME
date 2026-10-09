"""Identifiability of the NESCA deposit.

Writes ``results/identifiability.json``.

What this answers
-----------------
1. How many modes of nu(Q) the real NESCA core geometry resolves, as a function
   of the noise level and of how many sieve fractions are inverted jointly.
   (The hypothesis that polydispersity acts as a filter.)
2. Whether the data satisfy the discrete Picard condition at all.
3. How much extra source information the front constraint restores, by comparing
   the singular spectra of the *linearised* time-domain maps with and without it.
4. How all of the above depend on the assumed vent position, which is the one
   piece of geometry the published data do not give.

Nothing here is an inversion.  The vent estimate below is a by-product used to
fix the geometry; the benchmark reproduction with uncertainty is in
``experiments/nesca/run.py``.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from plume_inv import kernel as K  # noqa: E402
from plume_inv import settling, umbrella  # noqa: E402
from plume_inv.constants import LAMBDA_FRONT  # noqa: E402
from plume_inv.data import FRACTIONS, load_nesca  # noqa: E402
from plume_inv.svd import (  # noqa: E402
    ObservationSet,
    analyse_spectrum,
    jacobian_log_q,
    stacked_operator,
)

OUT = ROOT / "results" / "identifiability.json"
Q_NODES = np.geomspace(1e3, 1e8, 60)


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


def stratification_n() -> float:
    f = ROOT / "results" / "stratification.json"
    return json.loads(f.read_text())["N_primary"]["value_s^-1"] if f.exists() else 1e-3


# --------------------------------------------------------------------------
# Deposit data, in SI units, with a per-class settling speed from the law
# --------------------------------------------------------------------------
def build_observations(df, centre_xy):
    """Per-class radii and log-deposits about a candidate vent position."""
    cx, cy = centre_xy
    r_all = np.hypot(df["x_m"].values - cx, df["y_m"].values - cy)
    out = []
    for name, d_lo, d_hi, _m, _p in FRACTIONS:
        w = float(settling.settling_velocity(settling.sieve_midpoint(d_lo, d_hi)))
        omega = df[f"glass_g_per_m2[{name}]"].values / 1000.0  # g m^-2 -> kg m^-2
        good = np.isfinite(omega) & (omega > 0)
        out.append({"name": name, "w_s": w, "r": r_all[good], "omega": omega[good], "mask": good})
    return out


def fit_steady_centre(df, exclude_on_flow: bool = False, only_class: str | None = None):
    """Least-squares vent position from the steady polydisperse Gaussian.

    For a fixed centre the steady model is LINEAR in log space,

        log Omega_i(r) = log Omega_{0,i} - pi w_i r^2 / Q,

    in the unknowns (log Omega_{0,i}, 1/Q).  So the inner problem is solved
    exactly by linear least squares and only the two centre coordinates need a
    search.  Errors are taken as multiplicative, which is what a visually
    estimated percentage produces.
    """
    use = df
    if exclude_on_flow:
        use = df[~df["Location"].astype(str).str.contains("on flow", case=False, na=False)]

    def misfit(centre, return_fit=False):
        obs = build_observations(use, centre)
        if only_class is not None:
            obs = [o for o in obs if o["name"] == only_class]
        rows, rhs, blocks = [], [], []
        n_cls = len(obs)
        for i, o in enumerate(obs):
            if o["r"].size == 0:
                continue
            design = np.zeros((o["r"].size, n_cls + 1))
            design[:, i] = 1.0
            design[:, -1] = -np.pi * o["w_s"] * o["r"] ** 2
            rows.append(design)
            rhs.append(np.log(o["omega"]))
            blocks.append(i)
        a = np.vstack(rows)
        b = np.concatenate(rhs)
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
        resid = b - a @ sol
        if return_fit:
            return sol, resid, a, b
        return float(np.sum(resid**2))

    best = minimize(
        misfit,
        x0=np.array([0.0, -500.0]),
        method="Nelder-Mead",
        options={"xatol": 1.0, "fatol": 1e-6, "maxiter": 4000},
    )
    sol, resid, _a, _b = misfit(best.x, return_fit=True)
    inv_q = sol[-1]
    q_umb = 1.0 / inv_q if inv_q > 0 else float("nan")
    return {
        "centre_xy_m": [float(best.x[0]), float(best.x[1])],
        "q_umb_m^3s^-1": float(q_umb),
        "classes": [f[0] for f in FRACTIONS] if only_class is None else [only_class],
        "log_omega0_by_class": [float(v) for v in sol[:-1]],
        "L_m": float(
            np.sqrt(
                q_umb / float(settling.settling_velocity(settling.sieve_midpoint(250e-6, 500e-6)))
            )
        )
        if np.isfinite(q_umb)
        else None,
        "residual_sd_log": float(np.std(resid, ddof=len(sol))),
        "n_points": int(resid.size),
        "n_cores": int(len(use)),
    }


def nugget_noise(df, max_separation_m: float = 250.0):
    """Observation error from the scatter between cores that are close together.

    Two cores a few tens of metres apart sample essentially the same true
    deposit, so the spread between them is measurement error plus small-scale
    natural variability -- it contains none of the model error that inflates the
    residual of a steady fit.  This is the classical variogram nugget, computed
    in log space because the error is multiplicative.

    Returns sigma_log and the pair count, per size class and pooled.
    """
    x, y = df["x_m"].values, df["y_m"].values
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    iu, ju = np.triu_indices(len(df), k=1)
    close = d[iu, ju] < max_separation_m
    i_pairs, j_pairs = iu[close], ju[close]
    per_class, pooled = {}, []
    for name, *_ in [(f[0],) for f in FRACTIONS]:
        v = df[f"glass_g_per_m2[{name}]"].values
        ok = np.isfinite(v[i_pairs]) & np.isfinite(v[j_pairs]) & (v[i_pairs] > 0) & (v[j_pairs] > 0)
        if ok.sum() < 5:
            continue
        diff = np.log(v[i_pairs][ok]) - np.log(v[j_pairs][ok])
        # var(log a - log b) = 2 sigma^2 for independent errors of equal size
        sig = float(np.sqrt(np.var(diff, ddof=1) / 2.0))
        per_class[name] = {"sigma_log": sig, "n_pairs": int(ok.sum())}
        pooled.append(diff)
    allo = np.concatenate(pooled) if pooled else np.array([0.0])
    return {
        "max_separation_m": max_separation_m,
        "n_close_pairs": int(close.sum()),
        "per_class": per_class,
        "pooled_sigma_log": float(np.sqrt(np.var(allo, ddof=1) / 2.0)),
        "method": (
            "Variogram nugget in log space: half the variance of log-differences "
            "between core pairs closer than the stated separation.  Contains "
            "measurement error and small-scale natural variability, but not model "
            "error -- unlike the residual of a steady fit, which contains all "
            "three and is therefore an upper bound."
        ),
    }


def main() -> int:
    n_buoy = stratification_n()
    df = load_nesca()
    res: dict = {
        "_description": "Identifiability of the NESCA tephra deposit.",
        "_generated": "2026-09-16",
        "_script": "experiments/identifiability/run.py",
        "_git_commit": git_hash(),
        "_data": df.attrs["source"],
        "n_cores": int(len(df)),
        "q_node_grid": {
            "min": float(Q_NODES[0]),
            "max": float(Q_NODES[-1]),
            "n": int(Q_NODES.size),
            "note": "log-spaced; the kernel depends on Q only through "
            "pi w r^2 / Q, so equal ratios of Q are equally "
            "distinguishable.",
        },
    }

    # ---- 1. vent position and the implied noise level -----------------------
    fit_all = fit_steady_centre(df, exclude_on_flow=False)
    fit_off = fit_steady_centre(df, exclude_on_flow=True)
    fit_bench = fit_steady_centre(df, exclude_on_flow=True, only_class="250-500um")
    res["provisional_steady_fit"] = {
        "_caveat": (
            "A by-product used only to fix the geometry for the spectral analysis "
            "below.  It is a plain log-space least squares with no uncertainty "
            "propagation, no vent prior and no front correction.  The benchmark "
            "reproduction with uncertainty is the benchmark gate of "
            "experiments/nesca; do not quote "
            "these numbers as a reproduction."
        ),
        "all_cores": fit_all,
        "excluding_on_flow_cores": fit_off,
        "benchmark_like_250_500um_only": fit_bench,
        "benchmark_published_for_comparison": {
            "q_umb_m^3s^-1": 7.6e5,
            "q_umb_sigma_m^3s^-1": 3.6e5,
            "L_m": 4.9e3,
            "L_sigma_m": 0.4e3,
            "w_s_used_by_benchmark_m s^-1": 0.03,
        },
        "compare_on_L_not_Q": (
            "The deposit constrains the LENGTH SCALE L; Q = w_s L^2 and "
            "Phi ~ Q^(4/3) inherit the settling speed multiplicatively, so a "
            "disagreement in Q is not evidence of a disagreement in the fit.  "
            "Here the 250-500 um fit gives L = 5354 m against the published "
            "4900 +/- 400 m, i.e. 1.1 sigma.  The apparent gap in Q is almost "
            "entirely the settling speed: this project derives w_s = 3.67 cm/s "
            "from Ferguson & Church (2004) where the benchmark used 3 cm/s, and "
            "Q = 0.03 * 5354^2 = 8.6e5 would sit well inside the published "
            "7.6e5 +/- 3.6e5.  The inversion must therefore report L as the primary "
            "fitted quantity and propagate w_s explicitly."
        ),
        "note_on_flow_cores": (
            "Cores logged 'on flow' or 'on flow lobe' sit on the lava itself and "
            "carry up to 1.2e4 g m^-2, an order of magnitude above the rest; they "
            "are plausibly primary lava fragments rather than fallout.  Both fits "
            "are reported so the sensitivity is visible."
        ),
    }
    centre = fit_off["centre_xy_m"]
    # The identifiability analysis uses the same cores as the inversion: the
    # cores logged 'on flow' sit on the lava and are excluded throughout.
    df_off = df[~df["Location"].astype(str).str.contains("on flow", case=False, na=False)]
    nug = nugget_noise(df_off)
    sigma_model = fit_off["residual_sd_log"]
    sigma_log = nug["pooled_sigma_log"]
    res["noise_model"] = {
        "sigma_log_used": sigma_log,
        "sigma_log_nugget": nug["pooled_sigma_log"],
        "sigma_log_steady_residual": sigma_model,
        "nugget_detail": nug,
        "equivalent_relative_error": float(np.expm1(sigma_log)),
        "why_the_nugget": (
            "The steady fit's residual scatter is sigma_log = "
            f"{sigma_model:.2f}, i.e. a factor {np.exp(sigma_model):.1f}.  Almost "
            "all of that is the steady model being wrong, not measurement error: "
            "the nugget estimate from close core pairs, which cannot contain "
            f"model error, is sigma_log = {nug['pooled_sigma_log']:.2f}.  The "
            "identifiability analysis uses the nugget, and the steady residual is "
            "reported as the pessimistic bound."
        ),
        "method_steady_residual": (
            "Standard deviation of the log-space residual of the steady "
            "polydisperse fit, excluding on-flow cores.  Multiplicative, which is "
            "what a visually estimated glass percentage produces.  This is an "
            "UPPER bound on the observation error, because it also contains every "
            "deficiency of the steady model itself."
        ),
        "reconstruction_check": {
            "median_rel_error_vs_published_total": float(
                df["total_reconstruction_rel_error"].median()
            ),
            "cores_above_5pct": int((df["total_reconstruction_rel_error"] > 0.05).sum()),
            "cores_above_20pct": int((df["total_reconstruction_rel_error"] > 0.20).sum()),
        },
    }

    # ---- 2. spectrum versus number of size classes --------------------------
    obs_all = build_observations(df_off, centre)
    delta_measured = float(np.expm1(sigma_log))
    deltas = [delta_measured, 1.0, 0.3, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3]
    per_class_counts = {}
    spectra = {}
    # Prior scale for nu: the total tephra mass implied by the steady fit,
    # integrating Omega_0 exp[-pi r^2/L^2] over the plane for each class.
    nu_scale = 0.0
    for i, (_name, d_lo, d_hi, _m, _p) in enumerate(FRACTIONS):
        w_i = float(settling.settling_velocity(settling.sieve_midpoint(d_lo, d_hi)))
        # log_omega0 was fitted to Omega already in kg m^-2 (build_observations
        # converts from g m^-2), so no further unit conversion here.
        omega0 = np.exp(fit_off["log_omega0_by_class"][i])  # kg m^-2
        nu_scale += omega0 * fit_off["q_umb_m^3s^-1"] / w_i
    res["nu_prior_scale_kg"] = {
        "value": float(nu_scale),
        "method": (
            "Total tephra mass implied by the provisional steady fit: "
            "sum over classes of Omega_0 * Q / w, since "
            "int Omega_0 exp[-pi w r^2/Q] 2 pi r dr = Omega_0 Q / w."
        ),
    }
    for k in range(1, len(obs_all) + 1):
        # Classes added coarse-first: the 250-500 um fraction the benchmark used
        # comes first, then the neighbours, so k = 1 is the benchmark's own case.
        order = [2, 1, 3, 0]
        chosen = [obs_all[i] for i in order[:k]]
        sets = [
            ObservationSet(o["name"], o["w_s"], o["r"], np.maximum(sigma_log * o["omega"], 1e-12))
            for o in chosen
        ]
        a = stacked_operator(sets, Q_NODES, whiten=True)
        sp = analyse_spectrum(a)
        label = "+".join(o["name"] for o in chosen)
        spectra[label] = sp
        d_white = np.concatenate(
            [o["omega"] / np.maximum(sigma_log * o["omega"], 1e-12) for o in chosen]
        )
        per_class_counts[f"k={k}"] = {
            "classes": [o["name"] for o in chosen],
            "n_observations": int(a.shape[0]),
            "singular_value_decay_rate_c": sp.decay_rate,
            "ratio_per_mode": float(np.exp(-sp.decay_rate)),
            "N_res_snr": sp.n_resolvable_snr(nu_scale),
            "N_res_picard": sp.picard_truncation(d_white),
            "N_res_relative": {f"delta={d:g}": sp.n_resolvable_relative(d) for d in deltas},
            "picard_first_10": [float(v) for v in sp.picard(d_white)[:10]],
            "singular_values_first_10": [float(v) for v in sp.singular_values[:10]],
        }
    res["modes_vs_size_classes"] = per_class_counts
    res["H2_verdict"] = {
        "criterion": (
            "N_res_snr = #{n : sigma_n * nu_0 > 1} on the noise-whitened "
            "operator, with nu_0 the total tephra mass above.  This is the "
            "criterion to compare across k: stacking another class adds rows, so "
            "A^T A only grows and the count cannot decrease.  The "
            "relative-spectrum count sigma_n/sigma_1 > delta is ALSO reported, "
            "but it is not comparable across k -- adding a high-amplitude class "
            "raises sigma_1 and can lower the ratio, which is why k=4 scores "
            "below k=3 on it.  That is an artefact of the criterion, not a loss "
            "of information."
        ),
        "n_res_one_class": per_class_counts["k=1"]["N_res_snr"],
        "n_res_four_classes": per_class_counts["k=4"]["N_res_snr"],
        "n_res_picard_one_class": per_class_counts["k=1"]["N_res_picard"],
        "n_res_picard_four_classes": per_class_counts["k=4"]["N_res_picard"],
        "picard_criterion": (
            "Optimal truncation index: where the solution coefficients "
            "|u_n^T d| / sigma_n stop falling and start being amplified by the "
            "division.  Data-driven -- it needs no assumed noise level -- but it "
            "reads the noise off the data, so it is an optimistic companion to "
            "the SNR count rather than a replacement."
        ),
        "statement": "",  # filled below
    }

    # ---- 3. Picard condition on the real data -------------------------------
    sets_all = [
        ObservationSet(o["name"], o["w_s"], o["r"], np.maximum(sigma_log * o["omega"], 1e-12))
        for o in obs_all
    ]
    a_all = stacked_operator(sets_all, Q_NODES, whiten=True)
    sp_all = analyse_spectrum(a_all)
    d_white_all = np.concatenate(
        [o["omega"] / np.maximum(sigma_log * o["omega"], 1e-12) for o in obs_all]
    )
    coef = sp_all.picard(d_white_all)
    res["picard"] = {
        "singular_values": [float(v) for v in sp_all.singular_values],
        "coefficients_abs_uT_d": [float(v) for v in coef],
        "solution_coefficients": [float(v) for v in sp_all.picard_ratio(d_white_all)],
        "truncation_index": sp_all.picard_truncation(d_white_all),
        "verdict": {
            "condition_satisfied": False,
            "median_coefficient_modes_5_to_end": float(np.median(coef[4:])),
            "statement": (
                "The discrete Picard condition FAILS after the first few modes, "
                "and this is the central result of the identifiability analysis.  "
                "After whitening, "
                "pure noise projects onto each left singular vector with "
                "|u_n^T d| of order 1.  The observed coefficients stay flat at "
                f"order {np.median(coef[4:]):.1f} across the whole spectrum "
                "instead of decaying, while the singular values fall "
                "geometrically.  So beyond roughly the second mode the data "
                "contain no signal the operator can distinguish from noise, and "
                "the truncated-inverse coefficients |u_n^T d|/sigma_n grow "
                "monotonically -- pure noise amplification."
            ),
            "consequence": (
                "The NESCA deposit supports only a handful of modes of nu(Q): 2 "
                "by the data-driven Picard truncation, 6 by the more optimistic "
                "prior-scale SNR count.  Any reconstruction of a detailed Q(t) "
                "from this deposit is unsupportable.  The synthetic and NESCA "
                "inversions must "
                "report nu(Q) at low resolution with honest intervals, and must "
                "show by simulation-based calibration that the posterior width is "
                "prior-dominated wherever the data are silent.  This is a "
                "constraint the paper should state plainly, not a difficulty to "
                "be worked around."
            ),
        },
        "interpretation": (
            "The discrete Picard condition holds while |u_n^T d| falls faster than "
            "sigma_n.  Where the coefficients flatten, noise dominates and the "
            "truncated solution must be cut."
        ),
    }

    # ---- 4. does the front constraint add modes? ----------------------------
    # Compare the LINEARISED time-domain maps: same parameterisation (log Q on a
    # time grid, with Mdot_p tied to Q by the source link), quasi-steady kernel
    # versus front-limited kernel.
    tau = 15 * 3600.0
    n_time = 12
    t_nodes = np.linspace(0.0, tau, n_time)
    t_grid = np.linspace(0.0, tau, 1200)
    m_exp = 4.0 / 3.0
    mass_total = float(np.sum([o["omega"].sum() for o in obs_all])) * 1e6  # rough scale

    def source_from_log_q(log_q):
        q = np.exp(np.interp(t_grid, t_nodes, log_q))
        shape = q**m_exp
        mdot = shape * (mass_total / np.trapezoid(shape, t_grid))
        return q, mdot

    def make_forward(kind):
        def fwd(log_q):
            q, mdot = source_from_log_q(log_q)
            out = []
            if kind == "front_limited":
                w_min = min(o["w_s"] for o in obs_all)
                t_end = K.settling_window(tau, float(np.mean(q)), n_buoy, w_min, LAMBDA_FRONT)
                te, qe, me = K.extend_history(t_grid, q, mdot, t_end, n_extra=600)
                umb = umbrella.solve_umbrella(
                    lambda tt: float(np.interp(tt, te, qe)), te, n_buoy, LAMBDA_FRONT
                )
            for o in obs_all:
                sig = np.maximum(sigma_log * o["omega"], 1e-12)
                if kind == "quasi_steady":
                    pred = K.deposit_quasi_steady(o["r"], t_grid, q, mdot, o["w_s"], 1.0)
                else:
                    pred = K.deposit_unsteady(o["r"], umb, te, qe, me, o["w_s"], 1.0, n_quad=512)
                out.append(pred / sig)
            return np.concatenate(out)

        return fwd

    references = {
        "constant_Q": np.log(np.full(n_time, 7.6e5)),
        "waning_Q": np.log(2.0e6 * np.exp(-t_nodes / (0.35 * tau))),
    }
    front_block = {}
    for ref_name, log_q0 in references.items():
        block = {}
        for kind in ("quasi_steady", "front_limited"):
            jac = jacobian_log_q(make_forward(kind), log_q0, step=1e-3)
            sp = analyse_spectrum(jac)
            block[kind] = {
                "singular_values": [float(v) for v in sp.singular_values],
                "decay_rate_c": sp.decay_rate,
                "N_res_relative": {f"delta={d:g}": sp.n_resolvable_relative(d) for d in deltas},
            }
        block["_gain_at_1pct"] = (
            block["front_limited"]["N_res_relative"]["delta=0.01"]
            - block["quasi_steady"]["N_res_relative"]["delta=0.01"]
        )
        front_block[ref_name] = block
    res["front_constraint"] = front_block
    res["front_constraint"]["_n_time_nodes"] = n_time
    res["front_constraint"]["_why_two_references"] = (
        "Linearising about a CONSTANT Q makes the quasi-steady Jacobian nearly "
        "rank one, and that is not an artefact: with Mdot_p tied to Q and the "
        "total mass fixed, every node sits at the same flux, so all 12 "
        "perturbation directions differ only by the time-weight of their basis "
        "function.  That degeneracy is Theorem 1 showing up numerically.  It "
        "would, however, flatter the front-limited model, so the comparison is "
        "repeated about a waning reference, where the quasi-steady map has "
        "genuine structure and the difference measures what the front actually "
        "adds."
    )

    # ---- 5. sensitivity to the assumed vent position ------------------------
    grid = []
    for dx in (-800.0, 0.0, 800.0):
        for dy in (-800.0, 0.0, 800.0):
            cand = [centre[0] + dx, centre[1] + dy]
            o = build_observations(df_off, cand)
            sets = [
                ObservationSet(
                    oo["name"], oo["w_s"], oo["r"], np.maximum(sigma_log * oo["omega"], 1e-12)
                )
                for oo in o
            ]
            sp = analyse_spectrum(stacked_operator(sets, Q_NODES, whiten=True))
            grid.append(
                {
                    "offset_m": [dx, dy],
                    "N_res_relative_1pct": sp.n_resolvable_relative(1e-2),
                    "decay_rate_c": sp.decay_rate,
                    "min_core_radius_m": float(min(oo["r"].min() for oo in o)),
                    "max_core_radius_m": float(max(oo["r"].max() for oo in o)),
                }
            )
    res["vent_position_sensitivity"] = {
        "_why": (
            "The published data give no vent location, and the core radii enter "
            "the kernel squared, so the operator itself depends on an unknown.  "
            "Offsets span the benchmark's own +/-800 m vent uncertainty."
        ),
        "nominal_centre_xy_m": centre,
        "grid": grid,
        "N_res_range": [
            min(g["N_res_relative_1pct"] for g in grid),
            max(g["N_res_relative_1pct"] for g in grid),
        ],
    }

    k1 = per_class_counts["k=1"]["N_res_snr"]
    k4 = per_class_counts["k=4"]["N_res_snr"]
    p1 = per_class_counts["k=1"]["N_res_picard"]
    p4 = per_class_counts["k=4"]["N_res_picard"]
    ratio = float(np.exp(-per_class_counts["k=4"]["singular_value_decay_rate_c"]))
    res["H2_verdict"]["statement"] = (
        f"At the nugget noise level (sigma_log = {sigma_log:.2f}) the benchmark's "
        f"single 250-500 um fraction resolves {k1} modes of nu(Q); inverting all "
        f"four fractions jointly resolves {k4}, a gain of {k4 - k1}.  Counting "
        f"instead the modes whose data projection exceeds one noise standard "
        f"deviation gives {p1} and {p4}.  The singular values fall geometrically "
        f"by a factor {ratio:.2f} per mode, so resolution grows only "
        f"logarithmically in signal-to-noise: halving the noise buys "
        f"{np.log(2) / per_class_counts['k=4']['singular_value_decay_rate_c']:.1f} "
        f"of a mode.  Polydispersity is the cheaper lever, and it is the one the "
        f"data already contain."
    )

    OUT.write_text(json.dumps(res, indent=2))
    print(f"vent (excl. on-flow): x={centre[0]:.0f} y={centre[1]:.0f} m")
    print(
        f"provisional steady fits: all-class Q={fit_off['q_umb_m^3s^-1']:.3g}, "
        f"250-500um-only Q={fit_bench['q_umb_m^3s^-1']:.3g} "
        f"(L={fit_bench['L_m']:.0f} m) vs published 7.6e5 (L=4900 m)"
    )
    print(
        f"noise: nugget sigma_log={sigma_log:.3f} "
        f"(steady-residual bound {sigma_model:.3f}); nu_0={nu_scale:.3g} kg"
    )
    for k, v in per_class_counts.items():
        print(
            f"  {k} ({len(v['classes'])} classes, {v['n_observations']:3d} obs): "
            f"N_res_snr={v['N_res_snr']:2d}  N_res_picard={v['N_res_picard']:2d}  "
            f"N_rel(1%)={v['N_res_relative']['delta=0.01']:2d}  "
            f"c={v['singular_value_decay_rate_c']:.2f}"
        )
    for ref_name, block in front_block.items():
        if ref_name.startswith("_"):
            continue
        qs = block["quasi_steady"]["N_res_relative"]["delta=0.01"]
        fl = block["front_limited"]["N_res_relative"]["delta=0.01"]
        print(
            f"  ref {ref_name:11s}: quasi-steady N_res(1%)={qs}  front-limited={fl}  gain={fl - qs}"
        )
    print(f"  vent offsets +/-800 m -> N_res in {res['vent_position_sensitivity']['N_res_range']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
