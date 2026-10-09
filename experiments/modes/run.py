"""Mode structure of the NESCA forward operator: what each resolvable mode of
nu(Q) looks like in flux space and in data space, and how sharply the deposit
resolves the flux distribution.

Writes ``results/modes.json``.

The operator is the one of ``experiments/identifiability/run.py``, rebuilt with
that script's own functions (vent position, observations, nugget noise, 60-node
log-Q grid, class order, whitening, prior mass scale), so the mode counts it
reports there are reproduced here before anything new is computed.

What is new
-----------
1. The right singular vectors v_n(Q) of the four-fraction whitened operator,
   n = 1..10, and where in Q each has its support.
2. The deposit pattern each mode produces, as a function of radius and
   settling speed.  Divided by w it depends on the Laplace abscissa
   lambda = pi w r^2 alone, so it is stored as F_n(lambda).
3. Model resolution R = V_k V_k^T and the width of the averaging kernel at the
   posterior mass-weighted mean flux, for one fraction and for four.
4. The signal-to-noise mode count over (observation noise, number of fractions)
   and over the number of cores; the effect of halving the noise.
5. The sampled Laplace-abscissa range for one fraction and for four, and the
   share of the whitened data norm carried by the leading modes.
6. A units check on the prior mass scale nu_0 used by the signal-to-noise count.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from plume_inv import settling  # noqa: E402
from plume_inv.data import FRACTIONS, load_nesca  # noqa: E402
from plume_inv.svd import (  # noqa: E402
    ObservationSet,
    analyse_spectrum,
    averaging_kernel_row,
    captured_fraction,
    laplace_response,
    main_lobe_fwhm,
    orient_modes,
    resolution_matrix,
    sign_changes,
    stacked_operator,
)

OUT = ROOT / "results" / "modes.json"
N_MODES = 20
CLASS_ORDER = [2, 1, 3, 0]  # as in identifiability: 250-500 um first

# Prior mass scale for the signal-to-noise count.  "single_conversion" uses the
# steady-fit tephra mass with one g -> kg conversion, as identifiability/run.py
# now does.  "as_identifiability" reproduces the earlier, doubly converted value
# (1000x too small) and is kept only so the effect of that bug stays on record.
# See section 8 of the output.
NU0_CONVENTION = "single_conversion"


def _load_identifiability_module():
    spec = importlib.util.spec_from_file_location(
        "identifiability_run", ROOT / "experiments" / "identifiability" / "run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ID = _load_identifiability_module()
Q_NODES = ID.Q_NODES
LOG10_Q = np.log10(Q_NODES)


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


def rounded(a, sig: int = 6):
    a = np.asarray(a, dtype=float)
    return (
        [float(f"{v:.{sig}g}") for v in a.ravel()]
        if a.ndim == 1
        else [rounded(row, sig) for row in a]
    )


def nu_scales(fit_off):
    """Prior mass scale nu_0, as computed in identifiability and with units fixed.

    ``fit_steady_centre`` fits log Omega_0 to observations already converted to
    kg m^-2, and identifiability divides exp(log Omega_0) by 1000 a second time.
    Both values are returned so the effect of that conversion is on record.
    """
    as_id, fixed = 0.0, 0.0
    for i, (_name, d_lo, d_hi, _m, _p) in enumerate(FRACTIONS):
        w_i = float(settling.settling_velocity(settling.sieve_midpoint(d_lo, d_hi)))
        omega0 = np.exp(fit_off["log_omega0_by_class"][i])
        as_id += omega0 / 1000.0 * fit_off["q_umb_m^3s^-1"] / w_i
        fixed += omega0 * fit_off["q_umb_m^3s^-1"] / w_i
    return float(as_id), float(fixed)


def operator(obs, sigma_log, which):
    chosen = [obs[i] for i in which]
    sets = [
        ObservationSet(o["name"], o["w_s"], o["r"], np.maximum(sigma_log * o["omega"], 1e-12))
        for o in chosen
    ]
    a = stacked_operator(sets, Q_NODES, whiten=True)
    d = np.concatenate([o["omega"] / np.maximum(sigma_log * o["omega"], 1e-12) for o in chosen])
    return a, d


def fwhm_block(r_mat, log10_target):
    row = averaging_kernel_row(r_mat, LOG10_Q, log10_target)
    lobe = main_lobe_fwhm(LOG10_Q, row)
    return {
        "fwhm_log10Q": lobe["fwhm"],
        "lobe_log10Q": [lobe["x_low"], lobe["x_high"]],
        "peak_log10Q": lobe["x_peak"],
        "peak_offset_from_target_log10Q": lobe["x_peak"] - log10_target,
        "censored_low": lobe["censored_low"],
        "censored_high": lobe["censored_high"],
        "row": rounded(row),
    }


def support_quantiles(weights, lo=0.05, hi=0.95):
    c = np.cumsum(weights) / np.sum(weights)
    return [float(np.interp(lo, c, LOG10_Q)), float(np.interp(hi, c, LOG10_Q))]


def main() -> int:
    df = load_nesca()
    fit_off = ID.fit_steady_centre(df, exclude_on_flow=True)
    centre = fit_off["centre_xy_m"]
    # Same cores as the inversion: those logged 'on flow' are excluded.
    df = df[~df["Location"].astype(str).str.contains("on flow", case=False, na=False)]
    sigma_log = ID.nugget_noise(df)["pooled_sigma_log"]
    obs = ID.build_observations(df, centre)
    nu0_id, nu0_fixed = nu_scales(fit_off)
    nu0 = nu0_id if NU0_CONVENTION == "as_identifiability" else nu0_fixed
    ident = json.loads((ROOT / "results" / "identifiability.json").read_text())
    unsteady = json.loads((ROOT / "results" / "nesca_unsteady.json").read_text())
    q_mean = unsteady["posterior"]["q_mass_weighted_mean_m^3s^-1"]["median"]
    q_mean_ci = unsteady["posterior"]["q_mass_weighted_mean_m^3s^-1"]["ci95"]
    log10_qm = float(np.log10(q_mean))

    res: dict = {
        "_description": (
            "Mode structure of the NESCA quasi-steady forward operator: "
            "right singular vectors of nu(Q), their deposit patterns in "
            "(radius, settling speed), model resolution, and the "
            "signal-to-noise mode count against noise and data volume."
        ),
        "_generated": datetime.date.today().isoformat(),
        "_script": "experiments/modes/run.py",
        "_git_commit": git_hash(),
        "_data": df.attrs["source"],
        "construction": {
            "source": (
                "experiments/identifiability/run.py: build_observations, "
                "fit_steady_centre(exclude_on_flow=True), nugget_noise, Q_NODES, "
                "class order 250-500, 125-250, >500, 63-125 um, whitening by "
                "sigma_log * Omega_j, nu_0 as in that script."
            ),
            "centre_xy_m": centre,
            "sigma_log": sigma_log,
            "nu0_kg": nu0,
            "nu0_convention": NU0_CONVENTION,
            "n_q_nodes": int(Q_NODES.size),
            "q_range_m3s": [float(Q_NODES[0]), float(Q_NODES[-1])],
            "shape": settling.DEFAULT_SHAPE,
            "w_s_by_class_m_s": {o["name"]: o["w_s"] for o in obs},
            "n_obs_by_class": {o["name"]: int(o["r"].size) for o in obs},
            "note": (
                "results/identifiability.json was written before the settling law "
                "was pinned to the blocky deep-sea volcaniclast coefficients, so its "
                "vent position and singular values differ slightly from this "
                "rebuild; the mode counts it reports are reproduced below."
            ),
        },
        "q_mass_weighted_mean_m3s": {
            "median": q_mean,
            "ci95": q_mean_ci,
            "log10": log10_qm,
            "source": "results/nesca_unsteady.json#posterior.q_mass_weighted_mean_m^3s^-1",
        },
    }

    # ---- 1. reproduce the identifiability counts ---------------------------
    spectra, verify = {}, {}
    for k in range(1, 5):
        a, d = operator(obs, sigma_log, CLASS_ORDER[:k])
        sp = analyse_spectrum(a)
        spectra[k] = (a, d, sp)
        stored = ident["modes_vs_size_classes"][f"k={k}"]
        verify[f"k={k}"] = {
            "classes": [obs[i]["name"] for i in CLASS_ORDER[:k]],
            "N_res_snr": sp.n_resolvable_snr(nu0),
            "N_res_picard": sp.picard_truncation(d),
            "stored_N_res_snr": stored["N_res_snr"],
            "stored_N_res_picard": stored["N_res_picard"],
            "decay_ratio_per_mode": float(np.exp(-sp.decay_rate)),
            "singular_values_first_20": rounded(sp.singular_values[:20]),
            "snr_first_20": rounded(sp.singular_values[:20] * nu0),
        }
    a_nat, d_nat = operator(obs, sigma_log, [0, 1, 2, 3])
    sp_nat = analyse_spectrum(a_nat)
    picard_all = sp_nat.picard_truncation(d_nat)
    reproduced = (
        all(v["N_res_snr"] == v["stored_N_res_snr"] for v in verify.values())
        and verify["k=1"]["N_res_picard"] == verify["k=1"]["stored_N_res_picard"]
        and picard_all == ident["picard"]["truncation_index"]
    )
    res["verification"] = {
        "by_fraction_count": verify,
        "picard_truncation_four_fractions": picard_all,
        "stored_picard_truncation": ident["picard"]["truncation_index"],
        "reproduces_identifiability": bool(reproduced),
    }
    if not reproduced:
        raise SystemExit("identifiability counts not reproduced; stopping")

    a4, d4, sp4 = spectra[4]
    a1, d1, sp1 = spectra[1]
    n_snr4 = verify["k=4"]["N_res_snr"]
    n_snr1 = verify["k=1"]["N_res_snr"]
    n_pic = picard_all
    res["picard_plot"] = {
        "singular_values_over_s1": rounded(sp4.singular_values[:40] / sp4.singular_values[0]),
        "coefficients_abs_uT_d": rounded(sp4.picard(d4)[:40]),
        "truncation_index": n_pic,
    }

    # ---- 2. right singular vectors of nu(Q) --------------------------------
    vt4 = orient_modes(sp4.vt)
    vt1 = orient_modes(sp1.vt)
    modes = []
    for n in range(N_MODES):
        v = vt4[n]
        modes.append(
            {
                "n": n + 1,
                "sigma_over_sigma1": float(sp4.singular_values[n] / sp4.singular_values[0]),
                "snr": float(sp4.singular_values[n] * nu0),
                "peak_log10Q": float(LOG10_Q[np.argmax(np.abs(v))]),
                "support_log10Q_5_95": support_quantiles(v**2),
                "sign_changes": sign_changes(v),
                "v_at_mean_flux_over_max": float(
                    np.interp(log10_qm, LOG10_Q, v) / np.max(np.abs(v))
                ),
            }
        )
    res["right_singular_vectors"] = {
        "_convention": (
            "rows of V^T of the four-fraction whitened operator, unit norm "
            "over the 60 nodes, sign fixed so the largest entry is positive"
        ),
        "log10_q_nodes": rounded(LOG10_Q),
        "v_four_fractions": rounded(vt4[:N_MODES]),
        "v1_one_fraction": rounded(vt1[0]),
        "per_mode": modes,
        "signal_to_noise_truncation": n_snr4,
        "picard_truncation": n_pic,
    }

    # ---- 3. deposit pattern of each mode in (r, w) -------------------------
    lam_grid = np.geomspace(1.0, 1e9, 541)
    f_modes = np.array([laplace_response(Q_NODES, vt4[n], lam_grid) for n in range(6)])
    zeros = []
    for f in f_modes:
        s = np.sign(f)
        idx = np.where(s[1:] != s[:-1])[0]
        zeros.append([float(np.sqrt(lam_grid[i] * lam_grid[i + 1])) for i in idx])
    per_class_lam = {}
    for o in obs:
        lam = np.pi * o["w_s"] * o["r"] ** 2
        per_class_lam[o["name"]] = {
            "w_s": o["w_s"],
            "r_min_m": float(o["r"].min()),
            "r_max_m": float(o["r"].max()),
            "lambda_min_m3s": float(lam.min()),
            "lambda_max_m3s": float(lam.max()),
            "lambda_ratio": float(lam.max() / lam.min()),
            "r_ratio_squared": float((o["r"].max() / o["r"].min()) ** 2),
        }
    lam_all = np.concatenate([np.pi * o["w_s"] * o["r"] ** 2 for o in obs])
    w_all = np.array([o["w_s"] for o in obs])
    one = per_class_lam["250-500um"]
    res["data_space_patterns"] = {
        "_definition": (
            "F_n(lambda) = sum_k v_{n,k} exp(-lambda/Q_k)/Q_k = dOmega_n(r, w)/w "
            "with lambda = pi w r^2; the deposit change produced by mode n in "
            "a class of settling speed w at radius r is w F_n(pi w r^2)."
        ),
        "lambda_grid_m3s": rounded(lam_grid),
        "F_modes_1_to_6": rounded(f_modes),
        "zero_crossings_lambda_m3s": zeros,
        "n_zero_crossings": [len(z) for z in zeros],
    }
    one_rng = [one["lambda_min_m3s"], one["lambda_max_m3s"]]
    all_rng = [float(lam_all.min()), float(lam_all.max())]
    res["data_space_patterns"]["zero_crossings_inside_sampled_range"] = {
        "one_fraction_250_500": [sum(one_rng[0] <= z <= one_rng[1] for z in zz) for zz in zeros],
        "four_fractions": [sum(all_rng[0] <= z <= all_rng[1] for z in zz) for zz in zeros],
    }
    res["laplace_abscissa"] = {
        "per_class": per_class_lam,
        "one_fraction_250_500": {
            "lambda_range_m3s": [one["lambda_min_m3s"], one["lambda_max_m3s"]],
            "factor": one["lambda_ratio"],
            "decades": float(np.log10(one["lambda_ratio"])),
            "r_max_over_r_min_squared": one["r_ratio_squared"],
        },
        "four_fractions": {
            "lambda_range_m3s": [float(lam_all.min()), float(lam_all.max())],
            "factor": float(lam_all.max() / lam_all.min()),
            "decades": float(np.log10(lam_all.max() / lam_all.min())),
            "w_max_over_w_min": float(w_all.max() / w_all.min()),
            "gain_over_one_fraction": float((lam_all.max() / lam_all.min()) / one["lambda_ratio"]),
        },
    }

    # ---- 4. data norm carried by the leading modes -------------------------
    res["data_norm_captured"] = {
        "_definition": "sum_{n<=k} (u_n^T d)^2 / |d|^2 for the whitened data d",
        "four_fractions": {f"first_{k}": captured_fraction(sp4.u, d4, k) for k in (1, 2, 6, 10)},
        "one_fraction": {f"first_{k}": captured_fraction(sp1.u, d1, k) for k in (1, 2, 6)},
        "four_fractions_range_of_operator": captured_fraction(sp4.u, d4, sp4.u.shape[1]),
        "one_fraction_range_of_operator": captured_fraction(sp1.u, d1, sp1.u.shape[1]),
    }
    dn = res["data_norm_captured"]
    dn["four_fractions_share_of_range"] = {
        k: v / dn["four_fractions_range_of_operator"] for k, v in dn["four_fractions"].items()
    }

    # ---- 5. model resolution and averaging kernels -------------------------
    r1 = resolution_matrix(vt1, n_snr1)
    r4 = resolution_matrix(vt4, n_snr4)
    r4p = resolution_matrix(vt4, n_pic)
    r1p = resolution_matrix(vt1, verify["k=1"]["N_res_picard"])
    res["resolution"] = {
        "_definition": (
            "R = V_k V_k^T on the 60-node log-Q grid; the averaging kernel at "
            "<Q> is the row of R interpolated to log10 <Q>; widths are the full "
            "width at half maximum of the lobe that contains the row maximum."
        ),
        "target_log10Q": log10_qm,
        "one_fraction_snr": {"k": n_snr1, **fwhm_block(r1, log10_qm)},
        "one_fraction_picard": {"k": verify["k=1"]["N_res_picard"], **fwhm_block(r1p, log10_qm)},
        "four_fractions_snr": {"k": n_snr4, **fwhm_block(r4, log10_qm)},
        "four_fractions_picard": {"k": n_pic, **fwhm_block(r4p, log10_qm)},
        "resolved_window_log10Q_5_95": {
            "one_fraction_snr": support_quantiles(np.diag(r1)),
            "four_fractions_snr": support_quantiles(np.diag(r4)),
            "four_fractions_picard": support_quantiles(np.diag(r4p)),
        },
        "R_one_fraction_snr": rounded(r1, 5),
        "R_four_fractions_snr": rounded(r4, 5),
    }

    # ---- 6. mode count against noise and fractions -------------------------
    sig_grid = np.geomspace(0.05, 2.0, 161)
    count = np.array(
        [
            [analyse_spectrum_count(spectra[k][2], nu0, sigma_log, s) for s in sig_grid]
            for k in range(1, 5)
        ]
    )
    halving = {}
    for k in range(1, 5):
        sp = spectra[k][2]
        n0 = analyse_spectrum_count(sp, nu0, sigma_log, sigma_log)
        n_half = analyse_spectrum_count(sp, nu0, sigma_log, 0.5 * sigma_log)
        n_quarter = analyse_spectrum_count(sp, nu0, sigma_log, 0.25 * sigma_log)
        halving[f"k={k}"] = {
            "at_nugget": n0,
            "at_half_nugget": n_half,
            "at_quarter_nugget": n_quarter,
            "gain_from_halving": n_half - n0,
            "expected_gain_log2_over_c": float(np.log(2) / sp.decay_rate),
        }
    res["noise_vs_fractions"] = {
        "_definition": (
            "N_res_snr = #{n : sigma_n nu_0 sigma_nugget/sigma_log > 1}: "
            "whitening by sigma_log * Omega_j scales the whitened operator by "
            "1/sigma_log"
        ),
        "sigma_log_grid": rounded(sig_grid),
        "fraction_counts": [1, 2, 3, 4],
        "N_res_snr": count.astype(int).tolist(),
        "halving": halving,
        "gain_from_three_fractions_at_nugget": halving["k=4"]["at_nugget"]
        - halving["k=1"]["at_nugget"],
        "gain_from_halving_noise_four_fractions": halving["k=4"]["gain_from_halving"],
        "gain_from_halving_noise_one_fraction": halving["k=1"]["gain_from_halving"],
    }

    # ---- 7. mode count against the number of cores -------------------------
    rng = np.random.default_rng(20260923)
    n_cores_all = obs[0]["mask"].size
    sizes = [10, 20, 30, 45, 60, 80, 100, 120, n_cores_all]
    by_cores = {"n_cores": sizes, "n_draws": 40, "mean_N_res_snr": {}, "min_max": {}}
    for k in (1, 4):
        means, spans = [], []
        for m in sizes:
            vals = []
            for _ in range(1 if m == n_cores_all else 40):
                keep = np.zeros(n_cores_all, bool)
                keep[rng.choice(n_cores_all, m, replace=False)] = True
                sub = []
                for o in obs:
                    sel = keep[o["mask"]]
                    sub.append({**o, "r": o["r"][sel], "omega": o["omega"][sel]})
                a, _d = operator(sub, sigma_log, CLASS_ORDER[:k])
                vals.append(analyse_spectrum(a).n_resolvable_snr(nu0))
            means.append(float(np.mean(vals)))
            spans.append([int(min(vals)), int(max(vals))])
        by_cores["mean_N_res_snr"][f"k={k}"] = means
        by_cores["min_max"][f"k={k}"] = spans
    res["modes_vs_n_cores"] = by_cores

    # ---- 8. units check on nu_0 --------------------------------------------
    fixed_counts = {f"k={k}": spectra[k][2].n_resolvable_snr(nu0_fixed) for k in range(1, 5)}
    fixed_half = {
        f"k={k}": analyse_spectrum_count(spectra[k][2], nu0_fixed, sigma_log, 0.5 * sigma_log)
        for k in range(1, 5)
    }
    res["nu0_units_check"] = {
        "nu0_as_in_identifiability_kg": nu0_id,
        "nu0_single_conversion_kg": nu0_fixed,
        "posterior_total_mass_kg": unsteady["posterior"]["mass_total_kg"]["median"]
        if isinstance(unsteady["posterior"]["mass_total_kg"], dict)
        else unsteady["posterior"]["mass_total_kg"],
        "N_res_snr_single_conversion": fixed_counts,
        "N_res_snr_single_conversion_half_noise": fixed_half,
        "picard_unaffected": n_pic,
        "statement": (
            "fit_steady_centre fits log Omega_0 to observations already in kg m^-2, and "
            "identifiability divides exp(log Omega_0) by 1000 again, so its nu_0 is "
            "1000 times smaller than the total mass the steady fit implies.  With a "
            "single conversion nu_0 matches the posterior total mass, and the "
            "signal-to-noise counts rise.  The discrete Picard truncation does not "
            "use nu_0 and is unchanged."
        ),
    }

    OUT.write_text(json.dumps(res, indent=1))
    print(
        f"centre {centre[0]:.1f} {centre[1]:.1f}  sigma_log {sigma_log:.3f}  "
        f"nu0 {nu0:.4g} (single conversion {nu0_fixed:.4g})"
    )
    for k, v in verify.items():
        print(
            f"  {k}: snr {v['N_res_snr']} (stored {v['stored_N_res_snr']})  "
            f"picard {v['N_res_picard']} (stored {v['stored_N_res_picard']})"
        )
    print(f"  picard four fractions {picard_all}; reproduced {reproduced}")
    for m in modes:
        print(
            f"  v{m['n']}: peak {m['peak_log10Q']:.2f} support {m['support_log10Q_5_95']} "
            f"sign changes {m['sign_changes']} snr {m['snr']:.3g}"
        )
    for key in (
        "one_fraction_snr",
        "one_fraction_picard",
        "four_fractions_snr",
        "four_fractions_picard",
    ):
        b = res["resolution"][key]
        print(
            f"  {key}: k={b['k']} fwhm {b['fwhm_log10Q']:.3f} peak {b['peak_log10Q']:.2f} "
            f"censored {b['censored_low']},{b['censored_high']}"
        )
    print("  windows", res["resolution"]["resolved_window_log10Q_5_95"])
    print(
        "  lambda",
        res["laplace_abscissa"]["one_fraction_250_500"]["factor"],
        res["laplace_abscissa"]["four_fractions"]["factor"],
        res["laplace_abscissa"]["four_fractions"]["w_max_over_w_min"],
    )
    print("  captured", res["data_norm_captured"])
    print("  halving", halving)
    print("  by cores", by_cores["mean_N_res_snr"])
    print(
        "  zeros",
        res["data_space_patterns"]["n_zero_crossings"],
        res["data_space_patterns"]["zero_crossings_inside_sampled_range"],
    )
    print("  fixed nu0", fixed_counts, fixed_half)
    print(f"wrote {OUT}")
    return 0


def analyse_spectrum_count(sp, nu0, sigma_ref, sigma):
    """Signal-to-noise count when the log-space noise is ``sigma`` in place of
    ``sigma_ref``: the whitened operator scales by sigma_ref / sigma."""
    return sp.n_resolvable_snr(nu0 * sigma_ref / sigma)


if __name__ == "__main__":
    raise SystemExit(main())
