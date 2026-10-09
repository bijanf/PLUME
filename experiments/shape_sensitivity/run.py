"""Sensitivity of the clast-shape result to the diameter convention and top bin.

Writes ``results/shape_sensitivity.json``.

Barreyre et al. (2011) fitted the Ferguson and Church coefficients (C1, C2) to
half-phi sieve fractions, taking D as the opening of the retaining sieve.  The
NESCA fractions are one phi wide and the settling law is evaluated at their
geometric midpoint.  A one-phi bin [a, 2a] holds the half-phi bins [a, sqrt2 a]
and [sqrt2 a, 2a], whose Barreyre diameters a and sqrt2 a have geometric mean
2^(1/4) a, that is 2^(-1/4) times the geometric midpoint.  The open >500 um
fraction also needs an upper bound, set to 1 mm in ``data.FRACTIONS``.

This script repeats the profile test of ``experiments/shape_space/run.py`` (a
log-normal nu(Q) shared by the fractions, one free amplitude per fraction) with
every diameter scaled by f in {2^(-1/2), 2^(-1/4), 1} (sieve lower bound,
half-phi equivalent, geometric midpoint) and with the >500 um bin truncated at
1, 1.5 and 2 mm.  For each case it reports the shape weights and rescales the
vent-sampled heat-flux posterior of each shape by (<Q>_case / <Q>_midpoint)^(4/3),
with <Q> the profile test's mass-weighted mean flux, to give the heat flux per
shape and the shape-weighted mixture.
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

OUT = ROOT / "results" / "shape_sensitivity.json"
FACTORS = {"sieve_lower_bound": 2**-0.5, "half_phi_equivalent": 2**-0.25, "geometric_midpoint": 1.0}
TOP_BIN_UPPER_M = (1.0e-3, 1.5e-3, 2.0e-3)
VOLC = ("blocky", "long", "sheet")
SEED = 20260923


def _shape_space():
    path = ROOT / "experiments" / "shape_space" / "run.py"
    spec = importlib.util.spec_from_file_location("shape_space_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ss = _shape_space()
    u = json.loads((ROOT / "results" / "nesca_unsteady.json").read_text())
    vs = json.loads((ROOT / "results" / "vent_shape.json").read_text())
    df = load_nesca()
    keep = ~df["Location"].astype(str).str.contains("on flow", case=False, na=False).values
    cx, cy = u["vent_xy_m"]
    r_all = np.hypot(df["x_m"].values - cx, df["y_m"].values - cy)
    rad, ys, dives = [], [], []
    dive_all = df["dive"].astype(str).values
    for name, *_ in FRACTIONS:
        v = df[f"glass_g_per_m2[{name}]"].values / 1000.0
        g = keep & np.isfinite(v) & (v > 0)
        rad.append(r_all[g])
        ys.append(np.log(v[g]))
        dives.append(dive_all[g])
    n_obs = int(sum(y.size for y in ys))
    n_par = 2 + len(FRACTIONS)
    pt = ss.ProfileTest(ss.MU_FINE, ss.S_FINE, ss.LQN_FINE)

    def profile(diams):
        by = {}
        for s in VOLC:
            w = settling.settling_velocity(diams, shape=s)
            tot = sum(pt.rss_by_fraction(pt.tables(rad, w), ys))
            best, ks, km = pt.minimise(tot)
            # dense-grid minimum as well, since the parabolic refinement in s is
            # approximate on an asymmetric profile
            by[s] = {
                "rss_refined": float(best[0]),
                "rss_grid": float(tot.min()),
                "q_mass_weighted_mean_m3s": float(
                    np.exp(ss.MU_FINE[km[0]] + 0.5 * ss.S_FINE[ks[0]] ** 2)
                ),
                "w_s_m_s": [float(x) for x in w],
            }
        for key in ("rss_refined", "rss_grid"):
            rmin = min(b[key] for b in by.values())
            s2 = rmin / (n_obs - n_par)
            d = np.array([(by[s][key] - rmin) / s2 for s in VOLC])
            wgt = np.exp(-0.5 * (d - d.min()))
            wgt /= wgt.sum()
            for s, dd, ww in zip(VOLC, d, wgt, strict=True):
                by[s][f"delta_chi2_{key[4:]}"] = float(dd)
                by[s][f"weight_{key[4:]}"] = float(ww)
        return by

    mids = np.array([settling.sieve_midpoint(f[1], f[2]) for f in FRACTIONS])
    ref = profile(mids)
    phi_samples = {s: np.asarray(vs["phi_samples_W"]["by_shape"][s]) for s in VOLC}
    rng = np.random.default_rng(SEED)

    cases = {}
    for top in TOP_BIN_UPPER_M:
        base = mids.copy()
        base[-1] = settling.sieve_midpoint(FRACTIONS[-1][1], top)
        for lab, f in FACTORS.items():
            key = f"{lab}__top_{top * 1e3:g}mm"
            by = profile(f * base)
            phi_by = {}
            scaled = {}
            for s in VOLC:
                fac = (by[s]["q_mass_weighted_mean_m3s"] / ref[s]["q_mass_weighted_mean_m3s"]) ** (
                    4.0 / 3.0
                )
                scaled[s] = phi_samples[s] * fac
                phi_by[s] = {
                    "median_W": float(np.median(scaled[s])),
                    "ci95_W": [
                        float(np.percentile(scaled[s], 2.5)),
                        float(np.percentile(scaled[s], 97.5)),
                    ],
                    "scale_factor": float(fac),
                }
            wts = np.array([by[s]["weight_grid"] for s in VOLC])
            n_draw = 4000
            pick = rng.choice(len(VOLC), size=n_draw, p=wts)
            mix = np.array([rng.choice(scaled[VOLC[k]]) for k in pick])
            cases[key] = {
                "diameter_factor": f,
                "top_bin_upper_m": top,
                "diameters_um": [float(x * 1e6) for x in f * base],
                "profile_by_shape": by,
                "phi_by_shape": phi_by,
                "phi_mixture_W": {
                    "median": float(np.median(mix)),
                    "ci95": [float(np.percentile(mix, 2.5)), float(np.percentile(mix, 97.5))],
                },
                "best_shape": VOLC[int(np.argmin([by[s]["delta_chi2_grid"] for s in VOLC]))],
            }
            print(
                f"{key:40s} weights "
                + " ".join(f"{s}={by[s]['weight_grid']:.2f}" for s in VOLC)
                + "  Phi "
                + " ".join(f"{s}={phi_by[s]['median_W'] / 1e12:.2f}" for s in VOLC)
                + f"  mix={cases[key]['phi_mixture_W']['median'] / 1e12:.2f} TW",
                flush=True,
            )

    # Which fractions carry the evidence for a flux spread: the single-flux
    # penalty of the profile test under blocky settling, with every fraction and
    # with each fraction left out in turn (reference representation).
    def single_flux_penalty(keep_idx):
        w = settling.settling_velocity(mids[keep_idx], shape="blocky")
        rr = [rad[i] for i in keep_idx]
        yy = [ys[i] for i in keep_idx]
        tot = sum(pt.rss_by_fraction(pt.tables(rr, w), yy))
        per_s = tot.reshape(ss.S_FINE.size, ss.MU_FINE.size).min(axis=1)
        n_o = int(sum(y.size for y in yy))
        s2 = per_s.min() / (n_o - (2 + len(keep_idx)))
        return float((per_s[0] - per_s.min()) / s2)

    names = [f[0] for f in FRACTIONS]
    loo = {"all_fractions": single_flux_penalty(list(range(len(names))))}
    for j, nm in enumerate(names):
        loo[f"without_{nm}"] = single_flux_penalty([i for i in range(len(names)) if i != j])

    # Leave one ROV dive out: the single-flux penalty and the shape weights at
    # the reference representation, to show which parts of the survey carry
    # the evidence for a flux spread and for the shape ranking.
    def subset_scores(masks):
        rr = [r[m] for r, m in zip(rad, masks, strict=True)]
        yy = [y[m] for y, m in zip(ys, masks, strict=True)]
        n_o = int(sum(y.size for y in yy))
        out = {}
        rss = {}
        for sh in VOLC:
            w = settling.settling_velocity(mids, shape=sh)
            tot = sum(pt.rss_by_fraction(pt.tables(rr, w), yy))
            per_s = tot.reshape(ss.S_FINE.size, ss.MU_FINE.size).min(axis=1)
            rss[sh] = (float(per_s.min()), float(per_s[0]))
        rmin = min(v[0] for v in rss.values())
        s2 = rmin / (n_o - n_par)
        d = np.array([(rss[sh][0] - rmin) / s2 for sh in VOLC])
        wgt = np.exp(-0.5 * (d - d.min()))
        wgt /= wgt.sum()
        out["n_observations"] = n_o
        out["weights"] = {sh: float(x) for sh, x in zip(VOLC, wgt, strict=True)}
        out["single_flux_penalty_blocky"] = float((rss["blocky"][1] - rss["blocky"][0]) / s2)
        return out

    by_dive = {}
    for dv in sorted(set(np.concatenate(dives))):
        by_dive[f"without_{dv}"] = subset_scores([d_ != dv for d_ in dives])
    by_radius = {}
    for rcut in (4000.0, 6000.0):
        by_radius[f"r_below_{rcut / 1e3:g}km"] = subset_scores([r < rcut for r in rad])

    mix_medians = [c["phi_mixture_W"]["median"] for c in cases.values()]
    allphi = {s: [c["phi_by_shape"][s]["median_W"] for c in cases.values()] for s in VOLC}
    res = {
        "_description": (
            "Sensitivity of the profile-test shape weights and of the heat "
            "flux per shape to the diameter assigned to each sieve fraction "
            "and to the upper bound of the open >500 um fraction."
        ),
        "_generated": datetime.date.today().isoformat(),
        "_script": "experiments/shape_sensitivity/run.py",
        "_git_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "_inputs": [
            "experiments/shape_space/run.py (profile test)",
            "results/vent_shape.json#phi_samples_W.by_shape",
            "results/nesca_unsteady.json#vent_xy_m",
        ],
        "diameter_factors": FACTORS,
        "top_bin_upper_m": list(TOP_BIN_UPPER_M),
        "reference_geometric_midpoint_1mm": ref,
        "single_flux_penalty_blocky_delta_chi2": loo,
        "leave_one_dive_out": by_dive,
        "radial_cut": by_radius,
        "cases": cases,
        "summary": {
            "phi_mixture_median_range_W": [float(min(mix_medians)), float(max(mix_medians))],
            "phi_by_shape_median_range_W": {
                s: [float(min(v)), float(max(v))] for s, v in allphi.items()
            },
            "best_shape_by_case": {k: c["best_shape"] for k, c in cases.items()},
            "sheet_weight_range": [
                float(min(c["profile_by_shape"]["sheet"]["weight_grid"] for c in cases.values())),
                float(max(c["profile_by_shape"]["sheet"]["weight_grid"] for c in cases.values())),
            ],
        },
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
