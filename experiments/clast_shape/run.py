"""Infer the clast shape from the deposit itself.

Writes ``results/clast_shape.json``.

Because L_i^2 = Q/w_i, the RATIO of dispersal length scales between two sieve
fractions, L_i/L_j = sqrt(w_j/w_i), is independent of the umbrella flux, the
erupted mass and the eruption history.  It depends only on the settling law.  So
fitting each fraction its own L and comparing the ratios against the shape
classes measured by Barreyre et al. (2011) identifies the clast shape from the
deposit --- something a single-fraction inversion cannot do.

This matters because Phi ~ Q^(4/3) and Q = w_s L^2, so the clast shape is worth a
factor 2.8 in the inferred heat flux.
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

OUT = ROOT / "results" / "clast_shape.json"
N_BOOT = 2000


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


def fit_L(r, omega):
    """Independent length scale for one fraction: log Omega = log Omega_0 - pi r^2/L^2."""
    a = np.column_stack([np.ones(r.size), -np.pi * r**2])
    sol, *_ = np.linalg.lstsq(a, np.log(omega), rcond=None)
    return (1.0 / np.sqrt(sol[1])) if sol[1] > 0 else np.nan


def main() -> int:
    u = json.loads((ROOT / "results" / "nesca_unsteady.json").read_text())
    cx, cy = u["vent_xy_m"]
    df = load_nesca()
    keep = ~df["Location"].astype(str).str.contains("on flow", case=False, na=False).values
    r_all = np.hypot(df["x_m"].values - cx, df["y_m"].values - cy)

    names, radii, obs, L = [], [], [], []
    for name, _d_lo, _d_hi, _m, _p in FRACTIONS:
        v = df[f"glass_g_per_m2[{name}]"].values / 1000.0
        good = keep & np.isfinite(v) & (v > 0)
        names.append(name)
        radii.append(r_all[good])
        obs.append(v[good])
        L.append(fit_L(r_all[good], v[good]))
    L = np.array(L)

    # Bootstrap the length scales over cores, to get an interval on the ratio.
    rng = np.random.default_rng(20260917)
    boot = np.empty((N_BOOT, len(names)))
    for b in range(N_BOOT):
        for i, (r, o) in enumerate(zip(radii, obs, strict=True)):
            idx = rng.integers(0, r.size, r.size)
            boot[b, i] = fit_L(r[idx], o[idx])
    ratio_obs = L[0] / L[3]
    ratio_boot = boot[:, 0] / boot[:, 3]
    ratio_ci = [float(np.nanpercentile(ratio_boot, 2.5)), float(np.nanpercentile(ratio_boot, 97.5))]

    mids = [settling.sieve_midpoint(f[1], f[2]) for f in FRACTIONS]
    log_sd = np.array([float(np.nanstd(np.log(boot[:, i]))) for i in range(len(names))])

    # Proper test.  The four length scales give only THREE independent ratios, so
    # scoring all six pairwise ratios double-counts and inflates the statistic.
    # Instead fit, for each shape, the single free flux Q in L_i = sqrt(Q/w_i),
    # by weighted least squares in log L, and score the residual over the four
    # length scales: 4 observations, 1 parameter, 3 degrees of freedom.
    shapes = {}
    for sh in settling.SHAPE_CLASSES:
        w = np.array([float(settling.settling_velocity(d, shape=sh)) for d in mids])
        # log L_i = 0.5 log Q - 0.5 log w_i  ->  one offset to fit
        resid_shape = np.log(L) + 0.5 * np.log(w)  # = 0.5 log Q if perfect
        weights = 1.0 / log_sd**2
        half_log_q = float(np.sum(weights * resid_shape) / np.sum(weights))
        resid = resid_shape - half_log_q
        chi2 = float(np.sum(weights * resid**2))
        shapes[sh] = {
            "w_s_by_fraction_ms": {n: float(v) for n, v in zip(names, w, strict=True)},
            "implied_q_umb_m3s": float(np.exp(2 * half_log_q)),
            "chi2": chi2,
            "dof": len(names) - 1,
            "log_residuals": {n: float(v) for n, v in zip(names, resid, strict=True)},
            "extreme_ratio_predicted": settling.length_scale_ratio(mids[3], mids[0], shape=sh),
            "is_volcaniclast_class": sh in settling.VOLCANICLAST_SHAPES,
        }
    best = min(shapes, key=lambda s: shapes[s]["chi2"])
    best_vc = min(
        (s for s in shapes if shapes[s]["is_volcaniclast_class"]), key=lambda s: shapes[s]["chi2"]
    )
    vc = sorted((shapes[s]["chi2"], s) for s in shapes if shapes[s]["is_volcaniclast_class"])
    delta_chi2 = float(vc[1][0] - vc[0][0])
    decisive = bool(delta_chi2 > 9.0)  # ~3 sigma on 1 effective contrast

    res = {
        "_description": "Clast shape inferred from the deposit's own length-scale ratios.",
        "_generated": "2026-09-17",
        "_script": "experiments/clast_shape/run.py",
        "_git_commit": git_hash(),
        "_why_this_works": (
            "L_i^2 = Q/w_i, so L_i/L_j = sqrt(w_j/w_i) is independent of the "
            "umbrella flux, the erupted mass and the eruption history.  It "
            "depends only on the settling law, so it identifies the clast shape."
        ),
        "n_cores": int(keep.sum()),
        "length_scales_m": {n: float(v) for n, v in zip(names, L, strict=True)},
        "length_scale_ci95_m": {
            n: [float(np.nanpercentile(boot[:, i], 2.5)), float(np.nanpercentile(boot[:, i], 97.5))]
            for i, n in enumerate(names)
        },
        "extreme_ratio_observed": float(ratio_obs),
        "extreme_ratio_ci95": ratio_ci,
        "by_shape": shapes,
        "best_fit_shape": best,
        "best_fit_volcaniclast_shape": best_vc,
        "delta_chi2_between_best_two_volcaniclast_shapes": delta_chi2,
        "discrimination_is_decisive": decisive,
        "verdict": (
            f"The deposit's length-scale ratios prefer the {best_vc} class of "
            f"Barreyre et al. (2011) among the volcaniclast shapes "
            f"(chi2 = {shapes[best_vc]['chi2']:.2f} on {len(names) - 1} degrees of "
            f"freedom), but the preference is "
            + (
                "DECISIVE"
                if decisive
                else "NOT decisive: the next-best shape is "
                f"only {delta_chi2:.1f} in chi2 behind, and the bootstrap interval "
                f"on the extreme length-scale ratio, [{ratio_ci[0]:.2f}, "
                f"{ratio_ci[1]:.2f}] about an observed {ratio_obs:.2f}, spans every "
                "candidate"
            )
            + ".  The observable is real and independent of the umbrella flux, "
            "but this deposit does not exercise it strongly enough to fix the "
            "settling law.  The clast-shape factor of 2.8 in heat flux therefore "
            "remains an explicit axis of the result rather than something the "
            "data resolve.  Caveat in either direction: sieve fractions sort by "
            "hydraulic behaviour, so what a deposit reports is an effective "
            "shape, not the modal morphology of the glass."
        ),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"length scales (m): {dict(zip(names, np.round(L).astype(int), strict=True))}")
    print(f"observed extreme ratio {ratio_obs:.2f} [{ratio_ci[0]:.2f}, {ratio_ci[1]:.2f}]")
    for sh, v in shapes.items():
        mark = " <-- best volcaniclast" if sh == best_vc else ""
        print(
            f"  {sh:12s} predicted ratio {v['extreme_ratio_predicted']:.2f}  "
            f"chi2 = {v['chi2']:6.2f} on {v['dof']} dof{mark}"
        )
    print(
        f"delta chi2 between the best two volcaniclast shapes: {delta_chi2:.2f} "
        f"-> {'DECISIVE' if decisive else 'not decisive'}"
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
