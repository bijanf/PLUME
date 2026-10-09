"""Clast shape over the continuous settling-law plane, and the power of the test.

Writes ``results/shape_space.json``.

Starts from the construction of ``experiments/clast_shape/run.py`` (per-fraction
length scales L_i, a bootstrap over cores for their log standard deviations,
one free flux per settling law, chi-squared over the four L_i; called the
"length-scale test" below) and checks that it reproduces
``results/clast_shape.json`` to machine precision.  Then:

1. chi2(C1, C2) of the length-scale test on a dense grid of the Ferguson and
   Church plane, flux profiled at each point.  Every ratio w_i/w_j depends on
   (C1, C2) through the transition diameter D* = ((C1 nu)^2/(0.75 C2 R g))^(1/3)
   alone, so the surface is a valley along C1^2/C2 = const.
2. The heat flux over the same plane, Phi = pref <Q>^(4/3), with <Q> converted
   from the length scales by the profiled flux of each law and calibrated
   against the sampled posteriors of the three measured volcaniclast classes,
   and a shape-marginalised Phi under a uniform prior on (C1, log C2).
3. Synthetic recovery of the shape class on the NESCA core geometry.
4. Power against noise level, number of cores and number of sieve fractions.
5. Bootstrap distributions of the observed extreme ratio and of the fitted D*.

The synthetic recovery exposes a bias of the length-scale test: a spread of the
umbrella flux makes each fraction's deposit a mixture of Gaussians, whose
log-linear fit over a fixed set of radii compresses the ratio of length scales
in the same direction as sheet settling.  A second classifier, the "profile
test", fits the quasi-steady kernel itself with a log-normal nu(Q) whose
log-mean and log-sd are shared by all fractions and profiled out, and one free
amplitude per fraction.  Both classifiers are run on the real deposit and on
every synthetic deposit.
"""

from __future__ import annotations

import datetime
import itertools
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from plume_inv import settling, shape, stem  # noqa: E402
from plume_inv.constants import EPSILON_ENTRAIN, NU_SEAWATER, RHO_TEPHRA, RHO_W, G  # noqa: E402
from plume_inv.data import FRACTIONS, load_nesca  # noqa: E402

OUT = ROOT / "results" / "shape_space.json"
N_BOOT_REAL = 2000  # as in experiments/clast_shape/run.py
SEED_REAL = 20260917  # as in experiments/clast_shape/run.py
SHAPES = list(settling.SHAPE_CLASSES)  # 5 named classes
VOLC = list(settling.VOLCANICLAST_SHAPES)  # blocky, long, sheet

# Continuous grid of the Ferguson and Church plane.
C1_GRID = np.linspace(15.0, 45.0, 121)
C2_GRID = np.geomspace(0.3, 100.0, 141)
BOX_NOMINAL = {"c1": [15.0, 45.0], "c2": [0.3, 30.0]}
BOX_EXTENDED = {"c1": [15.0, 45.0], "c2": [0.3, 100.0]}
C1_REF = 24.0  # any C1 serves: the chi2 depends on C1^2/C2 alone

# Profile test: log-normal nu(Q) grids (log-mean mu, log-sd s, quadrature nodes).
# The spread enters log F at order s^2, so the misfit is close to parabolic in
# s^2 and the log-sd grids are uniform in s^2, which the parabolic refinement needs.
MU_FINE = np.log(np.geomspace(5e4, 5e6, 231))
S_FINE = np.sqrt(np.linspace(0.0, 1.44, 17))
LQN_FINE = np.linspace(np.log(1e2), np.log(1e9), 645)
MU_POW = np.log(np.geomspace(1e5, 4e6, 93))
S_POW = np.sqrt(np.linspace(0.0, 0.81, 10))
LQN_POW = np.linspace(np.log(1e3), np.log(1e8), 288)
D_STAR_PROFILE = np.geomspace(60e-6, 1500e-6, 49)

# Synthetic recovery.
N_REP_MAIN = 500  # replicates per true shape, main confusion matrices
N_BOOT_SYN_MAIN = 2000  # bootstrap draws per fraction, main
N_VENT_DRAWS = 10  # vent draws in the vent-uncertain variant
N_REP_POWER = 200  # replicates per true shape per power-grid cell
N_BOOT_POWER = 400  # bootstrap draws per fraction, power grid
N_BATCH_POWER = 4  # geometry re-draws per power-grid cell
JITTER_M = 250.0  # jitter of replicated cores: the nugget pair separation
SIGMA_POWER = np.round(np.arange(0.1, 1.51, 0.1), 2)
NCORE_POWER = np.array([10, 15, 20, 30, 45, 62, 93, 124, 186, 248, 372, 496, 744, 992])


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


def ci(v, w=None):
    v = np.asarray(v, dtype=float)
    if w is None:
        return {
            "median": float(np.nanmedian(v)),
            "ci95": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))],
        }
    o = np.argsort(v)
    c = np.cumsum(np.asarray(w, dtype=float)[o])
    c = c / c[-1]
    q = lambda p: float(np.interp(p, c, v[o]))  # noqa: E731
    return {"median": q(0.5), "ci95": [q(0.025), q(0.975)]}


def fl(a, nd=6):
    """Nested list of floats with a fixed number of significant digits."""
    a = np.asarray(a, dtype=float)
    return json.loads(
        json.dumps(
            np.vectorize(
                lambda x: float(f"{x:.{nd}g}") if np.isfinite(x) else None, otypes=[object]
            )(a).tolist()
        )
    )


def boot_counts(rng, n, n_boot):
    idx = rng.integers(0, n, (n_boot, n))
    off = idx + (np.arange(n_boot) * n)[:, None]
    return np.bincount(off.ravel(), minlength=n_boot * n).reshape(n_boot, n).astype(float)


def c2_of_dstar(d_star, c1=C1_REF):
    r_sub = (RHO_TEPHRA - RHO_W) / RHO_W
    return (c1 * NU_SEAWATER) ** 2 / (0.75 * r_sub * G * np.asarray(d_star) ** 3)


def threshold_crossing(x, y, th):
    """First x at which y reaches th, by linear interpolation; None if never."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = y >= th
    if not ok.any():
        return None
    j = int(np.argmax(ok))
    if j == 0:
        return float(x[0])
    f = (th - y[j - 1]) / (y[j] - y[j - 1])
    return float(x[j - 1] + f * (x[j] - x[j - 1]))


class ProfileTest:
    """Profile-likelihood scoring of settling laws with a log-normal nu(Q)."""

    def __init__(self, mu, s, lqn):
        self.mu, self.s, self.lqn = mu, s, lqn
        self.nw = shape.lognormal_node_weights(lqn, mu, s)

    def tables(self, radii, w_by_fraction):
        return [
            shape.log_profile_table(r, w, self.lqn, self.mu, self.s, self.nw)
            for r, w in zip(radii, w_by_fraction, strict=True)
        ]

    def rss_by_fraction(self, tables, ys):
        return [shape.profile_rss(y, t) for y, t in zip(ys, tables, strict=True)]

    def minimise(self, rss_sum):
        return shape.refined_grid_min(rss_sum, self.s.size, self.mu.size)


def main() -> int:
    t0 = time.time()
    u = json.loads((ROOT / "results" / "nesca_unsteady.json").read_text())
    cs = json.loads((ROOT / "results" / "clast_shape.json").read_text())
    vs = json.loads((ROOT / "results" / "vent_shape.json").read_text())
    ident = json.loads((ROOT / "results" / "identifiability.json").read_text())
    strat = json.loads((ROOT / "results" / "stratification.json").read_text())

    df = load_nesca()
    keep = ~df["Location"].astype(str).str.contains("on flow", case=False, na=False).values
    xy_all = np.c_[df["x_m"].values, df["y_m"].values]
    names = [f[0] for f in FRACTIONS]
    mids = np.array([settling.sieve_midpoint(f[1], f[2]) for f in FRACTIONS])
    good, vals = [], []
    for name in names:
        v = df[f"glass_g_per_m2[{name}]"].values / 1000.0
        good.append(keep & np.isfinite(v) & (v > 0))
        vals.append(v)
    w_class = np.array([settling.settling_velocity(mids, shape=s) for s in SHAPES])
    n_buoy = strat["N_primary"]["value_s^-1"]
    pref = stem.heat_flux_constant() * 0.187 * (n_buoy**5 / EPSILON_ENTRAIN**2) ** (1 / 3)

    # ------------------------------------------------------------------ 0
    # Reproduce results/clast_shape.json with the same vent, data and seed.
    cx, cy = u["vent_xy_m"]
    r_all = np.hypot(xy_all[:, 0] - cx, xy_all[:, 1] - cy)
    radii = [r_all[g_] for g_ in good]
    obs = [v[g_] for v, g_ in zip(vals, good, strict=True)]
    L = np.array([shape.fit_length_scale(r, o) for r, o in zip(radii, obs, strict=True)])
    boot = shape.bootstrap_length_scales(radii, obs, N_BOOT_REAL, SEED_REAL)
    log_sd = np.array([float(np.nanstd(np.log(boot[:, i]))) for i in range(len(names))])
    chi2_class, q_class, _ = shape.shape_chi2(L[None, :], log_sd[None, :], w_class)
    ratio_boot = boot[:, 0] / boot[:, 3]
    ratio_obs = float(L[0] / L[3])
    repro = {
        "max_abs_diff_chi2": float(
            max(abs(chi2_class[k] - cs["by_shape"][s]["chi2"]) for k, s in enumerate(SHAPES))
        ),
        "max_rel_diff_length_scale": float(
            max(abs(L[i] / cs["length_scales_m"][n] - 1.0) for i, n in enumerate(names))
        ),
        "max_rel_diff_implied_q": float(
            max(
                abs(q_class[k] / cs["by_shape"][s]["implied_q_umb_m3s"] - 1.0)
                for k, s in enumerate(SHAPES)
            )
        ),
        "extreme_ratio_ci95": [
            float(np.nanpercentile(ratio_boot, 2.5)),
            float(np.nanpercentile(ratio_boot, 97.5)),
        ],
    }
    repro["reproduces_clast_shape_json"] = bool(
        repro["max_abs_diff_chi2"] < 1e-9
        and repro["max_rel_diff_length_scale"] < 1e-12
        and abs(repro["extreme_ratio_ci95"][0] - cs["extreme_ratio_ci95"][0]) < 1e-9
        and abs(repro["extreme_ratio_ci95"][1] - cs["extreme_ratio_ci95"][1]) < 1e-9
    )
    print(f"reproduction: {repro}", flush=True)
    if not repro["reproduces_clast_shape_json"]:
        raise SystemExit("clast_shape.json is not reproduced; stop.")

    # Cross-fraction correlation of the log residuals at shared cores, to state
    # whether independent noise per fraction is a fair synthetic assumption.
    res_mat = np.full((keep.size, len(names)), np.nan)
    for i, g_ in enumerate(good):
        a = np.column_stack([np.ones(g_.sum()), -np.pi * r_all[g_] ** 2])
        sol, *_ = np.linalg.lstsq(a, np.log(vals[i][g_]), rcond=None)
        res_mat[g_, i] = np.log(vals[i][g_]) - a @ sol
    allf = np.all(np.isfinite(res_mat), axis=1)
    corr = np.corrcoef(res_mat[allf].T)
    off = corr[~np.eye(len(names), dtype=bool)]

    # ------------------------------------------------------------------ 1
    # Continuous shape space, length-scale test.
    c1g, c2g = np.meshgrid(C1_GRID, C2_GRID, indexing="ij")
    wg = settling.settling_velocity(mids[None, None, :], c1=c1g[..., None], c2=c2g[..., None])
    chi2_g, qhat_g, _ = shape.shape_chi2(L[None, None, :], log_sd[None, None, :], wg)
    dstar_g = shape.transition_diameter(c1g, c2g)

    d_star = np.geomspace(20e-6, 3e-3, 3001)
    w1 = settling.settling_velocity(mids[None, :], c1=C1_REF, c2=c2_of_dstar(d_star)[:, None])
    chi2_1d, _, _ = shape.shape_chi2(L[None, :], log_sd[None, :], w1)
    k1 = int(np.argmin(chi2_1d))
    xx = np.log(d_star[k1 - 1 : k1 + 2])
    pp = np.polyfit(xx, chi2_1d[k1 - 1 : k1 + 2], 2)
    d_best = float(np.exp(-pp[1] / (2 * pp[0])))
    chi2_min = float(np.polyval(pp, -pp[1] / (2 * pp[0])))
    kbest = float(C1_REF**2 / c2_of_dstar(d_best))
    dchi_1d = chi2_1d - chi2_min
    dchi_g = chi2_g - chi2_min

    def interval(prof_d, prof_dchi, th):
        m = prof_dchi <= th
        dd = prof_d[m]
        kk = C1_REF**2 / c2_of_dstar(dd)
        return {
            "transition_diameter_um": [float(dd.min() * 1e6), float(dd.max() * 1e6)],
            "c1sq_over_c2": [float(kk.min()), float(kk.max())],
            "touches_profile_edge": bool(m[0] or m[-1]),
        }

    k_line = np.array([kbest, 20.0, 150.0])
    spread = []
    for kk in k_line:
        c1s = np.linspace(15, 45, 50)
        ww = settling.settling_velocity(mids[None, :], c1=c1s[:, None], c2=(c1s**2 / kk)[:, None])
        cc, qq, _ = shape.shape_chi2(L[None, :], log_sd[None, :], ww)
        spread.append(float(np.ptp(cc)))
        if kk == kbest:
            q_c1_slope = float(np.polyfit(np.log(c1s), np.log(qq), 1)[0])

    named = {}
    for k, s in enumerate(SHAPES):
        c1, c2 = settling.SHAPE_CLASSES[s]
        dc = float(chi2_class[k] - chi2_min)
        named[s] = {
            "c1": c1,
            "c2": c2,
            "c1sq_over_c2": c1**2 / c2,
            "transition_diameter_um": float(shape.transition_diameter(c1, c2) * 1e6),
            "chi2": float(chi2_class[k]),
            "delta_chi2_from_continuous_min": dc,
            "inside_2param_68": bool(dc <= 2.30),
            "inside_2param_95": bool(dc <= 6.18),
            "inside_1param_68": bool(dc <= 1.0),
            "inside_1param_95": bool(dc <= 4.0),
        }
    ig = np.unravel_index(np.argmin(chi2_g), chi2_g.shape)
    print(
        f"length-scale test, continuous: chi2_min={chi2_min:.3f} D*={d_best * 1e6:.1f} um "
        f"C1^2/C2={kbest:.1f}; degeneracy spread {spread}",
        flush=True,
    )

    # ------------------------------------------------------------------ 2
    # Heat flux over shape space, length-scale test.
    cal = {
        s: vs["by_shape"][s]["q_mass_weighted_mean_m^3s^-1"]["median"]
        / float(q_class[SHAPES.index(s)])
        for s in VOLC
    }
    f_cal = float(np.exp(np.mean(np.log(list(cal.values())))))
    phi_g = pref * (f_cal * qhat_g) ** (4 / 3)
    phi_check = {}
    for s in VOLC:
        k = SHAPES.index(s)
        a_ = float(pref * (f_cal * q_class[k]) ** (4 / 3))
        b_ = vs["by_shape"][s]["phi_at_mass_weighted_mean_W"]["median"]
        phi_check[s] = {
            "phi_from_length_scales_W": a_,
            "phi_sampled_posterior_median_W": b_,
            "relative_difference": a_ / b_ - 1.0,
        }
    c1v = np.linspace(15.0, 45.0, 61)
    wv = settling.settling_velocity(mids[None, :], c1=c1v[:, None], c2=(c1v**2 / kbest)[:, None])
    _, qv, _ = shape.shape_chi2(L[None, :], log_sd[None, :], wv)
    phi_v = pref * (f_cal * qv) ** (4 / 3)

    rel = np.concatenate(
        [
            np.asarray(vs["phi_samples_W"]["by_shape"][s])
            / np.median(vs["phi_samples_W"]["by_shape"][s])
            for s in VOLC
        ]
    )
    rng_m = np.random.default_rng(20260923)

    def marginal(dchi_map, phi_map, box):
        m = (
            (c1g >= box["c1"][0])
            & (c1g <= box["c1"][1])
            & (c2g >= box["c2"][0])
            & (c2g <= box["c2"][1])
        )
        wts = np.where(m, np.exp(-0.5 * dchi_map), 0.0)
        p = wts[m] / wts[m].sum()
        draw = rng_m.choice(p.size, size=40000, p=p)
        reg = {}
        for th, lab in ((2.30, "dchi2_2.30"), (6.18, "dchi2_6.18")):
            mm = m & (dchi_map <= th)
            reg[lab] = [float(phi_map[mm].min()), float(phi_map[mm].max())]
        c1_lo = min(settling.SHAPE_CLASSES[s][0] for s in VOLC)
        c1_hi = max(settling.SHAPE_CLASSES[s][0] for s in VOLC)
        return {
            "box": box,
            "prior": "uniform in C1 and in log C2 over the box",
            "phi_shape_only_W": ci(phi_map[m], wts[m]),
            "phi_with_within_shape_scatter_W": ci(
                phi_map[m][draw] * rng_m.choice(rel, size=draw.size)
            ),
            "phi_range_in_region_W": reg,
            "posterior_mass_c1_inside_measured_volcaniclast_range": float(
                wts[m & (c1g >= c1_lo) & (c1g <= c1_hi)].sum() / wts[m].sum()
            ),
        }

    marg_len = {
        "nominal_box": marginal(dchi_g, phi_g, BOX_NOMINAL),
        "extended_box": marginal(dchi_g, phi_g, BOX_EXTENDED),
    }

    # ------------------------------------------------------------------ 2b
    # Profile test on the real deposit: log-normal nu(Q), (mu, s) shared by the
    # fractions and profiled out, one free amplitude per fraction.
    pt_fine = ProfileTest(MU_FINE, S_FINE, LQN_FINE)
    n_obs_real = int(sum(g_.sum() for g_ in good))
    n_par_prof = 2 + len(names)

    def profile_real(vent):
        rr = np.hypot(xy_all[:, 0] - vent[0], xy_all[:, 1] - vent[1])
        rad = [rr[g_] for g_ in good]
        ys = [np.log(v[g_]) for v, g_ in zip(vals, good, strict=True)]
        by, rss_s = {}, {}
        for k, s in enumerate(SHAPES):
            tot = sum(pt_fine.rss_by_fraction(pt_fine.tables(rad, w_class[k]), ys))
            best, ks, km = pt_fine.minimise(tot)
            per_s = tot.reshape(S_FINE.size, MU_FINE.size, 1).min(axis=1)[:, 0]
            by[s] = {
                "rss": float(best[0]),
                "best_log_q_sd": float(S_FINE[ks[0]]),
                "best_q_median_m3s": float(np.exp(MU_FINE[km[0]])),
                "best_q_mass_weighted_mean_m3s": float(
                    np.exp(MU_FINE[km[0]] + 0.5 * S_FINE[ks[0]] ** 2)
                ),
            }
            rss_s[s] = per_s
        rmin = min(v["rss"] for v in by.values())
        s2 = rmin / (n_obs_real - n_par_prof)
        for s in SHAPES:
            by[s]["delta_chi2"] = (by[s]["rss"] - rmin) / s2
            by[s]["delta_chi2_by_log_q_sd"] = fl((rss_s[s] - rmin) / s2, 5)
            by[s]["delta_chi2_single_flux_minus_best"] = float((rss_s[s][0] - by[s]["rss"]) / s2)
        dv = np.array([by[s]["delta_chi2"] for s in VOLC])
        wv_ = np.exp(-0.5 * (dv - dv.min()))
        wv_ = wv_ / wv_.sum()
        return by, s2, {s: float(x) for s, x in zip(VOLC, wv_, strict=True)}, rad, ys

    prof_by, prof_s2, prof_w, rad_real, ys_real = profile_real((cx, cy))
    vent_med = (
        vs["shape_marginal"]["vent_x_m"]["median"],
        vs["shape_marginal"]["vent_y_m"]["median"],
    )
    prof_by_v, prof_s2_v, prof_w_v, _, _ = profile_real(vent_med)
    print(
        "profile test, real deposit: "
        + " ".join(
            f"{s}={prof_by[s]['delta_chi2']:.2f}(s={prof_by[s]['best_log_q_sd']})" for s in SHAPES
        ),
        flush=True,
    )

    # D* profile of the profile test.
    d_prof = np.full(D_STAR_PROFILE.size, np.nan)
    q_prof = np.full(D_STAR_PROFILE.size, np.nan)
    s_prof = np.full(D_STAR_PROFILE.size, np.nan)
    for j, dd in enumerate(D_STAR_PROFILE):
        wj = settling.settling_velocity(mids, c1=C1_REF, c2=c2_of_dstar(dd))
        tot = sum(pt_fine.rss_by_fraction(pt_fine.tables(rad_real, wj), ys_real))
        best, ks, km = pt_fine.minimise(tot)
        d_prof[j] = best[0]
        s_prof[j] = S_FINE[ks[0]]
        q_prof[j] = np.exp(MU_FINE[km[0]] + 0.5 * S_FINE[ks[0]] ** 2)
    jp = int(np.argmin(d_prof))
    rmin_d = min(float(d_prof.min()), min(v["rss"] for v in prof_by.values()))
    dchi_prof_1d = (d_prof - rmin_d) / prof_s2
    prof_cont = {
        "d_star_um": fl(D_STAR_PROFILE * 1e6),
        "delta_chi2": fl(dchi_prof_1d, 5),
        "best_log_q_sd": fl(s_prof, 3),
        "q_mass_weighted_mean_at_c1_24_m3s": fl(q_prof, 5),
        "best_transition_diameter_um": float(D_STAR_PROFILE[jp] * 1e6),
        "best_c1sq_over_c2": float(C1_REF**2 / c2_of_dstar(D_STAR_PROFILE[jp])),
        "intervals_one_parameter": {
            "dchi2_1": interval(D_STAR_PROFILE, dchi_prof_1d, 1.0),
            "dchi2_4": interval(D_STAR_PROFILE, dchi_prof_1d, 4.0),
        },
        "regions_two_parameter": {
            "dchi2_2.30": interval(D_STAR_PROFILE, dchi_prof_1d, 2.30),
            "dchi2_6.18": interval(D_STAR_PROFILE, dchi_prof_1d, 6.18),
        },
    }
    # Map onto the (C1, C2) plane: the profile test also sees D* alone, and at
    # fixed D* its flux scales as 1/C1.
    ld = np.log(dstar_g)
    dchi_prof_g = np.interp(ld, np.log(D_STAR_PROFILE), dchi_prof_1d, left=np.nan, right=np.nan)
    q_prof_g = np.exp(np.interp(ld, np.log(D_STAR_PROFILE), np.log(q_prof))) * C1_REF / c1g
    cal_p = {
        s: vs["by_shape"][s]["q_mass_weighted_mean_m^3s^-1"]["median"]
        / prof_by[s]["best_q_mass_weighted_mean_m3s"]
        for s in VOLC
    }
    f_cal_p = float(np.exp(np.mean(np.log(list(cal_p.values())))))
    phi_prof_g = pref * (f_cal_p * q_prof_g) ** (4 / 3)
    inside = np.isfinite(dchi_prof_g)
    marg_prof = {}
    for lab, box in (("nominal_box", BOX_NOMINAL), ("extended_box", BOX_EXTENDED)):
        dm = np.where(inside, dchi_prof_g, np.inf)
        marg_prof[lab] = marginal(dm, phi_prof_g, box)

    # Shape-marginalised heat flux with the profile-test weights, from the same
    # sampled per-shape posteriors as the three-class result.
    counts_w = rng_m.multinomial(40000, [prof_w[s] for s in VOLC])
    phi_mix = np.concatenate(
        [
            rng_m.choice(np.asarray(vs["phi_samples_W"]["by_shape"][s]), size=int(n), replace=True)
            for s, n in zip(VOLC, counts_w, strict=True)
        ]
    )
    counts_w_v = rng_m.multinomial(40000, [prof_w_v[s] for s in VOLC])
    phi_mix_v = np.concatenate(
        [
            rng_m.choice(np.asarray(vs["phi_samples_W"]["by_shape"][s]), size=int(n), replace=True)
            for s, n in zip(VOLC, counts_w_v, strict=True)
        ]
    )
    print(f"profile-test weights {prof_w}; Phi mixture {ci(phi_mix)}", flush=True)

    # ------------------------------------------------------------------ 5
    # Bootstrap distributions of the length-scale test.
    rb = ratio_boot[np.isfinite(ratio_boot)]
    edges = np.geomspace(1.5, 40.0, 61)
    cnt, _ = np.histogram(rb, bins=edges)
    pred_ratio = {s: float(settling.length_scale_ratio(mids[3], mids[0], shape=s)) for s in SHAPES}
    dstar_boot = np.full(boot.shape[0], np.nan)
    for b in range(boot.shape[0]):
        cc, _, _ = shape.shape_chi2(boot[b][None, :], log_sd[None, :], w1)
        dstar_boot[b] = d_star[int(np.argmin(cc))]
    dsb = ci(dstar_boot * 1e6)
    bootstrap = {
        "n_draws": int(boot.shape[0]),
        "n_draws_fine_fraction_undefined": int(np.sum(~np.isfinite(boot[:, 0]))),
        "extreme_ratio": {
            "observed": ratio_obs,
            "ci95": repro["extreme_ratio_ci95"],
            "median": float(np.median(rb)),
            "hist_edges": fl(edges),
            "hist_counts": [int(c) for c in cnt],
            "n_above_hist_range": int(np.sum(rb > edges[-1])),
            "predicted_by_shape": pred_ratio,
            "predicted_continuous_best_fit": float(np.sqrt(w1[k1, 3] / w1[k1, 0])),
            "fraction_of_draws_below_prediction": {
                s: float(np.mean(rb < v)) for s, v in pred_ratio.items()
            },
        },
        "transition_diameter_um": {
            **dsb,
            "hist_edges_um": fl(np.geomspace(20, 3000, 51)),
            "hist_counts": [
                int(c) for c in np.histogram(dstar_boot * 1e6, bins=np.geomspace(20, 3000, 51))[0]
            ],
            "fraction_below_class": {
                s: float(np.mean(dstar_boot * 1e6 < named[s]["transition_diameter_um"]))
                for s in SHAPES
            },
        },
    }

    # ------------------------------------------------------------------ 3
    # Synthetic recovery on the NESCA core geometry.
    vtrue = np.array(vent_med)
    kxy = xy_all[keep]
    kmask = np.array([g_[keep] for g_ in good])  # (4, n_keep) per-fraction validity
    n_keep = kxy.shape[0]
    p_frac = np.array([p["median"] for p in u["posterior"]["p_fractions"]])
    p_frac = p_frac / p_frac.sum()
    mass = u["posterior"]["mass_total_kg"]["median"]
    q_nodes = np.asarray(u["q_nodes"])
    nu_med = np.asarray(u["nu_median"])
    wq = nu_med / nu_med.sum()
    q_mean_nu = float(np.sum(wq * q_nodes))
    lq_sd_nu = float(np.sqrt(np.sum(wq * (np.log(q_nodes) - np.sum(wq * np.log(q_nodes))) ** 2)))
    q_nesca = u["posterior"]["q_mass_weighted_mean_m^3s^-1"]["median"]
    q_true = {
        s: q_nesca * float(q_class[SHAPES.index(s)] / q_class[SHAPES.index("blocky")])
        for s in SHAPES
    }
    sig_nug = ident["noise_model"]["sigma_log_nugget"]
    sig_res = u["posterior"]["sigma_log"]["median"]

    # Noise-free diagnostic: the ratio each true shape produces under each flux
    # distribution, from the length-scale fit on the NESCA geometry.
    rt_k = np.hypot(kxy[:, 0] - vtrue[0], kxy[:, 1] - vtrue[1])

    def deposit(r, t, fi, variant):
        w = float(w_class[SHAPES.index(t), fi])
        if variant == "steady":
            return shape.quasi_steady_deposit(r, w, [q_true[t]], [1.0], mass * p_frac[fi])
        return shape.quasi_steady_deposit(
            r, w, q_nodes * q_true[t] / q_mean_nu, nu_med, mass * p_frac[fi]
        )

    noise_free = {}
    for variant in ("steady", "nesca_nu"):
        noise_free[variant] = {}
        for t in SHAPES:
            Ls = np.array(
                [
                    shape.fit_length_scale(
                        rt_k[kmask[fi]], deposit(rt_k[kmask[fi]], t, fi, variant)
                    )
                    for fi in range(4)
                ]
            )
            c_, _, _ = shape.shape_chi2(Ls[None, :], log_sd[None, :], w_class)
            noise_free[variant][t] = {
                "length_scales_m": Ls.tolist(),
                "extreme_ratio": float(Ls[0] / Ls[3]),
                "length_scale_test_chi2": dict(zip(SHAPES, c_.tolist(), strict=True)),
                "length_scale_test_pick": SHAPES[int(np.argmin(c_))],
            }

    def simulate(
        rng,
        xy,
        mask,
        truths,
        sigma,
        n_rep,
        n_boot,
        variant,
        cands,
        pt=None,
        fit_vent=None,
        subsets=None,
        rho=0.0,
    ):
        """One batch.  Returns L, log sd (n_truth, n_rep, 4) and profile-test RSS minima
        {subset: (n_truth, n_rep, n_cand)} plus the observation counts per subset."""
        rt = np.hypot(xy[:, 0] - vtrue[0], xy[:, 1] - vtrue[1])
        rf = rt if fit_vent is None else np.hypot(xy[:, 0] - fit_vent[0], xy[:, 1] - fit_vent[1])
        L_s = np.full((len(truths), n_rep, 4), np.nan)
        sd_s = np.full((len(truths), n_rep, 4), np.nan)
        ys = []
        # rho > 0: a per-core noise component shared by the fractions of one core
        z_core = rng.standard_normal((xy.shape[0], len(truths) * n_rep)) if rho > 0 else None
        for fi in range(4):
            m = mask[fi]
            counts = boot_counts(rng, int(m.sum()), n_boot)
            y = np.concatenate(
                [
                    np.log(deposit(rt[m], t, fi, variant))[:, None]
                    + sigma * np.sqrt(1.0 - rho) * rng.standard_normal((m.sum(), n_rep))
                    for t in truths
                ],
                axis=1,
            )
            if z_core is not None:
                y = y + sigma * np.sqrt(rho) * z_core[m]
            ys.append(y)
            Lf, sdf = shape.bootstrap_log_length_sd(rf[m], y, counts)
            L_s[:, :, fi] = Lf.reshape(len(truths), n_rep)
            sd_s[:, :, fi] = sdf.reshape(len(truths), n_rep)
        prof = None
        if pt is not None:
            subsets = subsets or [tuple(range(4))]
            prof = {sub: np.full((len(truths), n_rep, len(cands)), np.nan) for sub in subsets}
            rad = [rf[mask[fi]] for fi in range(4)]
            for c, cand in enumerate(cands):
                rss_f = pt.rss_by_fraction(pt.tables(rad, w_class[SHAPES.index(cand)]), ys)
                for sub in subsets:
                    best, _, _ = pt.minimise(sum(rss_f[i] for i in sub))
                    prof[sub][:, :, c] = best.reshape(len(truths), n_rep)
        n_obs = {
            sub: int(sum(mask[i].sum() for i in sub)) for sub in (subsets or [tuple(range(4))])
        }
        return L_s, sd_s, prof, n_obs

    def length_chi2(L_s, sd_s, cands, sub=None):
        sub = list(range(4)) if sub is None else list(sub)
        ci_ = [SHAPES.index(c) for c in cands]
        c2_, _, _ = shape.shape_chi2(
            L_s[..., None, sub], sd_s[..., None, sub], w_class[ci_][None, None, :, :][..., sub]
        )
        return c2_

    def profile_chi2(rss, n_obs):
        s2 = rss.min(axis=-1, keepdims=True) / (n_obs - n_par_prof)
        return (rss - rss.min(axis=-1, keepdims=True)) / s2

    def score(chi2, truths, cands, rng):
        pick = shape.classify_by_chi2(chi2, rng)
        conf = np.zeros((len(truths), len(cands)), dtype=int)
        dch = np.full(chi2.shape[:2], np.nan)
        for a, t in enumerate(truths):
            conf[a] = np.bincount(pick[a], minlength=len(cands))
            j = cands.index(t)
            dch[a] = np.delete(chi2[a], j, axis=-1).min(axis=-1) - chi2[a, :, j]
        # an undefined chi2 (no usable length scale) carries no preference
        dch = np.where(np.isfinite(dch), dch, 0.0)
        return conf, dch

    def summarise(chi2, truths, cands, rng, hist=True):
        conf, dch = score(chi2, truths, cands, rng)
        frac = conf / conf.sum(axis=1, keepdims=True)
        out = {
            "truths": truths,
            "candidates": cands,
            "confusion_counts": conf.tolist(),
            "confusion_fraction": fl(frac, 4),
            "p_correct_by_true_shape": {
                t: float(frac[a, cands.index(t)]) for a, t in enumerate(truths)
            },
            "p_correct_mean": float(
                np.mean([frac[a, cands.index(t)] for a, t in enumerate(truths)])
            ),
            "p_correct_mean_volcaniclast": float(
                np.mean([frac[a, cands.index(t)] for a, t in enumerate(truths) if t in VOLC])
            ),
            "delta_chi2_true_vs_runner_up": {
                t: {
                    "median": float(np.median(dch[a])),
                    "q05": float(np.percentile(dch[a], 5)),
                    "q25": float(np.percentile(dch[a], 25)),
                    "q75": float(np.percentile(dch[a], 75)),
                    "q95": float(np.percentile(dch[a], 95)),
                    "fraction_above_6": float(np.mean(dch[a] > 6.0)),
                    "fraction_above_9": float(np.mean(dch[a] > 9.0)),
                }
                for a, t in enumerate(truths)
            },
        }
        if hist:
            e = np.linspace(-20, 40, 61)
            out["delta_chi2_hist_edges"] = fl(e)
            out["delta_chi2_hist_counts"] = {
                t: [int(c) for c in np.histogram(np.clip(dch[a], -20, 40), bins=e)[0]]
                for a, t in enumerate(truths)
            }
        return out

    rng = np.random.default_rng(20260923)
    main_res = {}
    full = (0, 1, 2, 3)
    for variant in ("nesca_nu", "steady"):
        main_res[variant] = {}
        for sig, lab in ((sig_res, "sigma_residual"), (sig_nug, "sigma_nugget")):
            L_s, sd_s, prof, n_obs = simulate(
                rng,
                kxy,
                kmask,
                SHAPES,
                sig,
                N_REP_MAIN,
                N_BOOT_SYN_MAIN,
                variant,
                SHAPES,
                pt=pt_fine,
            )
            c_len = length_chi2(L_s, sd_s, SHAPES)
            c_pro = profile_chi2(prof[full], n_obs[full])
            main_res[variant][lab] = {
                "sigma_log": float(sig),
                "length_scale_test": {
                    "five_candidates": summarise(c_len, SHAPES, SHAPES, rng),
                    "volcaniclast_candidates": summarise(
                        length_chi2(L_s[2:], sd_s[2:], VOLC), VOLC, VOLC, rng, hist=False
                    ),
                    "mean_chi2_of_true_class": {
                        t: float(np.nanmean(c_len[a, :, a])) for a, t in enumerate(SHAPES)
                    },
                    "fraction_replicates_with_undefined_length_scale": float(
                        np.mean(np.any(~np.isfinite(L_s), axis=-1))
                    ),
                },
                "profile_test": {
                    "five_candidates": summarise(c_pro, SHAPES, SHAPES, rng),
                    "volcaniclast_candidates": summarise(
                        profile_chi2(prof[full][2:, :, 2:], n_obs[full]),
                        VOLC,
                        VOLC,
                        rng,
                        hist=False,
                    ),
                },
            }
            pl = main_res[variant][lab]["length_scale_test"]["five_candidates"][
                "p_correct_by_true_shape"
            ]
            pp_ = main_res[variant][lab]["profile_test"]["five_candidates"][
                "p_correct_by_true_shape"
            ]
            print(
                f"  {variant:8s} {lab:15s} length-scale "
                + " ".join(f"{v:.2f}" for v in pl.values())
                + " | profile "
                + " ".join(f"{v:.2f}" for v in pp_.values())
                + f"  ({time.time() - t0:.0f} s)",
                flush=True,
            )

    # Noise shared by the fractions of one core, at the mean cross-fraction
    # correlation of the real residuals.
    rho_obs = float(off.mean())
    L_s, sd_s, prof, n_obs = simulate(
        rng,
        kxy,
        kmask,
        SHAPES,
        sig_res,
        N_REP_MAIN,
        N_BOOT_SYN_MAIN,
        "nesca_nu",
        SHAPES,
        pt=pt_fine,
        rho=rho_obs,
    )
    main_res["nesca_nu_correlated_noise"] = {
        "sigma_residual": {
            "sigma_log": float(sig_res),
            "cross_fraction_correlation": rho_obs,
            "length_scale_test": {
                "five_candidates": summarise(length_chi2(L_s, sd_s, SHAPES), SHAPES, SHAPES, rng),
                "volcaniclast_candidates": summarise(
                    length_chi2(L_s[2:], sd_s[2:], VOLC), VOLC, VOLC, rng, hist=False
                ),
            },
            "profile_test": {
                "five_candidates": summarise(
                    profile_chi2(prof[full], n_obs[full]), SHAPES, SHAPES, rng
                ),
                "volcaniclast_candidates": summarise(
                    profile_chi2(prof[full][2:, :, 2:], n_obs[full]), VOLC, VOLC, rng, hist=False
                ),
            },
        }
    }
    pp_ = main_res["nesca_nu_correlated_noise"]["sigma_residual"]["profile_test"]["five_candidates"]
    print(f"  correlated-noise profile P(correct) {pp_['p_correct_by_true_shape']}", flush=True)

    # Vent uncertainty: the refit assumes a vent drawn from its posterior.
    vx = np.asarray(vs["vent_posterior_samples"]["x_m"])
    vy = np.asarray(vs["vent_posterior_samples"]["y_m"])
    cl, cp = [], []
    for _ in range(N_VENT_DRAWS):
        k = rng.integers(0, vx.size)
        L_s, sd_s, prof, n_obs = simulate(
            rng,
            kxy,
            kmask,
            SHAPES,
            sig_res,
            N_REP_MAIN // N_VENT_DRAWS,
            N_BOOT_SYN_MAIN,
            "nesca_nu",
            SHAPES,
            pt=pt_fine,
            fit_vent=(vx[k], vy[k]),
        )
        cl.append(length_chi2(L_s, sd_s, SHAPES))
        cp.append(profile_chi2(prof[full], n_obs[full]))
    main_res["nesca_nu_vent_uncertain"] = {
        "sigma_residual": {
            "sigma_log": float(sig_res),
            "length_scale_test": {
                "five_candidates": summarise(np.concatenate(cl, axis=1), SHAPES, SHAPES, rng)
            },
            "profile_test": {
                "five_candidates": summarise(np.concatenate(cp, axis=1), SHAPES, SHAPES, rng)
            },
        }
    }
    pp_ = main_res["nesca_nu_vent_uncertain"]["sigma_residual"]["profile_test"]["five_candidates"]
    print(
        f"  vent-uncertain profile P(correct) {pp_['p_correct_by_true_shape']}  "
        f"({time.time() - t0:.0f} s)",
        flush=True,
    )

    # ------------------------------------------------------------------ 4
    # Power: noise level x number of cores (four fractions), and noise level x
    # number of fractions at the NESCA geometry from the same replicates.
    pt_pow = ProfileTest(MU_POW, S_POW, LQN_POW)
    subsets = [sub for k in range(1, 5) for sub in itertools.combinations(range(4), k)]
    shp = (SIGMA_POWER.size, NCORE_POWER.size)
    grids = {
        key: np.full(shp, np.nan)
        for key in (
            "p_len_nu",
            "p_len_steady",
            "p_prof_nu",
            "d_len_nu",
            "d_len_steady",
            "d_prof_nu",
        )
    }
    grids_by = {s: np.full(shp, np.nan) for s in VOLC}
    frac_grids = {
        key: np.full((SIGMA_POWER.size, 4), np.nan)
        for key in ("p_len", "p_prof", "d_len", "d_prof", "p_len_steady", "d_len_steady")
    }
    p_subset = {}
    j124 = int(np.where(NCORE_POWER == n_keep)[0][0])
    for a, sig in enumerate(SIGMA_POWER):
        for b, nc in enumerate(NCORE_POWER):
            acc = {"L": [], "sd": [], "Ls": [], "sds": [], "prof": {sub: [] for sub in subsets}}
            nob = None
            for _ in range(N_BATCH_POWER):
                if nc <= n_keep:
                    sel = (
                        np.sort(rng.choice(n_keep, size=nc, replace=False))
                        if nc < n_keep
                        else np.arange(n_keep)
                    )
                    xy_b, m_b = kxy[sel], kmask[:, sel]
                else:
                    extra = rng.integers(0, n_keep, nc - n_keep)
                    xy_b = np.vstack(
                        [kxy, kxy[extra] + JITTER_M * rng.standard_normal((extra.size, 2))]
                    )
                    m_b = np.concatenate([kmask, kmask[:, extra]], axis=1)
                subs_here = subsets if nc == n_keep else [full]
                L_s, sd_s, prof, n_obs = simulate(
                    rng,
                    xy_b,
                    m_b,
                    VOLC,
                    float(sig),
                    N_REP_POWER // N_BATCH_POWER,
                    N_BOOT_POWER,
                    "nesca_nu",
                    VOLC,
                    pt=pt_pow,
                    subsets=subs_here,
                )
                L2, sd2, _, _ = simulate(
                    rng,
                    xy_b,
                    m_b,
                    VOLC,
                    float(sig),
                    N_REP_POWER // N_BATCH_POWER,
                    N_BOOT_POWER,
                    "steady",
                    VOLC,
                )
                acc["L"].append(L_s)
                acc["sd"].append(sd_s)
                acc["Ls"].append(L2)
                acc["sds"].append(sd2)
                for sub in subs_here:
                    acc["prof"][sub].append(profile_chi2(prof[sub], n_obs[sub]))
                nob = n_obs
            Lc = np.concatenate(acc["L"], axis=1)
            sdc = np.concatenate(acc["sd"], axis=1)
            for key_p, key_d, chi in (
                ("p_len_nu", "d_len_nu", length_chi2(Lc, sdc, VOLC)),
                (
                    "p_len_steady",
                    "d_len_steady",
                    length_chi2(
                        np.concatenate(acc["Ls"], axis=1), np.concatenate(acc["sds"], axis=1), VOLC
                    ),
                ),
                ("p_prof_nu", "d_prof_nu", np.concatenate(acc["prof"][full], axis=1)),
            ):
                conf, dch = score(chi, VOLC, VOLC, rng)
                fr = conf / conf.sum(axis=1, keepdims=True)
                grids[key_p][a, b] = float(np.mean(np.diag(fr)))
                grids[key_d][a, b] = float(np.median(dch))
                if key_p == "p_prof_nu":
                    for j, s in enumerate(VOLC):
                        grids_by[s][a, b] = float(fr[j, j])
            if nc == n_keep:
                Lst = np.concatenate(acc["Ls"], axis=1)
                sdst = np.concatenate(acc["sds"], axis=1)
                for k_ in range(1, 5):
                    pl_, pp2, dl_, dp_, pls, dls = [], [], [], [], [], []
                    for sub in [x for x in subsets if len(x) == k_]:
                        conf, dch = score(length_chi2(Lc, sdc, VOLC, sub), VOLC, VOLC, rng)
                        fr = conf / conf.sum(axis=1, keepdims=True)
                        pl_.append(float(np.mean(np.diag(fr))))
                        dl_.append(float(np.median(dch)))
                        conf, dch = score(length_chi2(Lst, sdst, VOLC, sub), VOLC, VOLC, rng)
                        fr = conf / conf.sum(axis=1, keepdims=True)
                        pls.append(float(np.mean(np.diag(fr))))
                        dls.append(float(np.median(dch)))
                        if k_ == 1:
                            # One fraction: the kernel sees w/Q alone, every law fits
                            # identically and the profile test is at chance.
                            pp2.append(1.0 / len(VOLC))
                            dp_.append(0.0)
                        else:
                            conf, dch = score(
                                np.concatenate(acc["prof"][sub], axis=1), VOLC, VOLC, rng
                            )
                            fr = conf / conf.sum(axis=1, keepdims=True)
                            pp2.append(float(np.mean(np.diag(fr))))
                            dp_.append(float(np.median(dch)))
                        if abs(sig - 1.0) < 1e-9 or abs(sig - 0.8) < 1e-9:
                            p_subset.setdefault(f"sigma_{sig:.1f}", {})[
                                "+".join(names[i] for i in sub)
                            ] = {"length_scale_test": pl_[-1], "profile_test": pp2[-1]}
                    frac_grids["p_len"][a, k_ - 1] = float(np.mean(pl_))
                    frac_grids["p_prof"][a, k_ - 1] = float(np.mean(pp2))
                    frac_grids["d_len"][a, k_ - 1] = float(np.mean(dl_))
                    frac_grids["d_prof"][a, k_ - 1] = float(np.mean(dp_))
                    frac_grids["p_len_steady"][a, k_ - 1] = float(np.mean(pls))
                    frac_grids["d_len_steady"][a, k_ - 1] = float(np.mean(dls))
            del nob
        print(
            f"  power sigma={sig:.1f}: profile "
            + " ".join(f"{v:.2f}" for v in grids["p_prof_nu"][a])
            + " | len-steady "
            + " ".join(f"{v:.2f}" for v in grids["p_len_steady"][a])
            + f"  ({time.time() - t0:.0f} s)",
            flush=True,
        )

    def decisive_block(pg, dg):
        out = {}
        for sig, lab in ((sig_res, "sigma_residual"), (sig_nug, "sigma_nugget")):
            pc = np.array([np.interp(sig, SIGMA_POWER, pg[:, b]) for b in range(NCORE_POWER.size)])
            dc = np.array([np.interp(sig, SIGMA_POWER, dg[:, b]) for b in range(NCORE_POWER.size)])
            o = {
                "sigma_log": float(sig),
                "p_correct_at_124_cores": float(pc[j124]),
                "median_delta_chi2_at_124_cores": float(dc[j124]),
            }
            for key, arr, th in (
                ("n_cores_for_p_correct_0.95", pc, 0.95),
                ("n_cores_for_median_dchi2_6", dc, 6.0),
                ("n_cores_for_median_dchi2_9", dc, 9.0),
            ):
                x = threshold_crossing(np.log(NCORE_POWER), arr, th)
                o[key] = None if x is None else float(np.exp(x))
            out[lab] = o
        col, dcol = pg[:, j124], dg[:, j124]
        out["nesca_geometry_124_cores"] = {
            "max_sigma_for_p_correct_0.95": threshold_crossing(SIGMA_POWER[::-1], col[::-1], 0.95),
            "max_sigma_for_median_dchi2_6": threshold_crossing(SIGMA_POWER[::-1], dcol[::-1], 6.0),
            "max_sigma_for_median_dchi2_9": threshold_crossing(SIGMA_POWER[::-1], dcol[::-1], 9.0),
        }
        return out

    def collapse(pg):
        """P(correct) against the information variable n_cores / sigma^2.

        Fits P = 1/3 + (2/3) / (1 + exp(-(ln x - a) / b)) by least squares over every
        cell of the grid and returns the x at which the fit reaches 0.95 and 0.80.
        """
        from scipy.optimize import curve_fit

        x = (NCORE_POWER[None, :] / SIGMA_POWER[:, None] ** 2).ravel()
        y = np.asarray(pg, dtype=float).ravel()

        def model(lx, a_, b_):
            return 1 / 3 + (2 / 3) / (1 + np.exp(-(lx - a_) / b_))

        (a_, b_), _ = curve_fit(model, np.log(x), y, p0=[np.log(1e3), 1.0])
        resid = y - model(np.log(x), a_, b_)

        def x_at(pv):
            return float(np.exp(a_ - b_ * np.log((2 / 3) / (pv - 1 / 3) - 1)))

        sr, sn = sig_res, sig_nug
        return {
            "fit": "P = 1/3 + (2/3)/(1 + exp(-(ln x - a)/b)), x = n_cores / sigma_log^2",
            "a": float(a_),
            "b": float(b_),
            "rms_residual": float(np.sqrt(np.mean(resid**2))),
            "max_abs_residual": float(np.max(np.abs(resid))),
            "x_range_simulated": [float(x.min()), float(x.max())],
            "x_for_p_0.95": x_at(0.95),
            "x_for_p_0.80": x_at(0.80),
            "n_cores_for_p_0.95_at_sigma_residual": x_at(0.95) * sr**2,
            "n_cores_for_p_0.95_at_sigma_nugget": x_at(0.95) * sn**2,
            "n_cores_for_p_0.80_at_sigma_residual": x_at(0.80) * sr**2,
            "sigma_for_p_0.95_at_124_cores": float(np.sqrt(n_keep / x_at(0.95))),
            "p_at_nesca_residual": float(model(np.log(n_keep / sr**2), a_, b_)),
            "p_at_nesca_nugget": float(model(np.log(n_keep / sn**2), a_, b_)),
        }

    power = {
        "truths": VOLC,
        "candidates": VOLC,
        "replicates_per_true_shape_per_cell": N_REP_POWER,
        "bootstrap_draws": N_BOOT_POWER,
        "profile_grid": {
            "mu_range_m3s": [float(np.exp(MU_POW[0])), float(np.exp(MU_POW[-1]))],
            "n_mu": int(MU_POW.size),
            "log_q_sd": fl(S_POW, 4),
        },
        "sigma_log": SIGMA_POWER.tolist(),
        "n_cores": NCORE_POWER.tolist(),
        "core_geometry": (
            "n <= 124: random subsets of the 124 off-flow NESCA cores, "
            "redrawn four times per cell; n > 124: the 124 cores plus "
            f"copies of randomly chosen cores jittered by N(0, {JITTER_M:.0f} m) "
            "in each coordinate, the separation within which the nugget "
            "was estimated"
        ),
        "profile_test_nesca_nu": {
            "p_correct": fl(grids["p_prof_nu"], 4),
            "p_correct_by_true_shape": {s: fl(v, 4) for s, v in grids_by.items()},
            "median_delta_chi2": fl(grids["d_prof_nu"], 4),
            "decisive": decisive_block(grids["p_prof_nu"], grids["d_prof_nu"]),
            "information_collapse": collapse(grids["p_prof_nu"]),
        },
        "length_scale_test_nesca_nu": {
            "p_correct": fl(grids["p_len_nu"], 4),
            "median_delta_chi2": fl(grids["d_len_nu"], 4),
            "decisive": decisive_block(grids["p_len_nu"], grids["d_len_nu"]),
        },
        "length_scale_test_steady": {
            "p_correct": fl(grids["p_len_steady"], 4),
            "median_delta_chi2": fl(grids["d_len_steady"], 4),
            "decisive": decisive_block(grids["p_len_steady"], grids["d_len_steady"]),
            "information_collapse": collapse(grids["p_len_steady"]),
        },
        "n_fractions": [1, 2, 3, 4],
        "by_n_fractions_nesca_nu": {
            "length_scale_test_p_correct": fl(frac_grids["p_len"], 4),
            "profile_test_p_correct": fl(frac_grids["p_prof"], 4),
            "length_scale_test_median_delta_chi2": fl(frac_grids["d_len"], 4),
            "profile_test_median_delta_chi2": fl(frac_grids["d_prof"], 4),
            "length_scale_test_steady_p_correct": fl(frac_grids["p_len_steady"], 4),
            "length_scale_test_steady_median_delta_chi2": fl(frac_grids["d_len_steady"], 4),
            "note": (
                "At the 124-core NESCA geometry; each entry averages over every subset "
                "of that many sieve fractions.  Deposits from the NESCA flux distribution, "
                "except the length_scale_test_steady entries, from a single flux.  With one "
                "fraction the kernel depends on w/Q alone, every law fits identically and "
                "P(correct) is chance, 1/3."
            ),
        },
        "p_correct_by_fraction_subset": p_subset,
    }

    res = {
        "_description": (
            "Clast shape over the continuous Ferguson and Church plane, the "
            "heat flux across it, synthetic recovery of the shape class on "
            "the NESCA core geometry by two classifiers, and the power of the test."
        ),
        "_generated": datetime.date.today().isoformat(),
        "_script": "experiments/shape_space/run.py",
        "_git_commit": git_hash(),
        "_inputs": [
            "results/clast_shape.json",
            "results/vent_shape.json",
            "results/nesca_unsteady.json",
            "results/identifiability.json",
            "results/stratification.json",
        ],
        "_classifiers": {
            "length_scale_test": (
                "per-fraction L by log-linear least squares, bootstrap sd of "
                "log L over cores, one free flux per law, chi2 over the four "
                "L_i on 3 dof; the construction of results/clast_shape.json"
            ),
            "profile_test": (
                "quasi-steady kernel with a log-normal nu(Q), log-mean and log-sd "
                "shared by all fractions and profiled on a grid with parabolic "
                "refinement, one free amplitude per fraction; delta chi2 = delta "
                "RSS / sigma_hat^2 with sigma_hat^2 = min RSS / (N - 6)"
            ),
        },
        "reproduction_of_clast_shape": repro,
        "real_data": {
            "vent_xy_m": [cx, cy],
            "n_cores": int(keep.sum()),
            "n_observations": n_obs_real,
            "fractions": names,
            "sieve_windows_m": [[f[1], f[2]] for f in FRACTIONS],
            "sieve_midpoints_m": mids.tolist(),
            "length_scales_m": L.tolist(),
            "length_scale_ci95_m": [
                [
                    float(np.nanpercentile(boot[:, i], 2.5)),
                    float(np.nanpercentile(boot[:, i], 97.5)),
                ]
                for i in range(4)
            ],
            "log_length_scale_sd": log_sd.tolist(),
            "residual_correlation_across_fractions": {
                "matrix": fl(corr, 4),
                "mean_off_diagonal": float(off.mean()),
                "n_cores_with_all_fractions": int(allf.sum()),
            },
        },
        "settling_curves": {
            "diameter_m": fl(np.geomspace(30e-6, 2e-3, 120)),
            "w_ms": {
                s: fl(settling.settling_velocity(np.geomspace(30e-6, 2e-3, 120), shape=s))
                for s in SHAPES
            },
            "w_ms_at_midpoints": {s: w_class[k].tolist() for k, s in enumerate(SHAPES)},
            "q_hat_m3s": {s: float(q_class[k]) for k, s in enumerate(SHAPES)},
            "transition_diameter_um": {s: named[s]["transition_diameter_um"] for s in SHAPES},
            "length_scale_test_best_fit": {
                "w_ms_at_c1_24": fl(
                    settling.settling_velocity(
                        np.geomspace(30e-6, 2e-3, 120), c1=C1_REF, c2=c2_of_dstar(d_best)
                    )
                ),
                "q_hat_at_c1_24_m3s": float(
                    shape.shape_chi2(
                        L,
                        log_sd,
                        settling.settling_velocity(mids, c1=C1_REF, c2=c2_of_dstar(d_best)),
                    )[1]
                ),
            },
        },
        "length_scale_test": {
            "continuous": {
                "c1": fl(C1_GRID),
                "c2": fl(C2_GRID),
                "grid_note": (
                    "C1 linear, C2 logarithmic.  The nominal box C2 0.3-30 was "
                    "extended to 100 because the best-fit valley leaves it at C2 = 30."
                ),
                "chi2": fl(chi2_g, 5),
                "delta_chi2": fl(dchi_g, 5),
                "phi_W": fl(phi_g, 5),
                "dof": 2,
                "best_fit": {
                    "chi2_min": chi2_min,
                    "transition_diameter_um": d_best * 1e6,
                    "c1sq_over_c2": kbest,
                    "grid_minimum": {
                        "c1": float(C1_GRID[ig[0]]),
                        "c2": float(C2_GRID[ig[1]]),
                        "chi2": float(chi2_g[ig]),
                    },
                    "note": (
                        "chi2 is constant along C2 = C1^2 / K, so the best fit is the line "
                        "K = c1sq_over_c2; the grid minimum is one point of it."
                    ),
                },
                "degeneracy": {
                    "constrained_combination": "C1^2/C2, equivalently the transition diameter D*",
                    "direction_dlog10C2_dlog10C1": 2.0,
                    "chi2_range_along_line_numerical": dict(
                        zip([f"K={k:.1f}" for k in k_line], spread, strict=True)
                    ),
                    "dlogQ_dlogC1_along_best_line": q_c1_slope,
                    "dlogPhi_dlogC1_along_best_line": q_c1_slope * 4 / 3,
                },
                "intervals_one_parameter": {
                    "dchi2_1": interval(d_star, dchi_1d, 1.0),
                    "dchi2_4": interval(d_star, dchi_1d, 4.0),
                },
                "regions_two_parameter": {
                    "dchi2_2.30": interval(d_star, dchi_1d, 2.30),
                    "dchi2_6.18": interval(d_star, dchi_1d, 6.18),
                },
                "d_star_profile": {
                    "d_star_um": fl(d_star[::10] * 1e6),
                    "delta_chi2": fl(dchi_1d[::10], 5),
                },
                "named_classes": named,
            },
            "heat_flux": {
                "q_conversion": (
                    "<Q> = f_cal * Q_hat(C1, C2), with Q_hat the flux profiled from "
                    "the four length scales, L_i^2 = Q/w_i, each fraction's own w_i "
                    "converting its L_i, combined with inverse-variance weights in "
                    "log L; f_cal is the ratio of the sampled posterior median of the "
                    "mass-weighted mean flux to Q_hat, averaged geometrically over "
                    "the three measured volcaniclast classes."
                ),
                "f_cal": f_cal,
                "f_cal_by_class": cal,
                "phi_check_by_class": phi_check,
                "valley": {
                    "c1sq_over_c2": kbest,
                    "c1": fl(c1v),
                    "phi_W": fl(phi_v),
                    "phi_at_c1_15_W": float(phi_v[0]),
                    "phi_at_c1_45_W": float(phi_v[-1]),
                },
                "marginal_continuous": marg_len,
            },
        },
        "profile_test": {
            "real_deposit": {
                "vent_xy_m": [cx, cy],
                "sigma_hat": float(np.sqrt(prof_s2)),
                "n_parameters": n_par_prof,
                "log_q_sd_grid": fl(S_FINE, 4),
                "by_shape": prof_by,
                "weights_volcaniclast": prof_w,
                "best_shape": min(prof_by, key=lambda s: prof_by[s]["delta_chi2"]),
                "worst_volcaniclast_shape": max(VOLC, key=lambda s: prof_by[s]["delta_chi2"]),
            },
            "real_deposit_vent_sampled_median": {
                "vent_xy_m": list(vent_med),
                "sigma_hat": float(np.sqrt(prof_s2_v)),
                "by_shape": prof_by_v,
                "weights_volcaniclast": prof_w_v,
                "best_shape": min(prof_by_v, key=lambda s: prof_by_v[s]["delta_chi2"]),
            },
            "continuous": {
                **prof_cont,
                "delta_chi2_on_grid": fl(dchi_prof_g, 5),
                "phi_W_on_grid": fl(phi_prof_g, 5),
                "f_cal": f_cal_p,
                "f_cal_by_class": cal_p,
                "marginal_continuous": marg_prof,
            },
            "heat_flux_three_class_mixture": {
                "weights": prof_w,
                "phi_W": ci(phi_mix),
                "phi_W_vent_sampled_median": ci(phi_mix_v),
                "weights_vent_sampled_median": prof_w_v,
                "per_shape_samples": "results/vent_shape.json#phi_samples_W.by_shape",
                "reference_length_scale_test": {
                    "weights": vs["shape_weights"],
                    "phi_W": vs["shape_marginal"]["phi_W"],
                },
            },
        },
        "bootstrap": bootstrap,
        "synthetic": {
            "setup": {
                "vent_xy_m": vtrue.tolist(),
                "vent_source": "results/vent_shape.json#shape_marginal (posterior median)",
                "n_cores": int(n_keep),
                "n_cores_by_fraction": [int(m.sum()) for m in kmask],
                "mass_kg": mass,
                "p_fractions": p_frac.tolist(),
                "q_mean_true_m3s": q_true,
                "q_true_rule": (
                    "NESCA posterior median <Q> under blocky settling, scaled by each "
                    "class's profiled flux relative to blocky, so every true shape "
                    "reproduces the observed spatial extent"
                ),
                "flux_distribution": {
                    "nesca_nu": (
                        "posterior-median nu(Q) of results/nesca_unsteady.json, relabelled "
                        "to <Q>; its mass-weighted log-Q sd is "
                        f"{lq_sd_nu:.3f}"
                    ),
                    "steady": "a single flux <Q>",
                },
                "nesca_nu_log_q_sd": lq_sd_nu,
                "noise": (
                    "log-normal, independent per core and fraction; the variant "
                    "nesca_nu_correlated_noise shares a per-core component across fractions "
                    "at the mean cross-fraction correlation of the real residuals"
                ),
                "sigma_residual": sig_res,
                "sigma_nugget": sig_nug,
                "replicates_per_true_shape": N_REP_MAIN,
                "bootstrap_draws": N_BOOT_SYN_MAIN,
                "vent_uncertain_variant": (
                    f"{N_VENT_DRAWS} vents drawn from "
                    "results/vent_shape.json#vent_posterior_samples, "
                    f"{N_REP_MAIN // N_VENT_DRAWS} replicates each"
                ),
            },
            "noise_free": noise_free,
            "main": main_res,
            "power": power,
        },
    }
    OUT.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
