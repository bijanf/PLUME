"""Figure: the NESCA inversion.

(a) the deposit and the posterior predictive band, per sieve fraction
(b) the posterior nu(Q)
(c) heat flux of the steady single-fraction fit against clast shape
(d) the steady single-fraction dispersal length against the reference value
(e) the joint posterior of the two calibrated modes, total mass and <Q>
(f) the posterior predictive deposit in plan view, one map per sieve fraction
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from plume_style import OKABE_ITO, TWO_COL, apply_style  # noqa: E402

apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from plume_inv.data import FRACTIONS, load_nesca  # noqa: E402

S = json.loads((ROOT / "results" / "nesca_steady.json").read_text())
M = json.loads((ROOT / "results" / "nesca_maps.json").read_text())
COLS = [OKABE_ITO["sky"], OKABE_ITO["green"], OKABE_ITO["blue"], OKABE_ITO["vermillion"]]
FRAC_LABEL = {
    "63-125um": "63–125 µm",
    "125-250um": "125–250 µm",
    "250-500um": "250–500 µm",
    ">500um": ">500 µm",
}
MAP_CMAP = "viridis"
MAP_LEVELS = np.arange(-4.0, 0.51, 0.25)


def load_cores():
    df = load_nesca()
    keep = ~df["Location"].astype(str).str.contains("on flow", case=False, na=False).values
    out = {}
    for name, *_ in FRACTIONS:
        v = df[f"glass_g_per_m2[{name}]"].values / 1000.0
        good = keep & np.isfinite(v) & (v > 0)
        out[name] = (df["x_m"].values[good], df["y_m"].values[good], v[good])
    return out


def panel_a(a, cores, vent):
    rad = M["posterior_predictive_radial"]
    rr = np.array(rad["r_m"]) / 1e3
    for i, (name, *_r) in enumerate(FRACTIONS):
        x, y, v = cores[name]
        r = np.hypot(x - vent[0], y - vent[1]) / 1e3
        a.plot(r, v, "o", ms=1.6, mew=0, color=COLS[i], alpha=0.6)
        b = rad["by_class"][name]
        a.fill_between(rr, b["lo95"], b["hi95"], color=COLS[i], alpha=0.16, lw=0)
        a.plot(rr, b["median"], "-", lw=1.1, color=COLS[i], label=FRAC_LABEL[name])
    a.set_yscale("log")
    a.set_xlim(0, 9)
    a.set_ylim(1e-4, 5)
    a.set_xlabel("radius from vent (km)")
    a.set_ylabel(r"deposit $\Omega$ (kg m$^{-2}$)")
    a.legend(
        handlelength=1.3,
        fontsize=5.5,
        ncol=1,
        loc="upper right",
        borderaxespad=0.2,
        labelspacing=0.3,
    )


def panel_b(b):
    q = np.array(M["q_nodes"])
    nu_med = np.array(M["nu_median"]) / 1e6
    lo, hi = (np.array(v) / 1e6 for v in M["nu_ci95"])
    col = OKABE_ITO["purple"]
    b.fill_between(q, lo, hi, color=col, alpha=0.25, lw=0, label="95% credible")
    b.plot(q, nu_med, "-", lw=1.3, color=col, label="posterior median")
    qm = M["posterior"]["q_mass_weighted_mean_m^3s^-1"]["median"]
    b.axvline(qm, color=OKABE_ITO["black"], lw=0.8, ls="--", label=r"$\langle Q\rangle$")
    b.set_xscale("log")
    b.set_xlim(q.min(), q.max())
    b.set_ylim(0, hi.max() * 1.12)
    b.set_xlabel(r"umbrella flux $Q$ (m$^3$ s$^{-1}$)")
    b.set_ylabel(r"$\nu(Q)$ (10$^6$ kg per node)")
    b.legend(handlelength=1.3, loc="upper left", fontsize=5.5, borderaxespad=0.2)


def panel_c(c):
    shapes = ["blocky", "long", "sheet"]
    phi = [S["by_shape"][s]["phi_W"] / 1e12 for s in shapes]
    cols = [OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["vermillion"]]
    bars = c.bar(np.arange(3), phi, width=0.6, color=cols)
    bars[2].set_hatch("////")
    bars[2].set_edgecolor("white")
    bars[2].set_linewidth(0)
    for x, v in zip(np.arange(3), phi, strict=True):
        c.text(x, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    c.set_xticks(np.arange(3))
    c.set_xticklabels(shapes)
    c.set_xlim(-0.6, 2.6)
    c.set_xlabel("clast shape")
    c.set_ylabel(r"heat flux $\Phi$ (TW)")
    c.set_ylim(0, 1.5)


def panel_d(d):
    L_fit = S["by_shape"]["blocky"]["L_m"] / 1e3
    Lb = S["published_benchmark"]["L_m"] / 1e3
    Ls = S["published_benchmark"]["L_sigma_m"] / 1e3
    xs = np.linspace(3.4, 6.6, 300)
    g = np.exp(-0.5 * ((xs - Lb) / Ls) ** 2)
    d.fill_between(xs, g, color=OKABE_ITO["grey"], alpha=0.22, lw=0)
    d.plot(xs, g, color=OKABE_ITO["black"], lw=1.0, label=f"reference, {Lb:.1f} $\\pm$ {Ls:.1f} km")
    d.axvline(
        L_fit,
        color=OKABE_ITO["vermillion"],
        lw=1.4,
        label=f"steady single-fraction fit, {L_fit:.2f} km",
    )
    z = S["by_shape"]["blocky"]["L_z_score_vs_published"]
    d.text(
        L_fit + 0.08, 0.62, f"$z={z:+.2f}$", fontsize=6, color=OKABE_ITO["vermillion"], va="center"
    )
    d.set_xlim(3.4, 6.6)
    d.set_xlabel(r"dispersal length scale $L$ (km)")
    d.set_ylabel("relative likelihood")
    d.set_ylim(0, 1.45)
    d.set_yticks([0, 0.5, 1.0])
    d.legend(handlelength=1.3, loc="upper left", fontsize=5.5, borderaxespad=0.2)


def panel_maps(axes, cax, cores, vent):
    mp = M["maps"]
    gx = np.array(mp["x_m"]) / 1e3
    gy = np.array(mp["y_m"]) / 1e3
    norm = mcolors.BoundaryNorm(MAP_LEVELS, plt.get_cmap(MAP_CMAP).N, extend="both")
    cf = None
    for k, (ax, (name, *_r)) in enumerate(zip(axes, FRACTIONS, strict=True)):
        z = np.array(mp["log10_omega"][name])
        cf = ax.contourf(gx, gy, z, levels=MAP_LEVELS, cmap=MAP_CMAP, norm=norm, extend="both")
        x, y, v = cores[name]
        ax.scatter(
            x / 1e3,
            y / 1e3,
            c=np.log10(v),
            cmap=MAP_CMAP,
            norm=norm,
            s=7,
            edgecolors=(0, 0, 0, 0.55),
            linewidths=0.25,
            zorder=3,
        )
        ax.plot(
            vent[0] / 1e3,
            vent[1] / 1e3,
            marker="*",
            ms=6.5,
            mfc="white",
            mec="black",
            mew=0.5,
            zorder=4,
            ls="none",
        )
        ax.set_aspect("equal")
        ax.set_xlim(-5.5, 5.5)
        ax.set_ylim(-7.5, 7.0)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-6, -3, 0, 3, 6])
        ax.set_title(FRAC_LABEL[name], fontsize=6, pad=2.5)
        ax.set_xlabel("east of centroid (km)", labelpad=1)
        if k == 0:
            ax.set_ylabel("north of centroid (km)", labelpad=1)
        else:
            ax.tick_params(labelleft=False)
        for sp in ax.spines.values():
            sp.set_visible(True)
    cb = plt.colorbar(cf, cax=cax)
    cb.set_ticks([-4, -3, -2, -1, 0])
    cb.set_ticklabels(["10$^{-4}$", "10$^{-3}$", "10$^{-2}$", "10$^{-1}$", "1"])
    cb.set_label(r"$\Omega$ (kg m$^{-2}$)", labelpad=2)
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.outline.set_linewidth(0.4)


def hpd_levels(z, fracs):
    """Density thresholds enclosing the given posterior fractions."""
    zs = np.sort(z.ravel())[::-1]
    cum = np.cumsum(zs)
    cum /= cum[-1]
    return [float(zs[np.searchsorted(cum, f)]) for f in fracs]


def panel_joint(main, top, right):
    sm = M["samples_thinned"]
    m = np.array(sm["mass_total_kg"]) / 1e7
    q = np.array(sm["q_mass_weighted_mean_m^3s^-1"]) / 1e6
    kde = gaussian_kde(np.vstack([m, q]))
    mx = np.linspace(1.62, 2.32, 160)
    qy = np.linspace(0.86, 1.48, 160)
    X, Y = np.meshgrid(mx, qy)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    lv = hpd_levels(Z, [0.95, 0.80, 0.50])
    fill = np.linspace(lv[0], Z.max(), 9)
    main.contourf(X, Y, Z, levels=fill, cmap="Blues", extend="neither")
    main.contour(
        X,
        Y,
        Z,
        levels=lv,
        colors=[OKABE_ITO["black"]] * 3,
        linewidths=[0.5, 0.5, 0.8],
        linestyles=["dotted", "dashed", "solid"],
    )
    main.plot(
        np.median(m),
        np.median(q),
        "o",
        ms=2.8,
        color=OKABE_ITO["vermillion"],
        mec="white",
        mew=0.4,
        zorder=4,
    )
    main.set_xlim(mx[0], mx[-1])
    main.set_ylim(qy[0], qy[-1])
    main.set_xlabel(r"total mass (10$^7$ kg)")
    main.set_ylabel(r"$\langle Q\rangle$ (10$^6$ m$^3$ s$^{-1}$)")
    main.set_xticks([1.7, 1.9, 2.1, 2.3])
    for sp in ("top", "right"):
        main.spines[sp].set_visible(True)

    for ax, data, grid, horiz in ((top, m, mx, False), (right, q, qy, True)):
        k = gaussian_kde(data)(grid)
        k = k / k.max()
        if horiz:
            ax.fill_betweenx(grid, 0, k, color=OKABE_ITO["blue"], alpha=0.35, lw=0)
            ax.plot(k, grid, lw=0.8, color=OKABE_ITO["blue"])
            ax.set_ylim(grid[0], grid[-1])
            ax.set_xlim(0, 1.08)
        else:
            ax.fill_between(grid, 0, k, color=OKABE_ITO["blue"], alpha=0.35, lw=0)
            ax.plot(grid, k, lw=0.8, color=OKABE_ITO["blue"])
            ax.set_xlim(grid[0], grid[-1])
            ax.set_ylim(0, 1.08)
        ax.axis("off")


def main() -> int:
    cores = load_cores()
    vent = M["vent_median_xy_m"]

    fw, fh = TWO_COL, 6.75
    fig = plt.figure(figsize=(fw, fh))

    def box(left, bottom, width, height, **kw):
        return fig.add_axes([left / fw, bottom / fh, width / fw, height / fh], **kw)

    # row 1
    a = box(0.50, 4.95, 3.20, 1.66)
    b = box(4.35, 4.95, 2.66, 1.66)
    # row 2
    c = box(0.50, 2.86, 1.55, 1.52)
    d = box(2.60, 2.86, 2.00, 1.52)
    ex, ey, ew, eh, mg = 5.30, 2.86, 1.30, 1.18, 0.30
    e_main = box(ex, ey, ew, eh)
    e_top = box(ex, ey + eh + 0.04, ew, mg - 0.04, sharex=e_main)
    e_right = box(ex + ew + 0.04, ey, mg - 0.04, eh, sharey=e_main)
    # row 3: four maps and the colour bar
    mw, mh, gap, left = 1.40, 1.85, 0.10, 0.50
    f_axes = [box(left + k * (mw + gap), 0.36, mw, mh) for k in range(4)]
    cax = box(left + 4 * mw + 3 * gap + 0.10, 0.36, 0.08, mh)

    panel_a(a, cores, vent)
    panel_b(b)
    panel_c(c)
    panel_d(d)
    panel_joint(e_main, e_top, e_right)
    panel_maps(f_axes, cax, cores, vent)

    def lab(ax, text, dx_in, dy_in=0.10):
        pos = ax.get_position()
        fig.text(
            pos.x0 - dx_in / fw,
            pos.y1 + dy_in / fh,
            text,
            fontsize=7,
            fontweight="bold",
            va="top",
            ha="left",
        )

    lab(a, "a", 0.46)
    lab(b, "b", 0.46)
    lab(c, "c", 0.46)
    lab(d, "d", 0.46)
    lab(e_top, "e", 0.50)
    lab(f_axes[0], "f", 0.46, 0.22)

    out = ROOT / "figures" / "fig_nesca.pdf"
    fig.savefig(out)
    print(f"wrote {out}")
    fig.savefig(ROOT / "figures" / "fig_nesca.png", dpi=300)
    print("wrote figures/fig_nesca.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
