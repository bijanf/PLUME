"""Figure: the vent position as a parameter, and the heat flux it carries.

(a) posterior for the vent position against the core geometry and the prior
(b) heat flux with the vent profiled out and with it sampled
(c) heat flux by clast shape, with the range of the per-shape medians across the
    representations of the sieve fractions (results/shape_sensitivity.json)
(d) misfit of the deposit over trial vent positions, in plan view
(e) the misfit basin enlarged, with the conditional and marginal vent posteriors
(f) the misfit as a surface
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
from plume_style import OKABE_ITO, TWO_COL, apply_style, panel_label  # noqa: E402

apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from plume_inv.data import load_nesca  # noqa: E402

V = json.loads((ROOT / "results" / "vent_shape.json").read_text())
U = json.loads((ROOT / "results" / "nesca_unsteady.json").read_text())
W = json.loads((ROOT / "results" / "vent_misfit.json").read_text())
M = json.loads((ROOT / "results" / "nesca_maps.json").read_text())
SS = json.loads((ROOT / "results" / "shape_sensitivity.json").read_text())
SHAPES = ("blocky", "long", "sheet")
SCOL = {"blocky": OKABE_ITO["blue"], "long": OKABE_ITO["green"], "sheet": OKABE_ITO["vermillion"]}
TW = 1e12
# Misfit levels: log-spaced so both the basin floor and the far field show.
DNLL_LEVELS = np.array([0, 1, 3, 10, 30, 100, 300])
MISFIT_CMAP = "YlGnBu_r"
ZOOM_HW_KM = 0.5


def cores_xy():
    df = load_nesca()
    on_flow = df["Location"].astype(str).str.contains("on flow", case=False, na=False).values
    return df["x_m"].values[~on_flow] / 1e3, df["y_m"].values[~on_flow] / 1e3


def panel_a(a, cx, cy):
    a.scatter(cx, cy, s=1.2, color=OKABE_ITO["grey"], alpha=0.55, lw=0, label="cores", zorder=1)
    scale = V["vent_prior"]["scale_m"] / 1e3
    a.add_patch(
        Circle((0, 0), scale, fill=False, lw=0.6, ls=(0, (3, 2)), ec=OKABE_ITO["grey"], zorder=2)
    )
    a.text(
        -scale * 0.74,
        -scale * 0.74,
        r"prior 1$\sigma$",
        fontsize=5,
        color=OKABE_ITO["grey"],
        ha="right",
        va="top",
        zorder=2,
    )
    vx = np.array(V["vent_posterior_samples"]["x_m"]) / 1e3
    vy = np.array(V["vent_posterior_samples"]["y_m"]) / 1e3
    a.scatter(
        vx,
        vy,
        s=0.8,
        color=OKABE_ITO["vermillion"],
        alpha=0.30,
        lw=0,
        zorder=3,
        label="vent posterior",
    )
    a.plot(0, 0, "+", ms=5, mew=0.9, color=OKABE_ITO["black"], zorder=4, label="survey centroid")
    px, py = np.array(U["vent_xy_m"]) / 1e3
    a.plot(
        px,
        py,
        "x",
        ms=4.0,
        mew=0.9,
        color=OKABE_ITO["blue"],
        zorder=5,
        label="profiled (fixed)",
        ls="none",
    )
    a.set_xlabel("east of centroid (km)")
    a.set_ylabel("north of centroid (km)")
    a.set_aspect("equal")
    a.set_xlim(-2.8, 2.8)
    a.set_ylim(-2.8, 2.8)
    leg = a.legend(
        loc="lower left",
        handlelength=0.9,
        borderpad=0.25,
        labelspacing=0.28,
        handletextpad=0.5,
        fontsize=5,
        borderaxespad=0.1,
    )
    leg.set_zorder(6)

    # Inset: the posterior is 0.4 km across, small at survey scale.
    ins = a.inset_axes([0.69, 0.69, 0.30, 0.30])
    ins.scatter(vx, vy, s=1.4, color=OKABE_ITO["vermillion"], alpha=0.35, lw=0)
    ins.plot(px, py, "x", ms=4.0, mew=0.9, color=OKABE_ITO["blue"])
    mx, my = float(np.median(vx)), float(np.median(vy))
    ins.set_xlim(mx - 0.42, mx + 0.42)
    ins.set_ylim(my - 0.42, my + 0.42)
    ins.set_aspect("equal")
    ins.set_xticks([round(mx - 0.3, 1), round(mx + 0.3, 1)])
    ins.set_yticks([round(my - 0.3, 1), round(my + 0.3, 1)])
    ins.tick_params(labelsize=5, pad=1, length=1.5)
    ins.yaxis.tick_right()
    for sp in ins.spines.values():
        sp.set_linewidth(0.4)
        sp.set_visible(True)
    a.indicate_inset_zoom(ins, edgecolor=OKABE_ITO["grey"], lw=0.4, alpha=0.9)


def panel_b(b):
    prof = U["posterior"]["phi_at_mass_weighted_mean_W"]
    samp = V["by_shape"]["blocky"]["phi_at_mass_weighted_mean_W"]
    widths = W["phi_ci95_widths"]
    rows = [
        (prof, widths["profiled_W"], OKABE_ITO["blue"]),
        (samp, widths["sampled_W"], OKABE_ITO["vermillion"]),
    ]
    for i, (d, wid, col) in enumerate(rows):
        lo, hi = d["ci95"][0] / TW, d["ci95"][1] / TW
        b.plot([lo, hi], [i, i], lw=2.2, color=col, solid_capstyle="round")
        b.plot(d["median"] / TW, i, "o", ms=3.4, color=col, zorder=3)
        b.text(
            hi + 0.05,
            i,
            f"width {wid / TW:.2f} TW",
            fontsize=5.5,
            va="center",
            ha="left",
            color=col,
        )
    b.set_yticks([0, 1])
    b.set_yticklabels(["vent\nprofiled", "vent\nsampled"])
    b.set_ylim(-0.6, 1.6)
    b.set_xlim(1.25, 3.25)
    b.set_xlabel(r"$\Phi$ at $\langle Q\rangle$, blocky clasts (TW)")


def panel_c(c):
    ps = V["phi_samples_W"]
    grid = np.linspace(0.2, 3.5, 440)

    def kde(x, bw=0.085):
        x = np.asarray(x) / TW
        d = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2).sum(1)
        return d / (d.max() + 1e-30)

    for s in SHAPES:
        c.plot(
            grid,
            kde(ps["by_shape"][s]),
            lw=1.1,
            color=SCOL[s],
            ls=(0, (2.5, 1.5)) if s == "sheet" else "-",
            zorder=3,
            label=s,
        )
    # Range of the per-shape medians across the representations of the sieve
    # fractions (diameter convention and upper bound of the coarse fraction).
    rng_ = SS["summary"]["phi_by_shape_median_range_W"]
    lo = min(v[0] for v in rng_.values()) / TW
    hi = max(v[1] for v in rng_.values()) / TW
    c.plot(
        [lo, hi],
        [-0.07, -0.07],
        lw=1.6,
        color=OKABE_ITO["grey"],
        solid_capstyle="butt",
        clip_on=False,
        label="sieve representation",
    )
    c.set_xlabel(r"$\Phi$ at $\langle Q\rangle$ (TW)")
    c.set_ylabel("posterior density (scaled)")
    c.set_xlim(0.2, 3.5)
    c.set_ylim(-0.12, 1.12)
    c.set_yticks([0, 0.5, 1.0])
    c.legend(
        loc="upper right",
        handlelength=1.0,
        borderpad=0.2,
        borderaxespad=0.1,
        labelspacing=0.25,
        fontsize=5.5,
    )


def misfit_grid():
    g = W["grid"]
    gx = np.array(g["x_m"]) / 1e3
    gy = np.array(g["y_m"]) / 1e3
    z = np.array(g["delta_nll"])
    return gx, gy, z


def blocky_vent_draws():
    sm = M["samples_thinned"]
    return np.array(sm["vent_x_m"]) / 1e3, np.array(sm["vent_y_m"]) / 1e3


def hpd_contour(ax, xs, ys, Z, fracs, **kw):
    zs = np.sort(Z.ravel())[::-1]
    cum = np.cumsum(zs) / zs.sum()
    lv = sorted(float(zs[np.searchsorted(cum, f)]) for f in fracs)
    ax.contour(xs, ys, Z, levels=lv, **kw)


def kde_grid(vx, vy, pad=0.15, n=140):
    kde = gaussian_kde(np.vstack([vx, vy]))
    xs = np.linspace(vx.min() - pad, vx.max() + pad, n)
    ys = np.linspace(vy.min() - pad, vy.max() + pad, n)
    X, Y = np.meshgrid(xs, ys)
    return xs, ys, kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)


def misfit_fill(ax, gx, gy, z, lines=True):
    cmap = plt.get_cmap(MISFIT_CMAP)
    norm = mcolors.BoundaryNorm(DNLL_LEVELS, cmap.N, extend="max")
    cf = ax.contourf(gx, gy, z, levels=DNLL_LEVELS, cmap=cmap, norm=norm, extend="max")
    if lines:
        ax.contour(gx, gy, z, levels=DNLL_LEVELS[1:], colors="white", linewidths=0.3, alpha=0.8)
    return cf


def panel_d(d, cx, cy):
    gx, gy, z = misfit_grid()
    cf = misfit_fill(d, gx, gy, z)
    d.scatter(
        cx, cy, s=2.2, facecolor="white", edgecolor=OKABE_ITO["black"], linewidths=0.25, zorder=3
    )
    scale = W["vent_prior_scale_m"] / 1e3
    d.add_patch(
        Circle((0, 0), scale, fill=False, lw=0.6, ls=(0, (3, 2)), ec=OKABE_ITO["black"], zorder=3)
    )
    vx, vy = blocky_vent_draws()
    xs, ys, Z = kde_grid(vx, vy)
    hpd_contour(
        d, xs, ys, Z, (0.95, 0.5), colors=[OKABE_ITO["orange"]], linewidths=[0.6, 1.0], zorder=4
    )
    d.plot(0, 0, "+", ms=5, mew=0.9, color=OKABE_ITO["black"], zorder=5)
    # Frame of the zoom in panel e.
    zx = np.array(W["zoom_grid"]["x_m"]) / 1e3
    zy = np.array(W["zoom_grid"]["y_m"]) / 1e3
    x0, y0 = W["zoom_grid"]["argmin_xy_m"]
    hw = ZOOM_HW_KM
    d.add_patch(
        plt.Rectangle(
            (x0 / 1e3 - hw, y0 / 1e3 - hw),
            2 * hw,
            2 * hw,
            fill=False,
            lw=0.5,
            ec=OKABE_ITO["black"],
            zorder=5,
        )
    )
    del zx, zy
    d.set_aspect("equal")
    d.set_xlim(-2.8, 2.8)
    d.set_ylim(-2.8, 2.8)
    d.set_xticks([-2, -1, 0, 1, 2])
    d.set_yticks([-2, -1, 0, 1, 2])
    d.set_xlabel("vent east of centroid (km)")
    d.set_ylabel("vent north of centroid (km)")
    for sp in d.spines.values():
        sp.set_visible(True)
    return cf


def panel_e(e):
    zg = W["zoom_grid"]
    gx = np.array(zg["x_m"]) / 1e3
    gy = np.array(zg["y_m"]) / 1e3
    misfit_fill(e, gx, gy, np.array(zg["delta_nll"]), lines=False)
    # Conditional posterior (misfit plus prior, nu shape fixed): 50 % and 95 %.
    hpd_contour(
        e,
        gx,
        gy,
        np.exp(np.array(zg["log_conditional_posterior"])),
        (0.95, 0.5),
        colors="white",
        linewidths=[0.6, 1.0],
        linestyles="dashed",
        zorder=4,
    )
    # Marginal posterior from the sampler: 50 % and 95 %.
    vx, vy = blocky_vent_draws()
    xs, ys, Z = kde_grid(vx, vy)
    hpd_contour(
        e, xs, ys, Z, (0.95, 0.5), colors=[OKABE_ITO["orange"]], linewidths=[0.6, 1.0], zorder=5
    )
    px, py = np.array(U["vent_xy_m"]) / 1e3
    e.plot(px, py, "x", ms=4.0, mew=1.0, color="white", zorder=6)
    x0, y0 = (v / 1e3 for v in zg["argmin_xy_m"])
    e.set_aspect("equal")
    e.set_xlim(x0 - ZOOM_HW_KM, x0 + ZOOM_HW_KM)
    e.set_ylim(y0 - ZOOM_HW_KM, y0 + ZOOM_HW_KM)
    e.set_xticks([0.2, 0.5, 0.8])
    e.set_yticks([-1.0, -0.75, -0.5])
    e.set_xlabel("vent east of centroid (km)")
    e.set_ylabel("vent north of centroid (km)", labelpad=1)
    for sp in e.spines.values():
        sp.set_visible(True)


def panel_f(f):
    gx, gy, z = misfit_grid()
    sel_x = (gx >= -2.8) & (gx <= 2.8)
    sel_y = (gy >= -2.8) & (gy <= 2.8)
    gx, gy, z = gx[sel_x], gy[sel_y], z[np.ix_(sel_y, sel_x)]
    X, Y = np.meshgrid(gx, gy)
    top = float(DNLL_LEVELS[-1])
    zl = np.log10(1.0 + np.minimum(z, top))
    cmap = plt.get_cmap(MISFIT_CMAP)
    cnorm = mcolors.Normalize(0.0, np.log10(1.0 + top))
    f.plot_surface(
        X,
        Y,
        zl,
        facecolors=cmap(cnorm(zl)),
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,
        rasterized=True,
    )
    f.contour(
        X,
        Y,
        zl,
        levels=np.log10(1.0 + DNLL_LEVELS[1:]),
        zdir="z",
        offset=0.0,
        colors=OKABE_ITO["grey"],
        linewidths=0.3,
    )
    f.set_xlim(-2.8, 2.8)
    f.set_ylim(-2.8, 2.8)
    f.set_zlim(0, np.log10(1.0 + top))
    f.set_xticks([-2, 0, 2])
    f.set_yticks([-2, 0, 2])
    f.set_zticks([np.log10(1 + v) for v in (0, 10, 100)])
    f.set_zticklabels(["0", "10", "100"])
    f.set_xlabel("east (km)", labelpad=-7)
    f.set_ylabel("north (km)", labelpad=-7)
    f.set_zlabel(r"$\Delta$NLL", labelpad=-7)
    f.set_box_aspect(None, zoom=0.82)
    f.patch.set_alpha(0.0)
    f.tick_params(axis="both", labelsize=5, pad=-3)
    f.tick_params(axis="z", labelsize=5, pad=-1)
    f.view_init(elev=28, azim=-58)
    for axis in (f.xaxis, f.yaxis, f.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((0.6, 0.6, 0.6, 1))
        axis._axinfo["grid"]["linewidth"] = 0.25
        axis._axinfo["grid"]["color"] = (0.8, 0.8, 0.8, 1)
        axis.line.set_linewidth(0.4)


def main() -> int:
    cx, cy = cores_xy()
    fw, fh = TWO_COL, 4.95
    fig = plt.figure(figsize=(fw, fh))

    def box(left, bottom, width, height, **kw):
        return fig.add_axes([left / fw, bottom / fh, width / fw, height / fh], **kw)

    top = GridSpec(1, 3, figure=fig, left=0.065, right=0.985, top=0.975, bottom=0.595, wspace=0.34)
    a = fig.add_subplot(top[0])
    b = fig.add_subplot(top[1])
    c = fig.add_subplot(top[2])
    side = 1.86
    d = box(0.46, 0.34, side, side)
    e = box(2.80, 0.34, side, side)
    cax = box(4.73, 0.34, 0.075, side)
    f = box(5.05, 0.14, 1.84, 2.26, projection="3d")

    panel_a(a, cx, cy)
    panel_b(b)
    panel_c(c)
    cf = panel_d(d, cx, cy)
    panel_e(e)
    panel_f(f)
    cb = plt.colorbar(cf, cax=cax)
    cb.set_ticks(DNLL_LEVELS)
    cb.set_label(r"$\Delta$NLL", labelpad=1)
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.outline.set_linewidth(0.4)

    panel_label(a, "a", dx=-0.25)
    panel_label(b, "b", dx=-0.25)
    panel_label(c, "c", dx=-0.25)
    panel_label(d, "d", dx=-0.23, dy=1.08)
    panel_label(e, "e", dx=-0.27, dy=1.08)
    f.text2D(
        0.02, 1.0, "f", transform=f.transAxes, fontsize=7, fontweight="bold", va="top", ha="left"
    )

    out = ROOT / "figures" / "fig_vent.pdf"
    fig.savefig(out, dpi=600)
    print(f"wrote {out}")
    fig.savefig(ROOT / "figures" / "fig_vent.png", dpi=300)
    print("wrote figures/fig_vent.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
