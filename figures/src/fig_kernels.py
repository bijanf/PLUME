"""Figure 1: quasi-steady Gaussian kernel versus the unsteady, front-limited kernel.

Panels
------
a  Umbrella front radius and thickness against time, with the cored radii.
b  Deposit for two size classes: quasi-steady versus front-limited, constant source.
c  Ratio front-limited / quasi-steady against radius, for four size classes.
d  Quasi-steady deposits of a constant, a waning and a two-pulse source at fixed
   total released mass.
e  Space-time map of the deposition rate of the 250-500 um fraction, with the
   front trajectory, the source shut-off, the cored radii and the time by which
   half of the local deposit has landed.
f  Ratio front-limited / quasi-steady over radius and settling speed.
g  Front-limited deposit over radius and settling speed as a surface, coloured
   by the ratio of panel f.

Panels e-g are drawn from ``results/kernel_maps_grids.npz`` and
``results/kernel_maps.json`` (``experiments/kernel_maps/run.py``).

Outputs figures/fig_kernels.pdf (vector) and a .png preview.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from plume_style import OKABE_ITO, TWO_COL, apply_style, panel_label  # noqa: E402

from plume_inv import forward as fwd  # noqa: E402
from plume_inv import kernel as K  # noqa: E402
from plume_inv import settling, umbrella  # noqa: E402
from plume_inv.constants import LAMBDA_FRONT  # noqa: E402

apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib import ticker as mticker  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

Q0 = 7.6e5
MASS = 1.0e9
TAU = 15 * 3600.0
FLOOR = 1e-2
SPACE_TIME_CLASS = "250-500"

strat = json.loads((ROOT / "results" / "stratification.json").read_text())
N_BUOY = strat["N_primary"]["value_s^-1"]
MAPS = json.loads((ROOT / "results" / "kernel_maps.json").read_text())
GRID = np.load(ROOT / "results" / "kernel_maps_grids.npz")
R_CORE_MIN = MAPS["cored_radii"]["r_min_m"] / 1e3
R_CORE_MAX = MAPS["cored_radii"]["r_max_m"] / 1e3
T_FRONT_RMAX = MAPS["front"]["t_reaches_r_max_h"]

CLASSES = [
    ("63-125", 63e-6, 125e-6, OKABE_ITO["sky"]),
    ("125-250", 125e-6, 250e-6, OKABE_ITO["green"]),
    ("250-500", 250e-6, 500e-6, OKABE_ITO["blue"]),
    ("500-1000", 500e-6, 1e-3, OKABE_ITO["vermillion"]),
]
W_OF = {
    n: float(settling.settling_velocity(settling.sieve_midpoint(a, b))) for n, a, b, _ in CLASSES
}
COL_OF = {n: c for n, _a, _b, c in CLASSES}
LABEL_OF = {"63-125": "63–125", "125-250": "125–250", "250-500": "250–500", "500-1000": ">500"}

# Ratio colour scale, shared by panels f and g: log2 ratio, 1/8 to 8.
RATIO_CMAP = plt.get_cmap("RdBu_r")
RATIO_LEVELS = np.arange(-3.0, 3.01, 0.5)
RATIO_NORM = mcolors.BoundaryNorm(RATIO_LEVELS, RATIO_CMAP.N, extend="both")
MASK_GREY = "#E3E3E3"


def unsteady(r, src, w, n_quad=4096):
    t_end = K.settling_window(float(src.t[-1]), float(np.mean(src.q)), N_BUOY, w, LAMBDA_FRONT)
    t, q, m = K.extend_history(src.t, src.q, src.mdot, t_end, n_extra=2500)
    umb = umbrella.solve_umbrella(lambda tt: float(np.interp(tt, t, q)), t, N_BUOY, LAMBDA_FRONT)
    return K.deposit_unsteady(r, umb, t, q, m, w, 1.0, n_quad=n_quad), umb


def pow2_label(x, _pos=None):
    v = 2.0**x
    return f"{v:g}" if v >= 1 else f"1/{1 / v:g}"


def panel_a(ax):
    t_h = np.linspace(0.02, 30.0, 400)
    umb_c = umbrella.solve_umbrella(
        lambda tt: Q0 if tt <= TAU else 0.0, t_h * 3600.0, N_BUOY, LAMBDA_FRONT
    )
    ax.axhspan(
        R_CORE_MIN, R_CORE_MAX, color=OKABE_ITO["yellow"], alpha=0.28, lw=0, label="cored radii"
    )
    ax.plot(t_h, umb_c.front_radius / 1e3, color=OKABE_ITO["black"], lw=1.2, label=r"front $R_f$")
    ax.axvline(TAU / 3600.0, color=OKABE_ITO["grey"], lw=0.6, ls=":")
    ax.text(
        TAU / 3600.0 + 0.4,
        9.8,
        "source off",
        fontsize=5,
        color=OKABE_ITO["grey"],
        rotation=90,
        va="top",
        ha="left",
    )
    ax.set_xlabel("time (h)")
    ax.set_ylabel("front radius (km)")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 10)
    ax.legend(loc="upper left", bbox_to_anchor=(0.11, 1.0), handlelength=1.4)
    axt = ax.twinx()
    axt.plot(t_h, umb_c.thickness, color=OKABE_ITO["purple"], lw=1.0, ls="--")
    axt.set_ylabel("thickness $h$ (m)", color=OKABE_ITO["purple"])
    axt.tick_params(axis="y", colors=OKABE_ITO["purple"], width=0.5)
    axt.set_ylim(0, 900)
    axt.spines["top"].set_visible(False)
    axt.spines["right"].set_visible(True)
    axt.spines["right"].set_linewidth(0.5)
    panel_label(ax, "a")


def panel_b(ax, r, src):
    for name, _a, _b, col in CLASSES[1:3]:
        w = W_OF[name]
        g = K.deposit_quasi_steady(r, src.t, src.q, src.mdot, w, 1.0)
        u, _ = unsteady(r, src, w)
        ax.plot(r / 1e3, g, color=col, lw=1.0, ls="--")
        ax.plot(
            r / 1e3,
            u,
            color=col,
            lw=1.2,
            ls="-",
            label=rf"{LABEL_OF[name]} $\mu$m, $w_s$ = {100 * w:.1f} cm s$^{{-1}}$",
        )
    ax.plot([], [], color=OKABE_ITO["black"], ls="--", lw=1.0, label="quasi-steady")
    ax.plot([], [], color=OKABE_ITO["black"], ls="-", lw=1.2, label="front-limited")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 3e2)
    ax.set_xlim(0, 9.5)
    ax.set_xlabel("radius (km)")
    ax.set_ylabel(r"deposit $\Omega$ (kg m$^{-2}$)")
    ax.legend(loc="upper right", handlelength=1.6)
    panel_label(ax, "b")


def panel_c(ax, r, src):
    ratio_250 = None
    for name, _a, _b, col in CLASSES:
        w = W_OF[name]
        g = K.deposit_quasi_steady(r, src.t, src.q, src.mdot, w, 1.0)
        u, _ = unsteady(r, src, w)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(g > FLOOR, u / g, np.nan)
        if name == "250-500":
            ratio_250 = ratio
        ax.plot(r / 1e3, ratio, color=col, lw=1.1, label=rf"{LABEL_OF[name]} $\mu$m")
        vis = np.where(g > FLOOR)[0]
        if vis.size and vis[-1] < r.size - 1:
            ax.plot(r[vis[-1]] / 1e3, ratio[vis[-1]], "o", color=col, ms=2.2, mew=0)
    ax.axhline(1.0, color=OKABE_ITO["grey"], lw=0.6, ls=":")
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 2.2)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("radius (km)")
    ax.set_ylabel("front-limited / quasi-steady")
    ax.legend(loc="upper right", handlelength=1.4, ncol=2, columnspacing=1.0)
    panel_label(ax, "c")
    return ratio_250


def panel_d(ax, r):
    w = W_OF["250-500"]
    histories = [
        ("constant $Q$", fwd.constant_history(Q0, MASS, TAU, n_t=3001), OKABE_ITO["blue"]),
        (
            "waning $Q$",
            fwd.waning_history(2.2e6, 0.28 * TAU, MASS, TAU, n_t=3001),
            OKABE_ITO["orange"],
        ),
        (
            "two pulses",
            fwd.two_pulse_history(2.5e5, 2.2e6, 0.25 * TAU, 0.45 * TAU, MASS, TAU, n_t=3001),
            OKABE_ITO["purple"],
        ),
    ]
    for label, h, col in histories:
        ax.plot(
            r / 1e3,
            K.deposit_quasi_steady(r, h.t, h.q, h.mdot, w, 1.0),
            color=col,
            lw=1.1,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 3e2)
    ax.set_xlim(0, 9.5)
    ax.set_xlabel("radius (km)")
    ax.set_ylabel(r"deposit $\Omega$ (kg m$^{-2}$)")
    ax.legend(loc="upper right", handlelength=1.6)
    panel_label(ax, "d")


def panel_e(ax, cax):
    r_km = GRID["r_km"]
    t_h = GRID["t_h"]
    rate = GRID[f"rate_{SPACE_TIME_CLASS}"].astype(float) * 3600.0  # per hour
    cum = GRID[f"cumfrac_{SPACE_TIME_CLASS}"].astype(float)
    keep = t_h <= 30.0 + 1e-9
    t_h, rate, cum = t_h[keep], rate[:, keep], cum[:, keep]
    with np.errstate(divide="ignore"):
        lr = np.where(rate > 0, np.log10(rate), np.nan)
    levels = np.arange(-3.0, 1.01, 0.5)
    cf = ax.contourf(t_h, r_km, lr, levels=levels, cmap="viridis", extend="max")
    ax.contour(t_h, r_km, cum, levels=[0.5], colors="white", linewidths=0.9)
    ax.plot(GRID["front_t_h"], GRID["front_r_km"], color=OKABE_ITO["black"], lw=1.1)
    ax.axvline(TAU / 3600.0, color=OKABE_ITO["black"], lw=0.6, ls=":")
    for rc in (R_CORE_MIN, R_CORE_MAX):
        ax.axhline(rc, color=OKABE_ITO["black"], lw=0.5, ls=(0, (3, 2)))
    ax.plot(
        [T_FRONT_RMAX],
        [R_CORE_MAX],
        "o",
        ms=3.0,
        mfc=OKABE_ITO["yellow"],
        mec=OKABE_ITO["black"],
        mew=0.5,
        zorder=5,
    )
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 9.5)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("radius (km)")
    cb = plt.colorbar(cf, cax=cax)
    cb.set_ticks([-3, -2, -1, 0, 1])
    cb.set_ticklabels(["10$^{-3}$", "10$^{-2}$", "10$^{-1}$", "10$^{0}$", "10$^{1}$"])
    cb.set_label(r"deposition rate (kg m$^{-2}$ h$^{-1}$)", labelpad=2)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2)
    panel_label(ax, "e", dx=-0.30)


def panel_f(ax):
    r_km = GRID["r_km"]
    w_cm = GRID["w_grid"] * 100.0
    ratio = GRID["ratio"]
    with np.errstate(divide="ignore", invalid="ignore"):
        l2 = np.log2(ratio)
    ax.set_facecolor(MASK_GREY)
    cf = ax.contourf(
        r_km, w_cm, l2, levels=RATIO_LEVELS, cmap=RATIO_CMAP, norm=RATIO_NORM, extend="both"
    )
    ax.contour(r_km, w_cm, l2, levels=[0.0], colors="black", linewidths=1.2)
    for name in COL_OF:
        ax.axhline(W_OF[name] * 100.0, color="black", lw=0.5, ls=(0, (1, 1.5)))
    ax.set_yscale("log")
    ax.set_ylim(w_cm.min(), w_cm.max())
    ax.set_xlim(0, 9.5)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_xlabel("radius (km)")
    ax.set_ylabel(r"settling speed $w_s$ (cm s$^{-1}$)")
    # Cored radii as a bracket above the data.
    ax.plot(
        [R_CORE_MIN, R_CORE_MAX],
        [1.035, 1.035],
        transform=ax.get_xaxis_transform(),
        color=OKABE_ITO["black"],
        lw=1.6,
        solid_capstyle="butt",
        clip_on=False,
    )
    ax.text(
        0.5 * (R_CORE_MIN + R_CORE_MAX),
        1.06,
        "cored radii",
        fontsize=5,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
    )
    # Fraction names on the right-hand axis.
    axr = ax.twinx()
    axr.set_yscale("log")
    axr.set_ylim(ax.get_ylim())
    axr.set_yticks([W_OF[n] * 100.0 for n in COL_OF])
    axr.set_yticklabels([LABEL_OF[n] for n in COL_OF], fontsize=5)
    axr.yaxis.set_minor_locator(mticker.NullLocator())
    axr.tick_params(axis="y", length=2, width=0.5, pad=1.5)
    for tl, n in zip(axr.get_yticklabels(), COL_OF, strict=True):
        tl.set_color(COL_OF[n])
    axr.spines["right"].set_visible(True)
    axr.spines["right"].set_linewidth(0.5)
    axr.spines["top"].set_visible(False)
    panel_label(ax, "f", dx=-0.30)
    return cf


def panel_g(ax):
    r_km = GRID["r_km"]
    lw = np.log10(GRID["w_grid"] * 100.0)
    z = GRID["log10_omega_front"]
    ratio = GRID["ratio"]
    zmin = np.log10(FLOOR)
    zc = np.clip(z, zmin, None)
    rr, ww = np.meshgrid(r_km, lw)
    with np.errstate(divide="ignore", invalid="ignore"):
        l2 = np.log2(ratio)
    # Continuous version of the panel f scale, so band edges do not step along
    # the grid cells of the surface.
    fc = RATIO_CMAP(
        mcolors.Normalize(RATIO_LEVELS[0], RATIO_LEVELS[-1])(np.nan_to_num(l2, nan=0.0))
    )
    fc[~np.isfinite(l2)] = mcolors.to_rgba(MASK_GREY)
    fc[z < zmin] = mcolors.to_rgba(MASK_GREY)
    ax.plot_surface(
        rr,
        ww,
        zc,
        facecolors=fc,
        rstride=1,
        cstride=1,
        shade=False,
        linewidth=0,
        antialiased=False,
        rasterized=True,
    )
    # Front-limited deposits of the four fractions, traced on the surface.
    for name in COL_OF:
        i = int(np.argmin(np.abs(GRID["w_grid"] - W_OF[name])))
        ax.plot(
            r_km,
            np.full_like(r_km, lw[i]),
            np.clip(z[i], zmin, None),
            color="black",
            lw=0.6,
            zorder=10,
        )
    ax.set_xlim(0, 9.5)
    ax.set_ylim(lw.min(), lw.max())
    ax.set_zlim(zmin, 2.5)
    ax.set_xticks([0, 4, 8])
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["0.1", "1", "10"])
    ax.set_zticks([-2, 0, 2])
    ax.set_zticklabels(["10$^{-2}$", "10$^{0}$", "10$^{2}$"])
    ax.set_xlabel("radius (km)", labelpad=-7)
    ax.set_ylabel(r"$w_s$ (cm s$^{-1}$)", labelpad=-7)
    ax.set_zlabel(r"$\Omega$ (kg m$^{-2}$)", labelpad=-6)
    ax.tick_params(axis="both", pad=-3, labelsize=5)
    ax.tick_params(axis="z", pad=-1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((0.6, 0.6, 0.6, 1))
        axis.pane.set_linewidth(0.4)
        axis._axinfo["grid"]["linewidth"] = 0.25
        axis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 1)
        axis.line.set_linewidth(0.5)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    ax.view_init(elev=30, azim=-42)
    ax.set_box_aspect((1.2, 1.0, 0.8), zoom=1.0)


def main() -> int:
    fig = plt.figure(figsize=(TWO_COL, 6.6))
    top = GridSpec(
        2, 2, figure=fig, left=0.075, right=0.985, top=0.985, bottom=0.425, hspace=0.36, wspace=0.34
    )
    bot = GridSpec(
        1,
        8,
        figure=fig,
        left=0.075,
        right=0.94,
        top=0.325,
        bottom=0.058,
        width_ratios=[1.0, 0.045, 0.70, 1.0, 0.36, 0.045, 0.36, 1.30],
        wspace=0.05,
    )

    src = fwd.constant_history(Q0, MASS, TAU, n_t=3001)
    r = np.linspace(50.0, 9500.0, 260)

    panel_a(fig.add_subplot(top[0, 0]))
    panel_b(fig.add_subplot(top[0, 1]), r, src)
    ratio_250 = panel_c(fig.add_subplot(top[1, 0]), r, src)
    panel_d(fig.add_subplot(top[1, 1]), r)

    panel_e(fig.add_subplot(bot[0, 0]), fig.add_subplot(bot[0, 1]))
    cf = panel_f(fig.add_subplot(bot[0, 3]))
    cax = fig.add_subplot(bot[0, 5])
    cb = fig.colorbar(cf, cax=cax, ticks=np.arange(-3, 4))
    cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(pow2_label))
    cb.set_label("front-limited / quasi-steady", labelpad=2)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2)
    ax_g = fig.add_subplot(bot[0, 7], projection="3d")
    panel_g(ax_g)
    # The 3D axes carry wide internal margins: enlarge the box within its slot.
    pos = ax_g.get_position()
    ax_g.set_position([pos.x0 - 0.035, pos.y0 - 0.05, pos.width + 0.04, pos.height + 0.08])
    # Same style as plume_style.panel_label, which 3D axes cannot take.
    fig.text(pos.x0 + 0.005, 0.325 + 0.012, "g", fontsize=7, fontweight="bold", va="top", ha="left")

    # Consistency: the map row of the 250-500 um class reproduces panel c.
    i = int(np.argmin(np.abs(GRID["w_grid"] - W_OF["250-500"])))
    rc = np.interp(GRID["r_km"] * 1e3, r, ratio_250)
    ok = np.isfinite(rc) & np.isfinite(GRID["ratio"][i]) & (GRID["r_km"] < 9.4)
    dev = np.max(np.abs(GRID["ratio"][i][ok] / rc[ok] - 1.0))
    print(f"panel f row (250-500 um) vs panel c: max rel diff {dev:.2e}")

    fig.savefig(ROOT / "figures" / "fig_kernels.pdf", dpi=600)
    fig.savefig(ROOT / "figures" / "fig_kernels.png", dpi=300)
    print("wrote figures/fig_kernels.pdf and .png")
    print(f"N = {N_BUOY:.3e} s^-1 (from results/stratification.json)")
    print("w_s:", {k: round(v, 4) for k, v in W_OF.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
