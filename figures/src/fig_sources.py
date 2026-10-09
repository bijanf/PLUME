"""Figure: which mechanism can power the NESCA megaplume, and how extreme it is.

(a) heat delivered over a 15 h eruption, against the megaplume range
(b) Phi(t) for each mechanism at the top of its bounded range
(c) Bayes factors, with the range over prior widths
(d) peak flux against the inferred flux under three settling laws
(e) the inferred flux on a ladder of verified heat fluxes
(f) heat delivered by lava cooling over (eruption duration, flow area)
(g) heat delivered by a dyke over (length, height) in 15 h
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
from plume_style import MM, OKABE_ITO, TWO_COL, apply_style  # noqa: E402

apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib import ticker  # noqa: E402

from plume_inv import sources  # noqa: E402

D = json.loads((ROOT / "results" / "model_comparison.json").read_text())
DS = json.loads((ROOT / "results" / "model_comparison_by_shape.json").read_text())
DT = D["delivery_test"]
HC = json.loads((ROOT / "results" / "heat_context.json").read_text())
VS = json.loads((ROOT / "results" / "vent_shape.json").read_text())

LABEL = {
    "lava_cooling": "lava\ncooling",
    "dyke_heating": "dyke\nheating",
    "volatile_exsolution": "CO$_2$\nexsolution",
    "hydrothermal_evacuation": "hydrothermal\nevacuation",
}
ORDER = ["lava_cooling", "volatile_exsolution", "dyke_heating", "hydrothermal_evacuation"]
COL = {
    "lava_cooling": OKABE_ITO["vermillion"],
    "volatile_exsolution": OKABE_ITO["orange"],
    "dyke_heating": OKABE_ITO["green"],
    "hydrothermal_evacuation": OKABE_ITO["blue"],
}
SCOL = {
    "blocky": OKABE_ITO["blue"],
    "long": OKABE_ITO["green"],
    "sheet": OKABE_ITO["vermillion"],
}  # as in every other figure
GROUP_COL = {
    "steady": "#6E6E6E",
    "global": "#6E6E6E",
    "megaplume": OKABE_ITO["purple"],
    "nesca": OKABE_ITO["black"],
}
FS = 5.5  # annotation and small tick text, pt
W_MM, H_MM = 180.0, 168.0


def ax_mm(fig, x, y_top, w, h):
    """Axes from a rectangle in mm, measured from the top-left of the figure."""
    return fig.add_axes([x / W_MM, 1 - (y_top + h) / H_MM, w / W_MM, h / H_MM])


def label_mm(fig, x, y_top, text):
    fig.text(x / W_MM, 1 - y_top / H_MM, text, fontsize=7, fontweight="bold", ha="left", va="top")


def sci(v: float) -> str:
    m, e = f"{v:.1e}".split("e")
    return rf"{m}$\times10^{{{int(e)}}}$"


def mech_axis(ax):
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([LABEL[m] for m in ORDER], fontsize=FS)
    ax.set_xlim(-0.55, 3.55)
    ax.tick_params(axis="x", length=0)


# --------------------------------------------------------------------------
def panel_a(a):
    lo, hi = DT["_megaplume_heat_range_J"]
    e = [DT[m]["energy_delivered_J"] for m in ORDER]
    a.axhspan(lo, hi, color=OKABE_ITO["grey"], alpha=0.22, lw=0, zorder=0)
    a.text(
        0.5,
        lo * 10**0.64,
        "megaplume\nrange",
        fontsize=FS,
        ha="center",
        va="center",
        color="#4D4D4D",
        linespacing=1.15,
    )
    a.bar(np.arange(4), e, width=0.6, color=[COL[m] for m in ORDER], zorder=2)
    for x, v in zip(np.arange(4), e, strict=True):
        a.text(x, v * 1.12, sci(v), ha="center", va="bottom", fontsize=FS)
    a.set_yscale("log")
    a.set_ylim(1e15, 1e19)
    a.yaxis.set_major_locator(ticker.LogLocator(numticks=6))
    a.yaxis.set_minor_locator(ticker.NullLocator())
    mech_axis(a)
    a.set_ylabel("heat delivered in 15 h (J)")


def panel_b(b):
    t = np.linspace(60.0, 15 * 3600.0, 2000)
    curves = {
        "lava_cooling": sources.phi_lava_cooling(t, 15e6, 1800.0),
        "dyke_heating": sources.phi_dyke(t, 3e4, 5e3),
        "volatile_exsolution": np.where(
            t <= 1800.0, sources.volatile_heat_budget(4.5e7, 0.02) / 1800.0, 1e6
        ),
        "hydrothermal_evacuation": sources.phi_hydrothermal(t, 1e-12, 1e8, 1000.0, 3e7, 400.0)[0],
    }
    for m in ORDER:
        b.plot(
            t / 3600.0,
            curves[m] / 1e12,
            "-",
            lw=1.1,
            color=COL[m],
            label=LABEL[m].replace("\n", " "),
        )
    phi_inf = DT["_inferred_phi_W"] / 1e12
    b.axhline(phi_inf, color=OKABE_ITO["black"], lw=0.8, ls="--")
    b.text(
        7.5,
        phi_inf / 1.45,
        rf"inferred $\Phi$, {phi_inf:.2f} TW",
        fontsize=FS,
        ha="center",
        va="top",
    )
    b.set_yscale("log")
    b.set_ylim(1e-4, 1e3)
    b.yaxis.set_major_locator(ticker.LogLocator(numticks=8))
    b.yaxis.set_minor_locator(ticker.NullLocator())
    b.set_xlim(0, 15)
    b.set_xticks([0, 5, 10, 15])
    b.set_xlabel("time since onset (h)")
    b.set_ylabel(r"heat flux $\Phi$ (TW)")
    b.legend(
        handlelength=1.2,
        fontsize=FS,
        loc="lower right",
        ncol=1,
        borderaxespad=0.2,
        labelspacing=0.25,
        handletextpad=0.5,
    )


def panel_c(c):
    """Bayes factors against dyke heating under each clast shape.

    Dots are the nominal priors; bars span the evidence of the same mechanism
    with its prior half-widths halved and doubled, relative to the nominal
    evidence of dyke heating under the same clast shape.
    """
    ln10 = np.log(10)
    offsets = {"blocky": -0.24, "long": 0.0, "sheet": 0.24}
    for sh, dx in offsets.items():
        ev = DS["by_shape"][sh]["evidence"]
        z0 = ev["dyke_heating"]["nominal"]["logz"]
        for i, m in enumerate(ORDER):
            v = (ev[m]["nominal"]["logz"] - z0) / ln10
            zz = [
                (ev[m][x]["logz"] - z0) / ln10
                for x in ("nominal", "half_width", "double_width")
                if "logz" in ev[m].get(x, {})
            ]
            c.plot(
                [i + dx, i + dx],
                [min(zz), max(zz)],
                "-",
                color=SCOL[sh],
                lw=0.9,
                alpha=0.8,
                zorder=2,
                solid_capstyle="butt",
            )
            c.plot(
                [i + dx],
                [v],
                "o",
                ms=3.6,
                color=SCOL[sh],
                mec="white",
                mew=0.4,
                zorder=4,
                label=sh if i == 0 else None,
            )
    c.axhline(0.0, color=OKABE_ITO["black"], lw=0.5)
    c.axhline(-2.0, color=OKABE_ITO["grey"], lw=0.7, ls=":", zorder=1)
    c.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    c.set_ylim(-8000, 60)
    c.set_yticks([-1000, -100, -10, -1, 0, 1, 10])
    c.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _p: f"{y:.0f}".replace("-", "−")))
    c.yaxis.set_minor_locator(ticker.NullLocator())
    mech_axis(c)
    c.set_ylabel(r"$\log_{10}B$ against dyke heating")
    c.legend(
        loc="lower right",
        fontsize=5.5,
        handlelength=0.8,
        borderpad=0.2,
        borderaxespad=0.2,
        labelspacing=0.2,
        frameon=False,
    )


def panel_d(d):
    sh = VS["by_shape"]
    for i, m in enumerate(ORDER):
        d.plot(
            [i],
            [DT[m]["peak_phi_W"] / 1e12],
            "o",
            ms=4.5,
            color="#4D4D4D",
            mec="white",
            mew=0.5,
            zorder=3,
        )
    for s, ls in zip(("blocky", "long", "sheet"), ("-", "--", ":"), strict=True):
        v = sh[s]["phi_at_mass_weighted_mean_W"]["median"] / 1e12
        d.axhline(v, color=SCOL[s], lw=0.9, ls=ls, label=f"{s}, {v:.2f} TW")
    d.set_yscale("log")
    d.set_ylim(1e-1, 1e4)
    d.yaxis.set_major_locator(ticker.LogLocator(numticks=7))
    d.yaxis.set_minor_locator(ticker.NullLocator())
    mech_axis(d)
    d.set_ylabel(r"peak $\Phi$ at prior ceiling (TW)")
    d.legend(
        handlelength=2.0,
        fontsize=FS,
        loc="upper left",
        borderaxespad=0.2,
        title=r"inferred $\Phi$",
        title_fontsize=FS,
        labelspacing=0.25,
        alignment="left",
    )


def panel_e(e):
    ladder = {r["key"]: r for r in HC["ladder"]}
    mp = HC["megaplume_mean_power"]
    ns = HC["nesca"]
    rows = [
        ("single black smoker", "steady", ladder["single_black_smoker"]),
        ("vent field", "steady", ladder["vent_field"]),
        ("ridge segment", "steady", ladder["ridge_segment"]),
        ("1986 megaplume", "megaplume", ladder["megaplume_ep86"]),
        (
            "megaplume heat over 1–100 h",
            "megaplume",
            {
                "low_W": mp["mean_power_W"][0],
                "high_W": mp["mean_power_W"][1],
                "inner": mp["mean_power_at_reference_W"],
            },
        ),
        (
            "NESCA, sheet",
            "nesca",
            {
                "low_W": ns["sheet"]["ci95_W"][0],
                "high_W": ns["sheet"]["ci95_W"][1],
                "value_W": ns["sheet"]["median_W"],
            },
        ),
        (
            "NESCA, blocky",
            "nesca",
            {
                "low_W": ns["blocky"]["ci95_W"][0],
                "high_W": ns["blocky"]["ci95_W"][1],
                "value_W": ns["blocky"]["median_W"],
            },
        ),
        ("global ridge axis", "global", ladder["global_ridge_axis"]),
        ("global hydrothermal", "global", ladder["global_hydrothermal"]),
        ("ocean floor", "global", ladder["ocean_floor"]),
        ("whole Earth", "global", ladder["whole_earth"]),
    ]
    lo_n = min(ns["sheet"]["ci95_W"][0], ns["blocky"]["ci95_W"][0])
    hi_n = max(ns["sheet"]["ci95_W"][1], ns["blocky"]["ci95_W"][1])
    e.axvspan(lo_n, hi_n, color=OKABE_ITO["black"], alpha=0.07, lw=0, zorder=0)
    for xv in (1e6, 1e9, 1e12):
        e.axvline(xv, color="#BDBDBD", lw=0.4, ls=(0, (2, 2)), zorder=0)
    for y, (_lab, grp, r) in enumerate(rows):
        col = GROUP_COL[grp]
        lo, hi, val = r.get("low_W"), r.get("high_W"), r.get("value_W")
        if grp == "nesca":
            e.plot([lo, hi], [y, y], "-", color=col, lw=1.0, zorder=3)
            e.plot([lo, lo], [y - 0.18, y + 0.18], "-", color=col, lw=0.8, zorder=3)
            e.plot([hi, hi], [y - 0.18, y + 0.18], "-", color=col, lw=0.8, zorder=3)
            e.plot([val], [y], "o", ms=4.5, color=col, zorder=4)
            continue
        if lo is not None and hi is not None and hi > lo:
            e.plot(
                [lo, hi],
                [y, y],
                "-",
                color=col,
                lw=4.0,
                alpha=0.45,
                solid_capstyle="butt",
                zorder=2,
            )
        if "inner" in r:
            e.plot(r["inner"], [y, y], "-", color=col, lw=4.0, solid_capstyle="butt", zorder=3)
        if val is not None:
            e.plot([val], [y], "o", ms=3.8, color=col, mec="white", mew=0.5, zorder=4)
    e.set_xscale("log")
    e.set_xlim(3e4, 3e14)
    e.xaxis.set_major_locator(ticker.LogLocator(numticks=12))
    e.xaxis.set_minor_locator(ticker.NullLocator())
    e.set_ylim(-0.7, len(rows) - 0.3)
    e.set_yticks(np.arange(len(rows)))
    e.set_yticklabels([r[0] for r in rows], fontsize=6)
    for tl, (_lab, grp, _r) in zip(e.get_yticklabels(), rows, strict=True):
        if grp == "nesca":
            tl.set_fontweight("bold")
    e.tick_params(axis="y", length=0)
    e.set_xlabel("heat flux (W)")
    top = e.secondary_xaxis("top")
    top.set_xticks([1e6, 1e9, 1e12])
    top.set_xticklabels(["1 MW", "1 GW", "1 TW"], fontsize=FS)
    top.tick_params(length=0, pad=1.5)
    top.spines["top"].set_visible(False)


NORM = mcolors.Normalize(vmin=-2.0, vmax=2.0)
LEVELS = np.arange(-2.0, 2.01, 0.25)
CMAP = plt.get_cmap("RdBu_r")


def energy_map(ax, x, y, e_j, floor_j, ceil_j):
    z = np.log10(np.asarray(e_j) / floor_j)
    cf = ax.contourf(x, y, z, levels=LEVELS, cmap=CMAP, norm=NORM, extend="both")
    ax.contour(x, y, z, levels=[0.0], colors="black", linewidths=1.4)
    ax.contour(
        x,
        y,
        z,
        levels=[np.log10(ceil_j / floor_j)],
        colors="black",
        linewidths=0.8,
        linestyles="--",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ticker.FuncFormatter(lambda v, _p: f"{v:g}"))
        axis.set_minor_formatter(ticker.NullFormatter())
    return cf


def panel_f(f):
    lm = HC["lava_map"]
    x, y = np.array(lm["duration_h"]), np.array(lm["area_km2"])
    cf = energy_map(f, x, y, lm["energy_J"], lm["floor_J"], lm["ceiling_J"])
    a0, t0 = lm["mapped_area_km2"], lm["reference_duration_h"]
    f.axhline(a0, color="black", lw=0.6, ls=(0, (1, 1.5)))
    f.axvline(t0, color="black", lw=0.6, ls=(0, (1, 1.5)))
    a_need = lm["area_needed_at_reference_duration_km2"]
    f.annotate(
        "",
        xy=(t0, a_need),
        xytext=(t0, a0),
        arrowprops=dict(
            arrowstyle="-|>", lw=0.8, color="black", mutation_scale=6, shrinkA=2.5, shrinkB=0
        ),
    )
    f.plot([t0], [a0], "o", ms=5, mfc="white", mec="black", mew=0.8, zorder=5)
    f.plot(
        [lm["crossing_duration_h_at_mapped_area"]],
        [a0],
        "D",
        ms=4,
        mfc="black",
        mec="white",
        mew=0.6,
        zorder=5,
    )
    f.set_xlim(x[0], x[-1])
    f.set_ylim(y[0], y[-1])
    f.set_xlabel("eruption duration (h)")
    f.set_ylabel(r"lava flow area (km$^2$)")
    return cf


def panel_g(g):
    dm = HC["dyke_map"]
    x, y = np.array(dm["length_km"]), np.array(dm["height_km"])
    energy_map(g, x, y, dm["energy_J"], HC["lava_map"]["floor_J"], HC["lava_map"]["ceiling_J"])
    pd = dm["plausible_dyke"]
    lx, hy = pd["length_m"] / 1e3, pd["height_m"] / 1e3
    g.annotate(
        "",
        xy=(lx, pd["height_needed_at_this_length_m"] / 1e3),
        xytext=(lx, hy),
        arrowprops=dict(
            arrowstyle="-|>", lw=0.8, color="black", mutation_scale=6, shrinkA=2.5, shrinkB=0
        ),
    )
    g.plot([lx], [hy], "o", ms=5, mfc="white", mec="black", mew=0.8, zorder=5)
    g.plot(
        [x[-1]], [y[-1]], "s", ms=4.5, mfc="white", mec="black", mew=0.8, zorder=6, clip_on=False
    )
    g.set_xlim(x[0], x[-1])
    g.set_ylim(y[0], y[-1])
    g.set_xticks([1, 3, 10, 30])
    g.set_yticks([0.1, 0.3, 1, 3])
    g.set_xlabel("dyke length (km)")
    g.set_ylabel("dyke height (km)")


def main() -> int:
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    assert abs(W_MM * MM - TWO_COL) < 1e-9

    # row 1: a, b, c
    r1, h1 = 4.0, 37.0
    a = ax_mm(fig, 14, r1, 42, h1)
    b = ax_mm(fig, 74, r1, 42, h1)
    c = ax_mm(fig, 136, r1, 42, h1)
    panel_a(a)
    panel_b(b)
    panel_c(c)
    label_mm(fig, 1, r1 - 3, "a")
    label_mm(fig, 61, r1 - 3, "b")
    label_mm(fig, 121, r1 - 3, "c")

    # row 2: d, e
    r2, h2 = 58.0, 44.0
    d = ax_mm(fig, 14, r2, 42, h2)
    e = ax_mm(fig, 101, r2, 77, h2)
    panel_d(d)
    panel_e(e)
    label_mm(fig, 1, r2 - 3, "d")
    label_mm(fig, 61, r2 - 3, "e")

    # row 3: f, g and the shared colour scale
    r3, h3 = 116.0, 41.0
    f = ax_mm(fig, 14, r3, 60, h3)
    g = ax_mm(fig, 92, r3, 60, h3)
    cf = panel_f(f)
    panel_g(g)
    label_mm(fig, 1, r3 - 3, "f")
    label_mm(fig, 79, r3 - 3, "g")
    cax = ax_mm(fig, 157, r3, 2.6, h3)
    cb = fig.colorbar(cf, cax=cax, ticks=[-2, -1, 0, 1, 2])
    cb.ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"], fontsize=FS)
    cb.ax.tick_params(length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    cb.add_lines([0.0], colors=["black"], linewidths=[1.4])
    cb.add_lines(
        [np.log10(HC["lava_map"]["ceiling_J"] / HC["lava_map"]["floor_J"])],
        colors=["black"],
        linewidths=[0.8],
    )
    cb.set_label("heat delivered / megaplume floor", fontsize=6, labelpad=2)

    fig.savefig(ROOT / "figures" / "fig_sources.pdf")
    fig.savefig(ROOT / "figures" / "fig_sources.png", dpi=300)
    print("wrote figures/fig_sources.pdf and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
