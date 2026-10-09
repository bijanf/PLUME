"""Figure: what a polydisperse deposit says about the clast shape.

(a) settling speed against diameter for the five shape classes, sieve windows
(b) per-fraction dispersal length scales against diameter, class predictions
(c) bootstrap distribution of the extreme length-scale ratio, class predictions
(d) delta chi2 over the (C1, C2) plane, both tests
(e) heat flux over the (C1, C2) plane
(f) relative likelihood of the profile test over shape space, coloured by heat flux
(g) confusion matrices of the two tests on synthetic NESCA deposits
(h) probability of correct classification against noise and number of cores
(i) probability of correct classification against number of sieve fractions

Every number is read from results/shape_space.json.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from plume_style import MM, OKABE_ITO, TWO_COL, apply_style  # noqa: E402

apply_style()
import matplotlib.pyplot as plt  # noqa: E402

# Mathematical symbols in the same sans-serif face as the text.
plt.rcParams.update(
    {
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
        "mathtext.sf": "Helvetica",
    }
)
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

R = json.loads((ROOT / "results" / "shape_space.json").read_text())
SHAPES = ["spheres", "quartz_sand", "blocky", "long", "sheet"]
VOLC = ["blocky", "long", "sheet"]
SCOL = {
    "blocky": OKABE_ITO["blue"],
    "long": OKABE_ITO["green"],
    "sheet": OKABE_ITO["vermillion"],
    "spheres": OKABE_ITO["purple"],
    "quartz_sand": OKABE_ITO["orange"],
}
SLAB = {
    "spheres": "spheres",
    "quartz_sand": "quartz sand",
    "blocky": "blocky",
    "long": "long",
    "sheet": "sheet",
}
SMARK = {"spheres": "o", "quartz_sand": "s", "blocky": "D", "long": "^", "sheet": "v"}
TW = 1e12


def label(ax, text, x=-0.2, y=1.06):
    put = ax.text2D if hasattr(ax, "text2D") else ax.text
    put(x, y, text, transform=ax.transAxes, fontsize=7, fontweight="bold", va="bottom", ha="left")


def class_markers(ax, ms=4.2, zorder=6):
    for s in SHAPES:
        c1 = R["length_scale_test"]["continuous"]["named_classes"][s]["c1"]
        c2 = R["length_scale_test"]["continuous"]["named_classes"][s]["c2"]
        ax.plot(c1, c2, SMARK[s], ms=ms, mfc=SCOL[s], mec="white", mew=0.6, zorder=zorder)


def log_axis(ax, which, ticks, labels):
    getattr(ax, f"set_{which}scale")("log")
    getattr(ax, f"set_{which}ticks")(ticks)
    getattr(ax, f"set_{which}ticklabels")(labels)
    getattr(ax, f"{which}axis").set_minor_formatter(NullFormatter())


def colourbar(fig, mappable, ax, ticks, labels, text):
    cb = fig.colorbar(mappable, ax=ax, pad=0.03, fraction=0.07, aspect=22, ticks=ticks)
    cb.ax.set_yticklabels(labels)
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    cb.set_label(text, fontsize=6, labelpad=2)
    cb.outline.set_linewidth(0.4)
    return cb


def main() -> int:
    fig = plt.figure(figsize=(TWO_COL, 190 * MM))
    outer = fig.add_gridspec(3, 1, left=0.07, right=0.985, bottom=0.05, top=0.965, hspace=0.42)
    row1 = outer[0].subgridspec(1, 3, wspace=0.45)
    row2 = outer[1].subgridspec(1, 3, wspace=0.42, width_ratios=[1, 1, 1.05])
    row3 = outer[2].subgridspec(1, 3, wspace=0.40, width_ratios=[1.38, 1.0, 0.92])

    # ---------------------------------------------------------------- a
    a = fig.add_subplot(row1[0, 0])
    sc = R["settling_curves"]
    d_um = np.asarray(sc["diameter_m"]) * 1e6
    win = np.asarray(R["real_data"]["sieve_windows_m"]) * 1e6
    for i, (lo, hi) in enumerate(win):
        a.axvspan(lo, hi, color=OKABE_ITO["grey"], alpha=0.10 if i % 2 == 0 else 0.20, lw=0)
    handles = []
    for s in SHAPES:
        a.plot(d_um, np.asarray(sc["w_ms"][s]) * 100, color=SCOL[s], lw=1.0)
        ds = sc["transition_diameter_um"][s]
        ws = np.interp(np.log(ds), np.log(d_um), np.log(np.asarray(sc["w_ms"][s]) * 100))
        a.plot(ds, np.exp(ws), SMARK[s], ms=3.4, mfc="white", mec=SCOL[s], mew=0.8, zorder=5)
        handles.append(
            Line2D(
                [],
                [],
                color=SCOL[s],
                lw=1.0,
                marker=SMARK[s],
                ms=3.4,
                mfc=SCOL[s],
                mec="white",
                mew=0.5,
                label=SLAB[s],
            )
        )
    a.set_xlim(40, 1600)
    a.set_ylim(0.1, 40)
    log_axis(a, "x", [100, 300, 1000], ["100", "300", "1000"])
    log_axis(a, "y", [0.1, 1, 10], ["0.1", "1", "10"])
    a.set_xlabel(r"diameter $D$ ($\mu$m)")
    a.set_ylabel(r"settling speed $w$ (cm s$^{-1}$)")
    for (lo, hi), txt in zip(win, ["63–\n125", "125–\n250", "250–\n500", ">500"], strict=True):
        a.text(
            np.sqrt(lo * hi),
            32,
            txt,
            fontsize=5,
            ha="center",
            va="top",
            color="0.30",
            linespacing=0.9,
        )
    a.legend(
        handles=handles,
        loc="lower right",
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.25,
        fontsize=5.5,
    )
    label(a, "a")

    # ---------------------------------------------------------------- b
    b = fig.add_subplot(row1[0, 1])
    mids = np.asarray(R["real_data"]["sieve_midpoints_m"]) * 1e6
    Lobs = np.asarray(R["real_data"]["length_scales_m"]) / 1e3
    Lci = np.asarray(R["real_data"]["length_scale_ci95_m"]) / 1e3
    cont = R["length_scale_test"]["continuous"]["named_classes"]
    dd = np.geomspace(50, 1200, 100)
    for s in SHAPES:
        w = np.exp(np.interp(np.log(dd), np.log(d_um), np.log(sc["w_ms"][s])))
        Lp = np.sqrt(sc["q_hat_m3s"][s] / w) / 1e3
        b.plot(dd, Lp, color=SCOL[s], lw=0.9, label=rf"{SLAB[s]}, $\chi^2$ {cont[s]['chi2']:.1f}")
    b.errorbar(
        mids,
        Lobs,
        yerr=[Lobs - Lci[:, 0], Lci[:, 1] - Lobs],
        fmt="o",
        ms=3.2,
        color="black",
        mfc="white",
        mew=0.8,
        elinewidth=0.7,
        capsize=1.5,
        zorder=6,
        label="fitted $L$, 95% interval",
    )
    b.set_xlim(50, 1200)
    b.set_ylim(2.5, 70)
    log_axis(b, "x", [100, 300, 1000], ["100", "300", "1000"])
    log_axis(b, "y", [3, 10, 30], ["3", "10", "30"])
    b.set_xlabel(r"sieve midpoint $D$ ($\mu$m)")
    b.set_ylabel(r"length scale $L$ (km)")
    b.legend(loc="upper right", fontsize=5, handlelength=1.3, borderpad=0.2, labelspacing=0.2)
    label(b, "b")

    # ---------------------------------------------------------------- c
    c = fig.add_subplot(row1[0, 2])
    bt = R["bootstrap"]["extreme_ratio"]
    e = np.asarray(bt["hist_edges"])
    cnt = np.asarray(bt["hist_counts"], dtype=float)
    dens = cnt / cnt.sum() / np.diff(np.log(e))
    c.stairs(dens, e, fill=True, color=OKABE_ITO["grey"], alpha=0.40, lw=0)
    c.stairs(dens, e, color="0.35", lw=0.5)
    ymax = dens.max() * 1.6
    for s in SHAPES:
        c.plot([bt["predicted_by_shape"][s]] * 2, [0, ymax * 0.74], color=SCOL[s], lw=1.0)
    for s in VOLC:
        rr = R["synthetic"]["noise_free"]["nesca_nu"][s]["extreme_ratio"]
        c.plot([rr] * 2, [0, ymax * 0.58], color=SCOL[s], lw=1.1, ls=(0, (1.0, 1.0)))
    c.plot([bt["observed"]] * 2, [0, ymax * 0.74], color="black", lw=1.3)
    c.set_xlim(2.0, 20)
    c.set_ylim(0, ymax)
    log_axis(c, "x", [2, 3, 4, 5, 7, 10, 20], ["2", "3", "4", "5", "7", "10", "20"])
    c.set_xlabel(r"extreme ratio $L_{63\text{–}125}\,/\,L_{>500}$")
    c.set_ylabel("bootstrap density")
    c.legend(
        handles=[
            Line2D([], [], color="black", lw=1.3, label="observed"),
            Line2D([], [], color="0.3", lw=1.0, label="predicted, single flux"),
            Line2D(
                [],
                [],
                color="0.3",
                lw=1.1,
                ls=(0, (1.0, 1.0)),
                label="predicted, NESCA flux spread",
            ),
        ],
        loc="upper right",
        fontsize=5,
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.25,
    )
    label(c, "c")

    # ---------------------------------------------------------------- d
    lc = R["length_scale_test"]["continuous"]
    c1 = np.asarray(lc["c1"])
    c2 = np.asarray(lc["c2"])
    C1, C2 = np.meshgrid(c1, c2, indexing="ij")
    dchi = np.asarray(lc["delta_chi2"], dtype=float)
    pc = R["profile_test"]["continuous"]
    dchi_p = np.asarray(pc["delta_chi2_on_grid"], dtype=float)
    c2_ticks = ([0.3, 1, 3, 10, 30, 100], ["0.3", "1", "3", "10", "30", "100"])
    d = fig.add_subplot(row2[0, 0])
    lev = [0, 1, 2.3, 4, 6.18, 9, 12, 16, 25]
    cf = d.contourf(
        C1,
        C2,
        np.clip(dchi, 0, 24.9),
        levels=lev,
        cmap="cividis",
        norm=mcolors.BoundaryNorm(lev, 256),
    )
    d.contour(C1, C2, dchi, levels=[2.30, 6.18], colors="white", linewidths=[1.0, 0.6])
    d.contour(
        C1,
        C2,
        dchi_p,
        levels=[2.30, 6.18],
        colors=OKABE_ITO["orange"],
        linewidths=[1.0, 0.6],
        linestyles="--",
    )
    class_markers(d)
    log_axis(d, "y", *c2_ticks)
    d.set_xlabel(r"$C_1$")
    d.set_ylabel(r"$C_2$")
    colourbar(
        fig,
        cf,
        d,
        [0, 2.3, 6.18, 12, 25],
        ["0", "2.3", "6.2", "12", "25"],
        r"$\Delta\chi^2$, length-scale test",
    )
    d.legend(
        handles=[
            Line2D([], [], color="0.35", lw=1.0, label="length-scale test"),
            Line2D([], [], color=OKABE_ITO["orange"], lw=1.0, ls="--", label="profile test"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.08, 1.0),
        ncol=2,
        fontsize=5,
        handlelength=1.8,
        borderpad=0.1,
        columnspacing=0.8,
        borderaxespad=0.1,
    )
    label(d, "d")

    # ---------------------------------------------------------------- e
    ee = fig.add_subplot(row2[0, 1])
    phi = np.asarray(lc["phi_W"], dtype=float) / TW
    plev = np.geomspace(0.2, 5.0, 15)
    pnorm = mcolors.LogNorm(plev[0], plev[-1])
    cf2 = ee.contourf(
        C1, C2, np.clip(phi, plev[0], plev[-1] * 0.999), levels=plev, cmap="inferno", norm=pnorm
    )
    ee.contour(C1, C2, dchi, levels=[6.18], colors="white", linewidths=0.9)
    ee.contour(
        C1, C2, dchi_p, levels=[6.18], colors=OKABE_ITO["sky"], linewidths=0.9, linestyles="--"
    )
    class_markers(ee)
    log_axis(ee, "y", *c2_ticks)
    ee.set_xlabel(r"$C_1$")
    ee.set_ylabel(r"$C_2$")
    colourbar(
        fig, cf2, ee, [0.2, 0.5, 1, 2, 5], ["0.2", "0.5", "1", "2", "5"], r"heat flux $\Phi$ (TW)"
    )
    label(ee, "e")

    # ---------------------------------------------------------------- f
    f = fig.add_subplot(row2[0, 2], projection="3d", computed_zorder=False)
    sub = (slice(None, None, 3), slice(None, None, 3))
    X = C1[sub]
    Y = np.log10(C2[sub])
    like = np.exp(-0.5 * np.nan_to_num(dchi_p, nan=50.0))
    Z = like[sub]
    fc = plt.get_cmap("inferno")(pnorm(np.clip(phi[sub], plev[0], plev[-1])))
    f.plot_surface(
        X,
        Y,
        Z,
        facecolors=fc,
        rstride=1,
        cstride=1,
        linewidth=0.08,
        edgecolor=(1, 1, 1, 0.35),
        antialiased=True,
        shade=False,
        zorder=1,
    )
    kl = R["length_scale_test"]["continuous"]["best_fit"]["c1sq_over_c2"]
    cl = np.linspace(c1[0], c1[-1], 60)
    yl = np.log10(cl**2 / kl)
    okl = (yl >= np.log10(c2[0])) & (yl <= np.log10(c2[-1]))
    f.plot(cl[okl], yl[okl], np.zeros(okl.sum()), color="black", lw=0.9, ls="--", zorder=3)
    f.view_init(elev=24, azim=-124)
    f.set_xlabel(r"$C_1$", labelpad=-8, fontsize=6)
    f.set_ylabel(r"log$_{10}C_2$", labelpad=-7, fontsize=6)
    f.zaxis.set_rotate_label(False)
    f.set_zlabel("relative\nlikelihood", labelpad=-5, fontsize=6, rotation=90, va="center")
    f.set_xticks([20, 30, 40])
    f.set_yticks([0, 1, 2])
    f.set_zticks([0, 0.5, 1])
    f.set_zlim(0, 1)
    f.tick_params(axis="x", labelsize=5, pad=-3)
    f.tick_params(axis="y", labelsize=5, pad=-3)
    f.tick_params(axis="z", labelsize=5, pad=-1.5)
    for ax_ in (f.xaxis, f.yaxis, f.zaxis):
        ax_.pane.set_facecolor((1, 1, 1, 0))
        ax_.pane.set_edgecolor("0.75")
        ax_._axinfo["grid"]["linewidth"] = 0.3
        ax_._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)
    f.set_box_aspect((1.15, 1.15, 0.8), zoom=1.0)

    # ---------------------------------------------------------------- g
    gg = row3[0, 0].subgridspec(1, 2, wspace=0.10)
    ms = R["synthetic"]["main"]["nesca_nu"]["sigma_residual"]
    for j, (key, ttl) in enumerate(
        (("length_scale_test", "length-scale test"), ("profile_test", "profile test"))
    ):
        g = fig.add_subplot(gg[0, j])
        m = np.asarray(ms[key]["volcaniclast_candidates"]["confusion_fraction"], dtype=float)
        g.imshow(m, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        for r_ in range(3):
            for c_ in range(3):
                g.text(
                    c_,
                    r_,
                    f"{m[r_, c_]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if m[r_, c_] > 0.55 else "black",
                )
        g.set_xticks(range(3))
        g.set_xticklabels(["blocky", "long", "sheet"], fontsize=5.5, rotation=0)
        g.set_yticks(range(3))
        g.set_yticklabels(["blocky", "long", "sheet"] if j == 0 else [], fontsize=5.5)
        g.tick_params(length=0, pad=1.5)
        for sp in g.spines.values():
            sp.set_visible(False)
        g.set_title(ttl, fontsize=6, pad=3)
        g.set_xlabel("assigned class", fontsize=6, labelpad=2)
        if j == 0:
            g.set_ylabel("true class", fontsize=6, labelpad=2)
            g_first = g

    # ---------------------------------------------------------------- h
    h = fig.add_subplot(row3[0, 1])
    pw = R["synthetic"]["power"]
    sig = np.asarray(pw["sigma_log"])
    nco = np.asarray(pw["n_cores"])
    NC, SG = np.meshgrid(nco, sig)
    P = np.asarray(pw["profile_test_nesca_nu"]["p_correct"], dtype=float)
    Pl = np.asarray(pw["length_scale_test_steady"]["p_correct"], dtype=float)
    plv = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    cf4 = h.contourf(
        NC,
        SG,
        np.clip(P, 0.3, 0.9999),
        levels=plv,
        cmap="viridis",
        norm=mcolors.BoundaryNorm(plv, 256),
    )
    h.contour(NC, SG, P, levels=[0.95], colors="white", linewidths=1.1)
    h.contour(NC, SG, Pl, levels=[0.95], colors="white", linewidths=0.8, linestyles="--")
    for sv, mk, ms_ in (
        (R["synthetic"]["setup"]["sigma_residual"], "*", 6.0),
        (R["synthetic"]["setup"]["sigma_nugget"], "P", 4.8),
    ):
        h.plot(
            R["synthetic"]["setup"]["n_cores"],
            sv,
            mk,
            ms=ms_,
            mfc="white",
            mec="black",
            mew=0.5,
            zorder=6,
        )
    h.set_xlim(nco[0], nco[-1])
    log_axis(h, "x", [10, 30, 100, 300, 1000], ["10", "30", "100", "300", "1000"])
    h.set_xlabel("number of cores")
    h.set_ylabel(r"noise $\sigma_{\log}$")
    colourbar(
        fig,
        cf4,
        h,
        [0.3, 0.5, 0.7, 0.9, 1.0],
        ["0.3", "0.5", "0.7", "0.9", "1"],
        "P(correct), profile test",
    )
    label(h, "h")

    # ---------------------------------------------------------------- i
    ii = fig.add_subplot(row3[0, 2])
    bf = pw["by_n_fractions_nesca_nu"]
    k = np.asarray(pw["n_fractions"])
    series = (
        ("profile_test_p_correct", OKABE_ITO["orange"], "-", "o", "profile test"),
        (
            "length_scale_test_steady_p_correct",
            OKABE_ITO["black"],
            "--",
            "s",
            "length-scale test, single flux",
        ),
        (
            "length_scale_test_p_correct",
            OKABE_ITO["grey"],
            ":",
            "^",
            "length-scale test, NESCA spread",
        ),
    )
    for sv, alpha in ((0.3, 0.5), (1.0, 1.0)):
        a_ = int(np.argmin(np.abs(sig - sv)))
        for key, col, ls, mk, _lab in series:
            if key == "length_scale_test_p_correct" and sv != 1.0:
                continue
            y = np.asarray(bf[key])[a_]
            ii.plot(k, y, ls=ls, marker=mk, ms=2.6, lw=1.0, color=col, alpha=alpha)
        yp = np.asarray(bf["profile_test_p_correct"])[a_][-1]
        ys = np.asarray(bf["length_scale_test_steady_p_correct"])[a_][-1]
        ii.text(
            4.12,
            0.5 * (yp + ys),
            rf"$\sigma$ {sv:.1f}",
            fontsize=5.5,
            va="center",
            ha="left",
            color="0.25",
        )
    ii.axhline(1 / 3, color="0.5", lw=0.6, ls=(0, (2, 2)))
    ii.text(2.5, 1 / 3 - 0.012, "chance", fontsize=5.5, color="0.4", ha="center", va="top")
    ii.set_xticks([1, 2, 3, 4])
    ii.set_xlim(0.8, 4.65)
    ii.set_ylim(0.2, 1.06)
    ii.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ii.spines["left"].set_bounds(0.2, 1.0)
    ii.spines["bottom"].set_bounds(1, 4)
    ii.set_xlabel("number of sieve fractions")
    ii.set_ylabel("P(correct)")
    ii.legend(
        handles=[
            Line2D([], [], color=col, ls=ls, marker=mk, ms=2.6, lw=1.0, label=lab)
            for _k, col, ls, mk, lab in series
        ],
        loc="upper left",
        fontsize=5,
        handlelength=2.0,
        borderpad=0.2,
        labelspacing=0.25,
    )
    label(ii, "i")

    # Labels of the 3D panel and of the matrices, aligned with their rows.
    fig.canvas.draw()
    pe, ph = ee.get_position(), h.get_position()
    pf, pg = f.get_position(), g_first.get_position()
    ytop = lambda p_: p_.y1 + 0.06 * p_.height  # noqa: E731
    fig.text(pf.x0 + 0.01, ytop(pe), "f", fontsize=7, fontweight="bold", va="bottom", ha="left")
    fig.text(pg.x0 - 0.055, ytop(ph), "g", fontsize=7, fontweight="bold", va="bottom", ha="left")

    out = ROOT / "figures" / "fig_shape.pdf"
    fig.savefig(out)
    fig.savefig(ROOT / "figures" / "fig_shape.png", dpi=300)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
