"""Figure: what the NESCA deposit resolves, and what the resolvable modes look like.

(a) singular spectra of the nu-operator for 1-4 sieve fractions
(b) discrete Picard plot on the real data
(c) resolvable modes against the number of fractions, on two criteria
(d) the front constraint: linearised spectra with and without it
(e) right singular vectors v_n(Q), n = 1..10, as a filled-contour map
(f) the leading six v_n(log10 Q) as a 3D waterfall
(g) deposit pattern of modes 1-4 over radius and settling speed
(h) model resolution R = V_k V_k^T, one fraction against four
(i) signal-to-noise mode count over observation noise and fraction count

Numbers come from results/modes.json (a-c, e-i) and results/identifiability.json (d).
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from plume_style import OKABE_ITO, TWO_COL, apply_style  # noqa: E402

apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

MM = 1.0 / 25.4
D = json.loads((ROOT / "results" / "identifiability.json").read_text())
M = json.loads((ROOT / "results" / "modes.json").read_text())
COLS = [OKABE_ITO["sky"], OKABE_ITO["green"], OKABE_ITO["blue"], OKABE_ITO["vermillion"]]
DIV = plt.get_cmap("RdBu_r")
AMP_LEVELS = np.linspace(-1.0, 1.0, 21)
LABEL_PT = 7  # panel labels: bold lowercase, at the 7 pt maximum
SMALL = 5  # smallest text used


LABELS: list = []


def label(ax, text, dx_mm: float = 9.0, dy_mm: float = 1.5):
    """Queue a bold lowercase panel label, placed in figure coordinates at a fixed
    distance (mm) left of and above the axes once the layout is final."""
    LABELS.append((ax, text, dx_mm, dy_mm))


def place_labels(fig):
    w_mm, h_mm = fig.get_size_inches() / MM
    for ax, text, dx, dy in LABELS:
        pos = ax.get_position()
        fig.text(
            max(pos.x0 - dx / w_mm, 0.002),
            pos.y1 + dy / h_mm,
            text,
            fontsize=LABEL_PT,
            fontweight="bold",
            va="bottom",
            ha="left",
        )


def small_axes(ax):
    ax.tick_params(labelsize=SMALL + 0.5, pad=1.5)
    ax.xaxis.label.set_size(6)
    ax.yaxis.label.set_size(6)
    ax.xaxis.labelpad = 1.5
    ax.yaxis.labelpad = 1.5


# --------------------------------------------------------------------------
def panel_a(ax):
    ver = M["verification"]["by_fraction_count"]
    for i, k in enumerate(("k=1", "k=2", "k=3", "k=4")):
        s = np.array(ver[k]["singular_values_first_20"][:10])
        ax.semilogy(
            np.arange(1, s.size + 1),
            s / s[0],
            "o-",
            ms=2.0,
            lw=0.9,
            color=COLS[i],
            label=f"{i + 1}",
        )
    ax.set_xlabel("mode $n$")
    ax.set_ylabel(r"$\sigma_n/\sigma_1$")
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks([1, 4, 7, 10])
    ax.set_ylim(5e-4, 2)
    ax.legend(
        title="fractions",
        title_fontsize=SMALL,
        fontsize=SMALL,
        handlelength=1.2,
        loc="lower left",
        labelspacing=0.2,
        borderaxespad=0.2,
    )
    small_axes(ax)
    label(ax, "a")


def panel_b(ax):
    pic = M["picard_plot"]
    s = np.array(pic["singular_values_over_s1"])
    c = np.array(pic["coefficients_abs_uT_d"])
    n = np.arange(1, s.size + 1)
    ax.semilogy(n, s, "-", lw=1.0, color=OKABE_ITO["black"], label=r"$\sigma_n/\sigma_1$")
    ax.semilogy(n, c, "o", ms=1.6, color=OKABE_ITO["vermillion"], label=r"$|u_n^{T}\tilde d|$")
    ax.axhline(1.0, color=OKABE_ITO["grey"], lw=0.6, ls=":", label="noise level")
    k = pic["truncation_index"]
    ax.axvline(k + 0.5, color=OKABE_ITO["purple"], lw=0.8, ls="--", label="Picard cut")
    ax.set_xlabel("mode $n$")
    ax.set_ylabel("magnitude")
    ax.set_xlim(0.5, s.size + 0.5)
    ax.set_xticks([1, 10, 20, 30, 40])
    ax.set_ylim(1e-6, 1e4)
    ax.legend(
        fontsize=SMALL,
        handlelength=1.4,
        loc="upper right",
        ncol=2,
        columnspacing=0.8,
        labelspacing=0.2,
        borderaxespad=0.1,
    )
    small_axes(ax)
    label(ax, "b")


def panel_c(ax):
    ver = M["verification"]["by_fraction_count"]
    ks = np.arange(1, 5)
    snr = [ver[f"k={k}"]["N_res_snr"] for k in ks]
    pic = [ver[f"k={k}"]["N_res_picard"] for k in ks]
    w = 0.36
    ax.bar(ks - w / 2, snr, width=w, color=OKABE_ITO["blue"], label="signal-to-noise")
    ax.bar(ks + w / 2, pic, width=w, color=OKABE_ITO["orange"], label="Picard")
    for k, v1, v2 in zip(ks, snr, pic, strict=True):
        ax.text(k - w / 2, v1 + 0.12, str(v1), ha="center", va="bottom", fontsize=SMALL)
        ax.text(k + w / 2, v2 + 0.12, str(v2), ha="center", va="bottom", fontsize=SMALL)
    ax.set_xlabel("sieve fractions")
    ax.set_ylabel(r"resolvable modes of $\nu(Q)$")
    ax.set_xticks(ks)
    ax.set_ylim(0, max(snr) + 2.6)
    ax.set_yticks(np.arange(0, max(snr) + 3, 5))
    ax.legend(
        fontsize=SMALL, handlelength=1.0, loc="upper left", labelspacing=0.2, borderaxespad=0.1
    )
    small_axes(ax)
    label(ax, "c")


def panel_d(ax):
    fc = D["front_constraint"]["waning_Q"]
    for kind, col, lab in (
        ("quasi_steady", OKABE_ITO["grey"], "quasi-steady"),
        ("front_limited", OKABE_ITO["purple"], "front-limited"),
    ):
        s2 = np.array(fc[kind]["singular_values"])
        ax.semilogy(
            np.arange(1, s2.size + 1), s2 / s2[0], "o-", ms=2.0, lw=0.9, color=col, label=lab
        )
    ax.axhline(1e-2, color=OKABE_ITO["grey"], lw=0.6, ls=":", label="1 % level")
    ax.set_xlabel(r"mode $n$ of $\log Q(t)$")
    ax.set_ylabel(r"$\sigma_n/\sigma_1$")
    ax.set_xticks([1, 4, 8, 12])
    ax.set_ylim(1e-6, 2)
    ax.legend(
        fontsize=SMALL, handlelength=1.4, loc="lower left", labelspacing=0.2, borderaxespad=0.1
    )
    small_axes(ax)
    label(ax, "d")


# --------------------------------------------------------------------------
def modes_normalised():
    rsv = M["right_singular_vectors"]
    x = np.array(rsv["log10_q_nodes"])
    v = np.array(rsv["v_four_fractions"])
    return x, v / np.max(np.abs(v), axis=1, keepdims=True)


def panel_e(ax, cax):
    x, v = modes_normalised()
    rsv = M["right_singular_vectors"]
    qm = M["q_mass_weighted_mean_m3s"]["log10"]
    cs = None
    for n in range(v.shape[0]):
        z = np.vstack([v[n], v[n]])
        cs = ax.contourf(x, [n + 0.5, n + 1.5], z, levels=AMP_LEVELS, cmap=DIV, extend="neither")
    for n in range(1, v.shape[0]):
        ax.axhline(n + 0.5, color="white", lw=0.4)
    k_snr = rsv["signal_to_noise_truncation"]
    k_pic = rsv["picard_truncation"]
    for k, style in ((k_pic, (0, (3, 1.5))), (k_snr, "-")):
        ax.axhline(k + 0.5, color="black", lw=0.9, ls=style)
    ax.axvline(qm, color="black", lw=0.8, ls=(0, (1, 1)))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0.5, v.shape[0] + 0.5)
    ax.set_yticks([1, 2, 5, 10, 15, 19] if v.shape[0] >= 19 else np.arange(1, v.shape[0] + 1))
    ax.set_xlabel(r"$\log_{10} Q$ (m$^3$ s$^{-1}$)")
    ax.set_ylabel("mode $n$")
    ax.tick_params(axis="y", length=0)
    # Cut-off and mean-flux labels sit outside the data area.
    ax.text(
        1.012,
        k_pic / v.shape[0],
        "Picard",
        transform=ax.transAxes,
        fontsize=SMALL,
        va="center",
        ha="left",
    )
    if k_snr < v.shape[0]:
        ax.text(
            1.012,
            k_snr / v.shape[0],
            "S/N",
            transform=ax.transAxes,
            fontsize=SMALL,
            va="center",
            ha="left",
        )
    ax.text(
        qm, v.shape[0] + 0.6, r"$\langle Q\rangle$", fontsize=SMALL + 0.5, ha="center", va="bottom"
    )
    for s in ax.spines.values():
        s.set_visible(True)
    small_axes(ax)
    cb = plt.colorbar(cs, cax=cax, ticks=[-1, 0, 1])
    cb.set_label("normalised amplitude", fontsize=SMALL + 0.5, labelpad=1)
    cb.ax.tick_params(labelsize=SMALL + 0.5, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    label(ax, "e")


def panel_f(ax):
    x, v = modes_normalised()
    qm = M["q_mass_weighted_mean_m3s"]["log10"]
    n_show = 6
    cmap = plt.get_cmap("viridis")
    for n in range(n_show - 1, -1, -1):
        col = cmap(0.1 + 0.8 * n / (n_show - 1))
        verts = (
            [(x[0], n + 1, 0.0)]
            + [(xi, n + 1, zi) for xi, zi in zip(x, v[n], strict=True)]
            + [(x[-1], n + 1, 0.0)]
        )
        poly = Poly3DCollection([verts], facecolors=[mcolors.to_rgba(col, 0.55)], edgecolors="none")
        ax.add_collection3d(poly)
        ax.plot(x, np.full_like(x, n + 1), v[n], color=col, lw=0.9)
        ax.plot([x[0], x[-1]], [n + 1, n + 1], [0, 0], color="0.6", lw=0.3)
    ax.plot([qm, qm], [1, n_show], [-1.0, -1.0], color="black", lw=0.8, ls=(0, (1, 1)))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0.6, n_show + 0.4)
    ax.set_zlim(-1.0, 1.0)
    ax.set_xticks([3, 5, 7])
    ax.set_yticks([1, 3, 5])
    ax.set_zticks([-1, 0, 1])
    ax.view_init(elev=26, azim=-72)
    ax.set_box_aspect((1.7, 1.2, 0.8), zoom=1.18)
    ax.tick_params(labelsize=SMALL + 0.5, pad=-2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
        axis.pane.set_edgecolor("0.75")
        axis._axinfo["grid"]["linewidth"] = 0.3
        axis._axinfo["grid"]["color"] = (0.8, 0.8, 0.8, 1.0)
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.2
    ax.set_xlabel(r"$\log_{10} Q$", fontsize=6, labelpad=-7)
    ax.set_ylabel("mode $n$", fontsize=6, labelpad=-7)
    ax.set_zlabel("$v_n$", fontsize=6, labelpad=-8)
    label(ax, "f", dx_mm=-2.0, dy_mm=1.5)


G_LEVELS = np.concatenate([-np.logspace(0, -4, 9), np.logspace(-4, 0, 9)])


def panel_g(axes, cax):
    dsp = M["data_space_patterns"]
    lam_grid = np.array(dsp["lambda_grid_m3s"])
    f_modes = np.array(dsp["F_modes_1_to_6"])
    la = M["laplace_abscissa"]
    lam_lo, lam_hi = la["four_fractions"]["lambda_range_m3s"]
    per = la["per_class"]
    r_km = np.linspace(0.0, 10.0, 301)
    w = np.geomspace(1e-3, 0.3, 241)
    rr, ww = np.meshgrid(r_km * 1e3, w)
    lam = np.clip(np.pi * ww * rr**2, lam_grid[0], lam_grid[-1])
    in_rng = (lam_grid >= lam_lo) & (lam_grid <= lam_hi)
    cs = None
    for n, ax in enumerate(axes):
        f = f_modes[n]
        f = f / np.max(np.abs(f[in_rng]))
        z = np.interp(np.log(lam), np.log(lam_grid), f)
        cs = ax.contourf(
            r_km,
            w,
            z,
            levels=G_LEVELS,
            cmap=DIV,
            extend="both",
            norm=mcolors.SymLogNorm(linthresh=1e-4, vmin=-1, vmax=1, base=10),
        )
        ax.contour(r_km, w, z, levels=[0.0], colors="0.25", linewidths=0.4)
        for blk in per.values():
            ax.plot(
                [blk["r_min_m"] / 1e3, blk["r_max_m"] / 1e3],
                [blk["w_s"]] * 2,
                color="black",
                lw=0.9,
                solid_capstyle="butt",
            )
            ax.plot([0, 10], [blk["w_s"]] * 2, color="black", lw=0.3, ls=(0, (1, 1.5)))
        ax.set_yscale("log")
        ax.set_xlim(0, 10)
        ax.set_ylim(w[0], w[-1])
        ax.set_xticks([0, 4, 8])
        ax.set_title(f"$n = {n + 1}$", fontsize=6, pad=2)
        ax.set_xlabel("radius (km)")
        for s in ax.spines.values():
            s.set_visible(True)
        small_axes(ax)
        if n == 0:
            ax.set_ylabel(r"settling speed $w$ (m s$^{-1}$)")
        else:
            ax.tick_params(labelleft=False)
    # Fraction names on the right of the last small multiple, outside the data.
    right = axes[-1].twinx()
    right.set_yscale("log")
    right.set_ylim(w[0], w[-1])
    names = {"63-125um": "63–125", "125-250um": "125–250", "250-500um": "250–500", ">500um": ">500"}
    right.set_yticks([blk["w_s"] for blk in per.values()])
    right.set_yticklabels([names[k] for k in per])
    right.yaxis.set_minor_locator(plt.NullLocator())
    right.tick_params(labelsize=SMALL + 0.5, length=2, pad=1)
    right.set_ylabel("fraction (µm)", fontsize=6, labelpad=2)
    right.spines["right"].set_visible(True)
    label(axes[0], "g", dy_mm=4.0)
    cb = plt.colorbar(cs, cax=cax, ticks=[-1, -1e-2, 0, 1e-2, 1])
    cb.ax.set_yticklabels(["$-1$", "$-10^{-2}$", "$0$", "$10^{-2}$", "$1$"])
    cb.set_label(r"normalised $\delta\Omega_n/w$", fontsize=SMALL + 0.5, labelpad=1)
    cb.ax.tick_params(labelsize=SMALL, length=1.5, pad=1)
    cb.ax.minorticks_off()
    cb.outline.set_linewidth(0.4)


def panel_h(ax1, ax2, cax):
    res = M["resolution"]
    x = np.array(M["right_singular_vectors"]["log10_q_nodes"])
    qm = M["q_mass_weighted_mean_m3s"]["log10"]
    r1 = np.array(res["R_one_fraction_snr"])
    r4 = np.array(res["R_four_fractions_snr"])
    vmax = max(np.abs(r1).max(), np.abs(r4).max())
    levels = np.linspace(-vmax, vmax, 21)
    cs = None
    for ax, r, title in (
        (ax1, r1, f"1 fraction, $k = {res['one_fraction_snr']['k']}$"),
        (ax2, r4, f"4 fractions, $k = {res['four_fractions_snr']['k']}$"),
    ):
        cs = ax.contourf(x, x, r, levels=levels, cmap=DIV)
        ax.axvline(qm, color="black", lw=0.6, ls=(0, (1, 1)))
        ax.axhline(qm, color="black", lw=0.6, ls=(0, (1, 1)))
        ax.plot([x[0], x[-1]], [x[0], x[-1]], color="0.35", lw=0.4)
        ax.set_aspect("equal")
        ax.set_xticks([3, 5, 7])
        ax.set_yticks([3, 5, 7])
        ax.set_title(title, fontsize=6, pad=2)
        ax.set_xlabel(r"$\log_{10} Q'$")
        for s in ax.spines.values():
            s.set_visible(True)
        small_axes(ax)
    ax1.set_ylabel(r"$\log_{10} Q$")
    ax2.tick_params(labelleft=False)
    cb = plt.colorbar(cs, cax=cax)
    ticks = np.round(np.array([-1, -0.5, 0, 0.5, 1]) * np.floor(vmax * 100) / 100, 2)
    cb.set_ticks(ticks)
    cb.set_label("resolution $R$", fontsize=SMALL + 0.5, labelpad=1)
    cb.ax.tick_params(labelsize=SMALL + 0.5, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    label(ax1, "h", dy_mm=4.0)


def panel_i(ax, cax):
    nv = M["noise_vs_fractions"]
    sig = np.array(nv["sigma_log_grid"])
    counts = np.array(nv["N_res_snr"])
    ks = np.array(nv["fraction_counts"])
    s0 = M["construction"]["sigma_log"]
    top = int(counts.max())
    bounds = np.arange(-0.5, top + 1.5, 1.0)
    cmap = plt.get_cmap("cividis", len(bounds) - 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    # Cell edges: geometric midpoints in sigma, unit steps in the fraction count.
    e = np.sqrt(sig[1:] * sig[:-1])
    sig_edges = np.concatenate([[sig[0] ** 2 / e[0]], e, [sig[-1] ** 2 / e[-1]]])
    pc = ax.pcolormesh(
        sig_edges,
        np.arange(0.5, ks.size + 1.0),
        counts,
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=False,
    )
    pc.set_edgecolor("face")
    for k in ks[1:]:
        ax.axhline(k - 0.5, color="white", lw=0.5)
    ax.axvline(s0, color="white", lw=0.7, ls=(0, (2, 1.2)))
    ax.axvline(0.5 * s0, color="white", lw=0.5, ls=(0, (1, 1)))
    for k in (1, 4):
        ax.plot(s0, k, marker="*", ms=5.5, mfc="white", mec="black", mew=0.4)
        ax.plot(0.5 * s0, k, marker="o", ms=3.0, mfc="none", mec="white", mew=0.7)
    ax.set_xscale("log")
    ax.set_xlim(sig_edges[0], sig_edges[-1])
    ax.set_ylim(0.5, ks.size + 0.5)
    ax.set_yticks(ks)
    ax.set_xticks([0.1, 0.2, 0.4, 0.8, 1.6])
    ax.set_xticklabels(["0.1", "0.2", "0.4", "0.8", "1.6"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(r"observation noise $\sigma_{\log}$")
    ax.set_ylabel("sieve fractions")
    ax.tick_params(axis="y", length=0)
    for s in ax.spines.values():
        s.set_visible(True)
    small_axes(ax)
    cb = plt.colorbar(pc, cax=cax, ticks=np.arange(0, top + 1, 2))
    cb.set_label("resolvable modes", fontsize=SMALL + 0.5, labelpad=1)
    cb.ax.tick_params(labelsize=SMALL + 0.5, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    label(ax, "i", dy_mm=4.0)


# --------------------------------------------------------------------------
def main() -> int:
    fig = plt.figure(figsize=(TWO_COL, 222 * MM))
    outer = GridSpec(
        4,
        1,
        figure=fig,
        height_ratios=[1.0, 1.5, 1.05, 1.15],
        hspace=0.42,
        left=0.06,
        right=0.985,
        top=0.975,
        bottom=0.04,
    )

    row1 = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.55)
    panel_a(fig.add_subplot(row1[0]))
    panel_b(fig.add_subplot(row1[1]))
    panel_c(fig.add_subplot(row1[2]))
    panel_d(fig.add_subplot(row1[3]))

    row2 = GridSpecFromSubplotSpec(
        1, 5, subplot_spec=outer[1], width_ratios=[1.0, 0.095, 0.03, 0.05, 0.95], wspace=0.0
    )
    panel_e(fig.add_subplot(row2[0]), fig.add_subplot(row2[2]))
    panel_f(fig.add_subplot(row2[4], projection="3d"))

    row3 = GridSpecFromSubplotSpec(
        1, 7, subplot_spec=outer[2], width_ratios=[1, 1, 1, 1, 0.42, 0.05, 0.22], wspace=0.1
    )
    axes_g = [fig.add_subplot(row3[i]) for i in range(4)]
    panel_g(axes_g, fig.add_subplot(row3[5]))

    row4 = GridSpecFromSubplotSpec(
        1,
        7,
        subplot_spec=outer[3],
        width_ratios=[1.0, 1.0, 0.05, 0.5, 1.45, 0.05, 0.16],
        wspace=0.1,
    )
    panel_h(fig.add_subplot(row4[0]), fig.add_subplot(row4[1]), fig.add_subplot(row4[2]))
    panel_i(fig.add_subplot(row4[4]), fig.add_subplot(row4[5]))
    place_labels(fig)

    out_pdf = ROOT / "figures" / "fig_svd.pdf"
    fig.savefig(out_pdf)
    fig.savefig(ROOT / "figures" / "fig_svd.png", dpi=300)
    print("wrote figures/fig_svd.pdf and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
