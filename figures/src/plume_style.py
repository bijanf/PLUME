"""Matplotlib style for Nature-family journal figures.

Springer Nature "Guide to Preparing Final Artwork": sans-serif, 5-7 pt text,
RGB, vector PDF, 1-column 88 mm and 2-column 180 mm.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

MM = 1.0 / 25.4
ONE_COL = 88.0 * MM  # 3.465 in
TWO_COL = 180.0 * MM  # 7.087 in

# Colour-blind-safe qualitative palette (Okabe & Ito), RGB.
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#7F7F7F",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.format": "pdf",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.0,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax, text: str, dx: float = -0.16, dy: float = 1.04) -> None:
    ax.text(
        dx, dy, text, transform=ax.transAxes, fontsize=7, fontweight="bold", va="top", ha="left"
    )
