"""Chen Liu's open-axis print styling with a saturated Nord palette.

https://github.com/ChenLiu-1996/figures4papers/tree/main/scientific-figure-making
https://www.nordtheme.com/docs/colors-and-palettes
"""

from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


E = "#266FC1"       # Frost blue for E and E–E
I = "#CB3A4D"       # Aurora red for I and I–I
CROSS = "#087F87"   # Frost teal for E–I
JOINT = "#2E3440"   # Polar Night for the whole network
SLATE = "#4C566A"
MUTED = "#657088"
LIGHT = "#D8DEE9"
WASH = "#ECEFF4"
FROST = "#6AADD7"
COMPOSITION = {0.6: "#397DA9", 0.8: "#2D4E79"}
SIZE_COLORS = dict.fromkeys((1000, 2000, 4000), MUTED)
SIZE_MARKERS = {1000: "o", 2000: "s", 4000: "^"}
DIMENSION_CMAP = LinearSegmentedColormap.from_list("nord_dimension", [WASH, FROST, E, JOINT])
LAG_CMAP = LinearSegmentedColormap.from_list("nord_signed", [E, WASH, I])


def apply_style():
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.titlepad": 7,
        "axes.labelsize": 9,
        "axes.labelcolor": JOINT,
        "axes.edgecolor": JOINT,
        "axes.linewidth": 1.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": JOINT,
        "ytick.color": JOINT,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.color": LIGHT,
        "text.color": JOINT,
        "lines.linewidth": 1.7,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.handlelength": 1.9,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, letter):
    ax.text(0, 1.035, f"({letter.lower()})", transform=ax.transAxes,
            ha="left", va="bottom", weight="bold", fontsize=10, color=JOINT)


def save_figure(fig, stem, formats=("pdf", "png", "svg")):
    for suffix in formats:
        fig.savefig(stem.with_suffix("." + suffix), dpi=300)
    plt.close(fig)
