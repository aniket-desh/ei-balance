"""Draw the two-site cavity and the conditional gain of two excitatory cells.

The network edges are schematic. The rate curve uses the archived Bernoulli
composition reference, whose local laws are common to the two compositions.
The tagged operating points mark the lower and upper quartiles of the E input
mean distribution, drawn above the curve as the Gaussian density N(h_E, Delta_E).
No cell trajectories or measured synapses enter the figure.
Run `python3 cavity_schematic.py` to regenerate the adjacent PDF, SVG and PNG.

Visual precedents (layout and line work adapted, no code reused):
- Clark, Abbott and Litwin-Kumar, PRL 131, 118401 (2023), Fig. 1, and Clark et
  al., PRX 15, 041019 (2025), Fig. 11: a black-outlined reservoir disc with a
  faint gray gradient, small filled units with black outlines, black curved
  arrows, sans-serif words with Computer Modern symbols, and an index-set line
  under each group title.
- Clark, arXiv:2610.12426 (2026), Fig. 3: thin axes, small bold panel letters
  and direct labels in place of a legend box.
- figures4papers, scientific-figure-making/references/design-theory.md
  (https://github.com/ChenLiu-1996/figures4papers): open top/right spines,
  frameless direct annotation, a blue/red/neutral semantic palette and editable
  vector text. Its bar and trend demos have no network schematic, so the
  schematic layout follows the Clark figures above.
"""
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import ArrowStyle, Circle, ConnectionPatch, FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr, ndtri


HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "archive/recovered/bundle/diagnosis/theory/summary.json"
sys.path.insert(0, str(HERE.parent / "experiments/figures"))
from paper_style import E, I, apply_style, save_figure

INK = "#111111"


def muted(color, white=0.16):
    """Mix a palette color with white so black outlines and labels stay dominant."""
    return tuple((1 - white)*np.array(to_rgb(color)) + white)


FILL = {"E": muted(E), "I": muted(I)}
WORDS = 7.6     # sans-serif words, in points at the manuscript's text width
SYMBOLS = 9.2   # Computer Modern symbols


def style():
    apply_style()
    plt.rcParams.update({
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.7,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    })


def cavity(ax):
    ax.set(xlim=(0.0, 5.75), ylim=(-0.28, 3.30), aspect="equal")
    ax.axis("off")

    # Reservoir: a crisp outline over a faint top-to-bottom gray gradient.
    center, radius = (1.50, 1.42), 1.20
    disc = Circle(center, radius, facecolor="none", edgecolor=INK, lw=0.9, zorder=1.5)
    shade = ax.imshow(np.linspace(0, 1, 64)[:, None], origin="upper", aspect="auto",
                      cmap=LinearSegmentedColormap.from_list("disc", ["#FBFBFB", "#D9D9D9"]),
                      extent=(center[0]-radius, center[0]+radius,
                              center[1]-radius, center[1]+radius), zorder=1)
    ax.add_patch(disc)
    shade.set_clip_path(disc)

    for x, words, symbols in ((center[0], "reservoir neurons", r"$k,\ell\,\in\,\{1,\ldots,N-2\}$"),
                              (4.62, "cavity neurons", r"$i,j\,\in\,\mathrm{E}$")):
        ax.text(x, 3.17, words, ha="center", va="center", fontsize=WORDS)
        ax.text(x, 2.89, symbols, ha="center", va="center", fontsize=SYMBOLS)

    # Fixed positions keep the topology stable under edits to labels or routes.
    cells = {
        "e1": (0.62, 1.72, "E"), "e2": (1.02, 2.20, "E"), "e3": (1.72, 2.30, "E"),
        "e4": (1.47, 1.52, "E"), "e5": (0.92, 0.78, "E"), "e6": (1.62, 0.52, "E"),
        "n": (0.98, 1.27, "I"), "k": (2.24, 1.83, "I"), "l": (2.22, 0.98, "E"),
        "i": (4.62, 2.02, "E"), "j": (4.62, 0.86, "E"),
    }
    patches = {}
    for name, (x, y, kind) in cells.items():
        patches[name] = Circle((x, y), 0.125 if name in "ij" else 0.105,
                               facecolor=FILL[kind], edgecolor=INK, lw=0.8, zorder=4)
        ax.add_patch(patches[name])
    for name, symbol, dx, dy in (("k", r"$k$", 0.02, 0.25), ("l", r"$\ell$", 0.03, -0.25),
                                 ("i", r"$i$", 0.25, 0.0), ("j", r"$j$", 0.25, -0.02)):
        x, y, _ = cells[name]
        ax.text(x+dx, y+dy, symbol, ha="center", va="center", fontsize=SYMBOLS, zorder=5)

    # The presynaptic cell sets the sign: an arrowhead for E, a flat bar for I.
    def connection(source, target, bend, lw=0.9, scale=7.0):
        inhibitory = cells[source][2] == "I"
        head = (ArrowStyle("|-|", widthA=0, widthB=0.42) if inhibitory
                else ArrowStyle("-|>", head_length=0.62, head_width=0.26))
        ax.add_patch(FancyArrowPatch(
            cells[source][:2], cells[target][:2], patchA=patches[source], patchB=patches[target],
            connectionstyle=f"arc3,rad={bend}", arrowstyle=head, mutation_scale=scale,
            shrinkA=0.6, shrinkB=1.6 if inhibitory else 0.6, lw=lw, color=INK,
            joinstyle="miter", capstyle="butt", zorder=3))

    for source, target, bend in (("e1", "e2", -0.15), ("e2", "e3", -0.12), ("e3", "k", -0.15),
                                 ("e2", "e4", 0.12), ("n", "e4", -0.12), ("e4", "k", -0.10),
                                 ("k", "e3", -0.22), ("l", "e4", -0.12), ("e5", "e6", 0.14),
                                 ("e6", "l", 0.14), ("n", "e5", 0.10), ("e1", "n", 0.14),
                                 ("e4", "e6", 0.12)):
        connection(source, target, bend, lw=0.6, scale=5.2)

    # J_ik multiplies the rate of reservoir neuron k in the input to neuron i.
    # Curving both directions to their own right puts the arrows on separate arcs.
    for source, target, bend in (("k", "i", -0.20), ("i", "k", -0.16),
                                 ("l", "j", -0.16), ("j", "l", -0.16),
                                 ("i", "j", -0.26), ("j", "i", -0.26)):
        connection(source, target, bend)
    for x, y, label in ((3.40, 2.36, r"$J_{ik}$"), (3.52, 1.57, r"$J_{ki}$"),
                        (3.30, 1.27, r"$J_{j\ell}$"), (3.42, 0.49, r"$J_{\ell j}$"),
                        (5.20, 1.44, r"$J_{ij},J_{ji}$")):
        ax.text(x, y, label, ha="center", va="center", fontsize=SYMBOLS)

    # Key: one unit of each type with its outgoing terminal.
    for x, kind, word in ((0.45, "E", "excitatory"), (1.98, "I", "inhibitory")):
        unit = Circle((x, -0.10), 0.085, facecolor=FILL[kind], edgecolor=INK, lw=0.8, zorder=4)
        ax.add_patch(unit)
        head = (ArrowStyle("|-|", widthA=0, widthB=0.42) if kind == "I"
                else ArrowStyle("-|>", head_length=0.62, head_width=0.26))
        ax.add_patch(FancyArrowPatch((x, -0.10), (x+0.42, -0.10), patchA=unit, arrowstyle=head,
                                     mutation_scale=6.0, shrinkA=0.6, shrinkB=0, lw=0.8,
                                     color=INK, joinstyle="miter"))
        ax.text(x+0.50, -0.105, word, ha="left", va="center", fontsize=WORDS)


def operating_points(ax, margin):
    reference = json.loads(REFERENCE.read_text())
    h = reference["mean_inputs"][0]
    delta = reference["static_variances"][0]
    q = reference["dynamic_variances"][0]
    width = np.sqrt(1 + q)
    limits = (-8.4, 3.6)
    mu = np.linspace(*limits, 500)
    ax.plot(mu, 2*ndtr(mu/width), color=FILL["E"], lw=1.6, solid_capstyle="butt")

    # Static operating points are Gaussian, mu = h + sqrt(Delta) z, in the closure.
    density = np.exp(-0.5*(mu-h)**2/delta)
    margin.fill_between(mu, density, color="#DCDCDC", lw=0)
    margin.plot(mu, density, color=INK, lw=0.7)
    margin.set(xlim=limits, ylim=(0, 1.12))
    margin.axis("off")
    margin.text(limits[0]+0.15, 1.34, r"E operating points, $\mu=h_E+\sqrt{\Delta_E}\,z$",
                ha="left", va="bottom", fontsize=WORDS, transform=margin.transData)

    # For a fixed q, differentiating the smoothed mean rate gives E[phi'(x)].
    locations = h + np.sqrt(delta)*ndtri([0.25, 0.75])
    for name, point, reach, label_at in zip("ij", locations, (1.35, 1.25),
                                            ((-0.5, 0.30), (-0.55, 0.30))):
        mean = 2*ndtr(point/width)
        gain = np.sqrt(2/np.pi)/width * np.exp(-0.5*(point/width)**2)
        dx = np.array([-reach, reach])
        ax.plot(point+dx, mean+gain*dx, color=INK, lw=0.8, zorder=3)
        ax.plot(point, mean, "o", ms=5.6, color=FILL["E"], markeredgewidth=0.8,
                markeredgecolor=INK, zorder=4)
        ax.text(point+label_at[0], mean+label_at[1], f"${name}$", ha="center", va="center",
                fontsize=SYMBOLS)
        # The quartile of the density above selects each operating point.
        top = np.exp(-0.5*(point-h)**2/delta)
        ax.figure.add_artist(ConnectionPatch((point, top), (point, mean), margin.transData,
                                             ax.transData, color=INK, lw=0.6,
                                             linestyle=(0, (1.2, 2.2)), shrinkB=3.4,
                                             zorder=2))
    # Label the slope in the empty corner under the curve's upper branch.
    ax.annotate(r"slope $\alpha_E(z)$", (locations[1]+0.95, 2*ndtr(locations[1]/width)+0.95*gain),
                xytext=(-0.35, 0.30), textcoords="data", ha="left", va="center", fontsize=WORDS,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkA=1.5, shrinkB=2.5))
    ax.set(xlim=limits, ylim=(-0.13, 2.06), xticks=[-8, -6, -4, -2, 0, 2], yticks=[0, 1, 2],
           xlabel=r"Operating point $\mu$", ylabel=r"Mean rate $m_E$")
    ax.xaxis.label.set_size(WORDS+0.6)
    ax.yaxis.label.set_size(WORDS+0.6)
    ax.tick_params(length=2.6, width=0.7, labelsize=7.2, pad=2.2)


def main():
    style()
    # The manuscript's text width is 6.6 in, so fonts print at their stated size.
    fig = plt.figure(figsize=(6.6, 2.42), facecolor="white")
    network = fig.add_axes([0.012, 0.015, 0.615, 0.955])
    rates = fig.add_axes([0.715, 0.185, 0.270, 0.475])
    margin = fig.add_axes([0.715, 0.685, 0.270, 0.115])
    cavity(network)
    operating_points(rates, margin)
    for x, letter in ((0.012, "(a)"), (0.640, "(b)")):
        fig.text(x, 0.975, letter, ha="left", va="top", weight="bold", fontsize=8.6)
    save_figure(fig, HERE / "cavity_schematic")


if __name__ == "__main__":
    main()
