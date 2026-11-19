"""Draw the two-site cavity and the conditional gain of two excitatory cells.

The network edges are schematic. The rate curve uses the archived Bernoulli
composition reference, whose local laws are common to the two compositions.
The tagged operating points mark the lower and upper quartiles of the E input
mean distribution. No cell trajectories or measured synapses enter the figure.
Run `python3 cavity_schematic.py` to regenerate the adjacent PDF, SVG and PNG.
"""
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle
import numpy as np
from scipy.special import ndtr, ndtri


HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "archive/recovered/bundle/diagnosis/theory/summary.json"
sys.path.insert(0, str(HERE.parent / "experiments/figures"))
from paper_style import E, I, JOINT, LIGHT, MUTED, WASH, apply_style, panel_label, save_figure


def cavity(ax):
    ax.set(xlim=(-0.09, 5.65), ylim=(-0.2, 3.44), aspect="equal")
    ax.axis("off")
    ax.add_patch(Ellipse((1.51, 1.67), 2.83, 2.43, facecolor=WASH,
                         edgecolor=LIGHT, linewidth=0.85))
    ax.text(1.51, 3.06, r"Reservoir ($N-2$ neurons)", ha="center", color=JOINT,
            fontsize=8.3)
    ax.text(4.8, 3.06, "Tagged E neurons", ha="center", color=JOINT, fontsize=8.3)

    # Each node has a fixed position so that edits to labels or arrow routes
    # preserve the topology. The presynaptic cell sets each edge's sign.
    cells = {
        "e1": (0.63, 1.88, "E"), "e2": (1.07, 2.46, "E"),
        "e3": (1.93, 2.51, "E"), "e4": (1.73, 1.72, "E"),
        "e5": (0.94, 0.94, "E"), "ih": (1.05, 1.58, "I"),
        "k": (2.44, 1.99, "I"), "l": (2.12, 1.00, "E"),
        "i": (4.80, 2.34, "E"), "j": (4.80, 0.99, "E"),
    }
    colors = {"E": E, "I": I}
    patches = {}
    labels = {"k": r"$k$", "l": r"$\ell$", "i": r"$i$", "j": r"$j$"}
    for name, (x, y, kind) in cells.items():
        radius = 0.185 if name in ("i", "j") else 0.145 if name in labels else 0.092
        fill = colors[kind]
        if kind == "I":
            patch = Rectangle((x-radius, y-radius), 2*radius, 2*radius,
                              facecolor=fill, edgecolor="white", lw=0.7, zorder=4)
        else:
            patch = Circle((x, y), radius, facecolor=fill, edgecolor="white",
                           lw=0.7, zorder=4)
        ax.add_patch(patch)
        patches[name] = patch
        if name in labels:
            ax.text(x, y, labels[name], ha="center", va="center", color="white",
                    fontsize=9 if name in ("i", "j") else 7.4, zorder=5)

    def connection(source, target, bend, internal=False):
        color = colors[cells[source][2]]
        arrow = FancyArrowPatch(
            cells[source][:2], cells[target][:2],
            patchA=patches[source], patchB=patches[target],
            connectionstyle=f"arc3,rad={bend}", arrowstyle="-|>",
            shrinkA=0.3, shrinkB=0.3, mutation_scale=5.4 if internal else 7.5,
            lw=0.6 if internal else 1, color=color,
            alpha=0.57 if internal else 0.95, zorder=2,
        )
        ax.add_patch(arrow)

    for source, target in (("e1", "e2"), ("e2", "e3"), ("e3", "k"),
                           ("e2", "e4"), ("ih", "e4"), ("e4", "k"),
                           ("k", "e2"), ("l", "e4"), ("e5", "l"),
                           ("ih", "e5"), ("e1", "ih")):
        connection(source, target, 0.07, internal=True)

    # J_ik multiplies the rate of reservoir neuron k in the input to neuron i.
    # Curving both directions to their own right puts the arrows on separate arcs.
    for source, target, bend in (("k", "i", -0.20), ("i", "k", -0.15),
                                 ("l", "j", -0.15), ("j", "l", -0.15),
                                 ("i", "j", -0.23), ("j", "i", -0.23)):
        connection(source, target, bend)
    for x, y, label in ((3.57, 2.59, r"$J_{ik}$"), (3.57, 1.77, r"$J_{ki}$"),
                        (3.47, 1.35, r"$J_{j\ell}$"), (3.47, 0.59, r"$J_{\ell j}$")):
        ax.text(x, y, label, ha="center", va="center", fontsize=8.3, color=JOINT)

    for x, label, marker, color in [(0.55, "Excitatory", "o", E),
                                    (2.41, "Inhibitory", "s", I)]:
        ax.plot(x, 0.03, marker=marker, markersize=5.3, color=color, linestyle="none")
        ax.text(x+0.19, 0.03, label, ha="left", va="center", fontsize=7.6, color=MUTED)


def operating_points(ax):
    reference = json.loads(REFERENCE.read_text())
    h = reference["mean_inputs"][0]
    delta = reference["static_variances"][0]
    q = reference["dynamic_variances"][0]
    width = np.sqrt(1 + q)
    mu = np.linspace(-6, 3.4, 400)
    ax.plot(mu, 2*ndtr(mu/width), color=E, lw=1.85)

    # For a fixed q, differentiating the smoothed mean rate gives E[phi'(x)].
    locations = h + np.sqrt(delta)*ndtri([0.25, 0.75])
    for name, point in zip(("i", "j"), locations):
        mean = 2*ndtr(point/width)
        gain = np.sqrt(2/np.pi)/width * np.exp(-0.5*(point/width)**2)
        dx = np.array([-1.16, 1.16])
        ax.plot(point+dx, mean+gain*dx, color=MUTED, lw=1.0, linestyle=(0, (3, 2)))
        ax.plot(point, mean, "o", ms=4.9, color=E,
                markeredgewidth=0.7, markeredgecolor="white", zorder=4)
        ax.annotate(f"${name}$", (point, mean), xytext=(-3.5, 6.5),
                    textcoords="offset points", ha="center", fontsize=9.5)
    ax.plot([], [], color=MUTED, lw=1, linestyle=(0, (3, 2)),
            label=r"Local gain $\alpha_E$ (tangent)")
    ax.legend(loc="upper left", frameon=False, borderpad=0, handlelength=1.7,
              handletextpad=0.5, fontsize=7.2)
    ax.set(xlim=(-6, 3.5), ylim=(-0.08, 2.08), xticks=[-6, -4, -2, 0, 2],
           yticks=[0, 1, 2], xlabel=r"Operating point $\mu$",
           ylabel=r"Mean firing rate $m_E$")
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(JOINT)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(length=2.7, width=0.65, labelsize=7.4, color=JOINT, pad=2.5)


def main():
    apply_style()
    fig = plt.figure(figsize=(7.05, 2.56), facecolor="white")
    network = fig.add_axes([0.019, 0.035, 0.52, 0.83])
    rates = fig.add_axes([0.666, 0.215, 0.30, 0.61])
    cavity(network)
    operating_points(rates)
    panel_label(network, "A")
    panel_label(rates, "B")
    # The axes have different heights. Place both letters on the same baseline.
    label_a, label_b = network.texts[-1], rates.texts[-1]
    anchor = network.transAxes.transform(label_a.get_position())
    label_b.set_y(rates.transAxes.inverted().transform(anchor)[1])
    save_figure(fig, HERE / "cavity_schematic")


if __name__ == "__main__":
    main()
