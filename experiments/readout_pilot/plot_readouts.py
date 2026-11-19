"""Fixed-size readout dimensions and the paired changes across compositions."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figures"))
from paper_style import E, I, LIGHT, MUTED, apply_style, panel_label, save_figure


def show_networks(ax, x, values, color, jitter):
    """One point per connectivity seed, after averaging the eight cell subsets."""
    ax.scatter(x + np.linspace(-jitter, jitter, len(values)), values,
               color=color, s=14, alpha=.45, linewidths=0, zorder=3)
    se = np.std(values, ddof=1) / np.sqrt(len(values))
    ax.errorbar(x, np.mean(values), yerr=t.ppf(.975, len(values)-1)*se,
                color=color, fmt="s", ms=4.6, elinewidth=1, capsize=2.7, zorder=4)


def plot(data):
    seeds = (6101, 6102, 6103, 6104)
    fractions = (.8, .88, .94)
    observations = {(r["fraction_e"], r["seed"]): r["windows"]["T8000_gap0"]
                    for r in data["records"]}
    expected = {(f, seed) for f in fractions for seed in seeds}
    if len(data["records"]) != len(expected) or observations.keys() != expected:
        raise ValueError("The readout figure requires all four network seeds at each of the three fractions.")

    apply_style()
    fig, (a,b) = plt.subplots(1, 2, figsize=(7.3, 3.25), layout="constrained")
    b.axhline(0, lw=1, ls="--", color=MUTED, zorder=1)
    for population, color, side in [("E", E, -1), ("I", I, 1)]:
        values = []
        for f in fractions:
            cells = [observations[f,seed][population]["PR"] for seed in seeds]
            if any(len(cell["draws"]) != 8 for cell in cells):
                raise ValueError("Each network and population must supply eight measured 100-cell readouts.")
            means = np.array([cell["mean"] for cell in cells])
            np.testing.assert_allclose(means, [np.mean(cell["draws"]) for cell in cells])
            values.append(means)
            show_networks(a, f + .003*side, means, color, .0012)
        # Each column uses the same seed label at the two fractions being compared.
        for index, delta in enumerate(np.diff(values, axis=0)):
            show_networks(b, index + .065*side, delta, color, .025)

    a.set(xlabel=r"Excitatory fraction $f_E$", ylabel="Dimension of a 100-cell readout",
          xlim=(.775, .965))
    a.set_xticks(fractions, ["0.80", "0.88", "0.94"])
    b.set(xlabel="Change in excitatory fraction", ylabel="Paired change in readout dimension",
          xlim=(-.35,1.35))
    b.set_xticks([0,1], [r"$0.80\to0.88$", r"$0.88\to0.94$"])
    for ax, letter in [(a,"A"), (b,"B")]:
        ax.grid(axis="y", color=LIGHT, lw=.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        panel_label(ax, letter)
    handles = [Line2D([], [], color=color, marker="s", ls="none", ms=4.5, label=label)
               for color, label in [(E, "100 excitatory cells"), (I, "100 inhibitory cells")]]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, columnspacing=2)
    save_figure(fig, HERE / "fixed_size_readout", formats=("pdf", "png"))


if __name__ == "__main__":
    plot(json.loads((HERE / "summary.json").read_text()))
