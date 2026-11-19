"""Render the spectral replication from the network results and one saved spectrum."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figures"))
from paper_style import JOINT, MUTED, LIGHT, WASH, apply_style, panel_label, save_figure


def main():
    summary = json.loads((HERE / "summary.json").read_text())
    cases = [r for r in summary["network_results"] if "primary" in r]
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 6.4))
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.105, top=0.945, wspace=0.49, hspace=0.48)
    for letter, ax in zip("ABCD", axes.flat):
        panel_label(ax, letter)
    a, b, c, d = axes.flat

    # Display the first eligible seed in fixed cohort order.
    with np.load(HERE / cases[0]["root_mode_file"]) as z:
        eigen = z["eigenvalues"]
        variance = z["input_variance"][0]
    order = np.argsort(variance)
    cmap = LinearSegmentedColormap.from_list("slate", [WASH, MUTED, JOINT])
    pts = a.scatter(
        eigen.real[order],
        eigen.imag[order],
        c=variance[order],
        cmap=cmap,
        s=14,
        edgecolors=MUTED,
        linewidths=0.3,
        vmin=0,
    )
    a.axvline(1, linestyle="--", color=MUTED, linewidth=1, zorder=0)
    a.set(xlabel=r"$\mathrm{Re}\,\lambda$", ylabel=r"$\mathrm{Im}\,\lambda$", aspect="equal")
    cbar = fig.colorbar(pts, ax=a, pad=0.03, shrink=0.78, aspect=17)
    cbar.set_label("Projected input variance", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)
    b.axhline(0, color=LIGHT, zorder=0, linewidth=1)
    label_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=m,
            color=color,
            markerfacecolor=face,
            markersize=5,
            label=label,
        )
        for m, color, face, label in [
            ("o", JOINT, JOINT, "First root"),
            ("o", MUTED, "white", "Other root"),
            ("_", MUTED, MUTED, "Shuffle median"),
        ]
    ]
    for j, n in enumerate((250, 500)):
        group = [r for r in cases if r["N"] == n]
        for x, row in zip(j + np.linspace(-0.29, 0.29, len(group)), group):
            dat = row["primary"]
            b.plot([x, x], [dat["shuffle_rho"], dat["rho"]], color=LIGHT, lw=0.85, zorder=1)
            b.scatter(x, dat["shuffle_rho"], marker="_", s=20, color=MUTED, zorder=2)
            for other in row["all_root_stats"][1:]:
                b.scatter(
                    x,
                    other["rho"],
                    s=14,
                    facecolors="none",
                    edgecolors=MUTED,
                    linewidths=0.7,
                    zorder=3,
                )
            b.scatter(x, dat["rho"], s=13, color=JOINT, zorder=4)
        for ax, observed, null in [
            (c, "subspace_energy", "gain_null_energy"),
            (d, "residual_pr_ratio", "gain_null_pr_ratio"),
        ]:
            ax.scatter(
                [r["primary"][null] for r in group],
                [r["primary"][observed] for r in group],
                marker=("o", "^")[j],
                s=28,
                facecolors=("white", JOINT)[j],
                edgecolors=JOINT,
                lw=0.85,
                zorder=3,
                label=f"$N={n}$",
            )
    b.set(
        xticks=[0, 1],
        xticklabels=["250", "500"],
        xlim=(-0.42, 1.42),
        xlabel="Network size",
        ylabel="Mode–variance rank correlation",
    )
    b.legend(
        handles=label_handles,
        loc="upper left",
        bbox_to_anchor=(0, 1.04),
        ncol=2,
        fontsize=7,
        columnspacing=0.5,
        handlelength=1.15,
        handletextpad=0.25,
    )
    b.margins(y=0.27)
    c.set(
        xlabel="Shuffled energy fraction",
        ylabel="Observed energy fraction",
        xlim=(-0.035, 1.035),
        ylim=(-0.035, 1.035),
    )
    c.legend(loc="lower right", fontsize=8)
    ratios = [
        r["primary"][key] for r in cases for key in ("residual_pr_ratio", "gain_null_pr_ratio")
    ]
    low = min(ratios + [1]) * 0.91
    high = max(ratios + [1]) * 1.06
    d.set(
        xlabel="Shuffled complement PR / full PR",
        ylabel="Observed complement PR / full PR",
        xlim=(low, high),
        ylim=(low, high),
    )
    for ax in (c, d):
        ax.plot(ax.get_xlim(), ax.get_xlim(), color=MUTED, dashes=(3, 3), linewidth=0.9, zorder=0)
        ax.set_aspect("equal", adjustable="box")
    save_figure(fig, HERE / "spectral_replication")


if __name__ == "__main__":
    main()
