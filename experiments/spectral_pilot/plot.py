"""Regenerate the fixed-point modal variance plot from the measured projections."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "ei-matplotlib"))
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import PowerNorm
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figures"))
from paper_style import DIMENSION_CMAP, MUTED, apply_style, panel_label, save_figure


def main():
    summary = json.loads((HERE / "summary.json").read_text())
    panels = []
    for case in summary["cases"]:
        if case["roots"]:
            with np.load(HERE / case["roots"][0]["variance_file"]) as z:
                panels.append((z["eigenvalues"], z["variance"][0]))
    if not panels:
        raise RuntimeError("No network has a verified fixed point.")
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(5.7, 5.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=.13, right=.835, bottom=.115, top=.935, hspace=.29, wspace=.19)
    extent = max(np.abs(np.r_[values.real, values.imag]).max() for values, _ in panels)
    limit = np.ceil(1.02*extent*2)/2
    maximum = max(float(np.max(v)) for _, v in panels)
    upper = 2**np.ceil(np.log2(maximum)) if maximum > 0 else 1.
    norm = PowerNorm(gamma=.5, vmin=0, vmax=upper)
    for i, ax in enumerate(axes.flat):
        if i >= len(panels):
            ax.set_axis_off()
            continue
        eigvals, variance = panels[i]
        order = np.argsort(variance)
        panel_label(ax, "ABCD"[i])
        ax.axvline(1, color=MUTED, lw=.85, dashes=(3, 2), zorder=0)
        dots = ax.scatter(eigvals.real[order], eigvals.imag[order], c=variance[order], norm=norm,
                          cmap=DIMENSION_CMAP, s=10, linewidths=.2, edgecolors=MUTED)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        if i // 2 == 1:
            ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
        if i % 2 == 0:
            ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
    color_ax = fig.add_axes([.865,.21,.025,.58])
    cbar = fig.colorbar(dots, cax=color_ax)
    ticks = sorted({v for v in (0,1,4,16,64,256,1024) if v <= upper} | {upper})
    cbar.set_ticks(ticks)
    cbar.set_label(r"Projected input variance  $u^\dagger C_x u$")
    save_figure(fig, HERE / "spectral_pilot", formats=("pdf","png"))


if __name__ == "__main__":
    main()
