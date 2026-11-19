"""Plot population PR from the two-site theory and completed network cohorts.

Run `python population_pr.py` to regenerate the PDF and PNG beside this file.
The degree slice connects the 13 solved theory points. The composition slice
evaluates the fixed-coupling formula directly at every plotted fraction. The
pair panel uses the matched four-window diagonal energies in the primary
records and their per-network local add-ons, wherever available.

For a fixed covariance and a uniform K-neuron subset S of one population,
E_S[tr(C_SS²)]/K = mean(v_i²) + (K-1) P_aa / N, where v_i = C_ii and
P_aa = N mean(C_ij²) over distinct ordered pairs. The slope of P_aa thus
selects the best composition for a fixed-size readout in the concentrated
large-network limit, where the one-site moments are fixed. The finite
expectation of the measured subset PR also depends on fluctuations of its
numerator and denominator.
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "ei-matplotlib"))
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, PercentFormatter
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent / "experiments"
sys.path.insert(0, str(ROUND / "figures"))
sys.path.insert(0, str(HERE.parent / "src"))
from eicavity.composition import dimension_at_fraction
from paper_style import E, I, JOINT, LIGHT, WASH, SIZE_MARKERS, apply_style, panel_label, save_figure
from sweep_records import cohort_handles, composition_moments, dense_records, draw_group, point_offsets


def dimensions(prediction):
    """Return D_E/N_E, D_I/N_I, and D/N in that order."""
    return [*prediction["within_population_dimension_fractions"], prediction["dimension_fraction"]]


def theory_slices():
    degree_dir = ROUND / "degree"
    archive = json.loads((degree_dir / "theory_curve.json").read_text())
    rows = sorted(archive["rows"], key=lambda row: row["chi"])
    chi = np.array([r["chi"] for r in rows])
    degree = np.array([dimensions(r["prediction"]) for r in rows])

    # Changing f holds W = G diag(f) and the operating-point laws fixed.
    # Reuse the solved Bernoulli law for the full family instead of rerunning DMFT.
    tensor, means, seconds = composition_moments()

    # d_E = v_E² / (a + b f/(1-f)). Maximizing f*d_E at fixed N gives f_star.
    a, b = tensor[0, 0]
    f_star = np.sqrt(a) / (np.sqrt(a) + np.sqrt(b))
    # Remove the same-neuron contribution. Then P_aa = c_aE/f + c_aI/(1-f)
    # is N times the mean squared covariance over distinct a,a pairs.
    coefficients = tensor[[0, 1], [0, 1], :].copy()
    coefficients[np.diag_indices(2)] -= seconds
    roots = np.sqrt(coefficients)
    pair_minima = roots[:, 0] / roots.sum(axis=1)
    fractions = np.unique(np.r_[np.linspace(0.5, 0.95, 226), f_star, pair_minima])
    fraction_matrix = np.column_stack((fractions, 1 - fractions))
    pairs = (1 / fraction_matrix) @ coefficients.T
    results = [dimension_at_fraction(tensor, means, seconds, f) for f in fraction_matrix]
    composition = np.array([dimensions(r) for r in results])
    np.testing.assert_allclose(pairs, [np.diag(r["pair_covariance_energy"]) for r in results])
    np.testing.assert_allclose(fractions[pairs.argmin(axis=0)], pair_minima)
    at_ref = np.flatnonzero(np.isclose(fractions, archive["fractions"][0])).item()
    np.testing.assert_allclose(composition[at_ref], degree[chi == 1][0], rtol=1e-10)
    assert np.isclose(fractions[np.argmax(fractions * composition[:, 0])], f_star)
    return chi, degree, fractions, composition, f_star, pairs, pair_minima


def network_groups(dense=None):
    original = json.loads((ROUND / "figures/campaign_figure_values.json").read_text())
    if dense is None:
        dense = dense_records()
    groups = defaultdict(list)
    pair_groups = defaultdict(list)
    for cohort, records in [("original", original), ("dense", dense)]:
        for record in records:
            cfg, primary = record["config"], record["primary"]
            experiment = record["job"]["experiment"]
            f = np.array([cfg["fraction_e"], 1 - cfg["fraction_e"]])
            block_energy = np.asarray(primary["population_energy"])   # ||C_ab||²_F / N
            variance = np.asarray(primary["nu_by_population"])
            within = f * variance**2 / np.diag(block_energy)
            np.testing.assert_allclose((f @ variance)**2 / block_energy.sum(), primary["d"])
            control = cfg["degree_dispersion"] if experiment == "degree" else f[0]
            key = (experiment, control, cfg["n"], cohort)
            groups[key].append([*within, primary["d"]])
            if experiment == "composition":
                assert cfg["ensemble"] == "bernoulli"
                diagonal = primary.get("diagonal_population_energy")
                if cohort == "original" and diagonal is None:
                    addon = ROUND / "composition" / (record["name"] + "_local_addon.json")
                    if addon.is_file():
                        local = json.loads(addon.read_text())
                        assert local["duration"] == 8000 and local["block_duration"] == 500
                        diagonal = local["diagonal_energy_four_window"]["by_population_total_N_normalization"]
                if diagonal is not None:
                    n = cfg["n"]
                    counts = np.rint(n * f).astype(int)
                    off = (np.diag(block_energy) - diagonal) * n**2 / (counts * (counts-1))
                    pair_groups[key].append(off)
    for experiment, count in (("degree", 24), ("composition", 40)):
        assert sum(len(v) for k, v in groups.items() if k[0]==experiment and k[3]=="original") == count
    assert sum(len(v) for k, v in pair_groups.items() if k[3]=="original") == 28
    return groups, pair_groups


def main(dense=None):
    apply_style()
    chi, degree, fractions, composition, f_star, pairs, minima = theory_slices()
    groups, pair_groups = network_groups(dense)
    fig, ((ax, bx), (cx, dx)) = plt.subplots(2, 2, figsize=(8.2, 5.9))
    fig.subplots_adjust(left=0.099, right=0.975, bottom=0.175, top=0.941, wspace=0.27, hspace=0.43)
    colors = (E, I, JOINT)
    styles = ("-", "--", "-.")
    labels = (r"E  $D_E/N_E$", r"I  $D_I/N_I$", r"Joint  $D/N$")
    for i, color in enumerate(colors):
        ax.plot(np.sqrt(chi), degree[:, i], color=color, ls=styles[i])
        bx.plot(fractions, composition[:, i], color=color, ls=styles[i])
    exc_total = fractions * composition[:, 0]
    cx.plot(fractions, exc_total, color=E)
    cx.plot(f_star, exc_total.max(), marker="D", ms=5.0, mfc=cx.get_facecolor(), mec=E, zorder=5)
    cx.annotate(rf"$f_E^*={f_star:.3f}$", (f_star, exc_total.max()),
                xytext=(-2, 12), textcoords="offset points", ha="center", color=E)
    dx.axvspan(minima[1], minima[0], color=WASH, zorder=0)
    for i, color in enumerate(colors[:2]):
        dx.plot(fractions, pairs[:, i], color=color, ls=styles[i])
        dx.plot(minima[i], pairs[:, i].min(), marker="D", ms=4, mfc=dx.get_facecolor(), mec=color, zorder=5)
    dx.text(0.954, pairs[-1, 1], "II", color=I, ha="left", va="center", size=8)
    dx.text(0.954, pairs[-1, 0], "EE", color=E, ha="left", va="center", size=8)

    for experiment, axis in [("degree", ax), ("composition", bx)]:
        displayed = {}
        for key, records in groups.items():
            which, setting, size, cohort = key
            if which == experiment:
                x = np.sqrt(setting) if which == "degree" else setting
                displayed[x, size, cohort] = key
        offsets = point_offsets(displayed, .026 if experiment=="degree" else .009)
        jitter = .004 if experiment=="degree" else .0013
        for key, group in displayed.items():
            x, size, cohort = key
            x_plot = x+offsets[key]
            values = np.asarray(groups[group])
            for i, color in enumerate(colors):
                draw_group(axis, x_plot, values[:,i], SIZE_MARKERS[size], color, cohort, jitter=jitter)
            if experiment == "composition":
                draw_group(cx, x_plot, x*values[:,0], SIZE_MARKERS[size], E, cohort, jitter=jitter)
                if group in pair_groups:
                    measured_pairs = np.asarray(pair_groups[group])
                    for i, color in enumerate(colors[:2]):
                        draw_group(dx, x_plot, measured_pairs[:,i], SIZE_MARKERS[size], color,
                                   cohort, jitter=jitter)

    for letter, axis in zip("ABCD", (ax, bx, cx, dx)):
        panel_label(axis, letter)
    ax.set(xlabel=r"Degree dispersion $\chi$ (square-root scale)", ylabel="Per-neuron dimension",
           xlim=(-.065, np.sqrt(2)+.065), yscale="log")
    ticks = [0, .05, .2, .5, 1, 1.5, 2]
    ax.set_xticks(np.sqrt(ticks), [f"{value:g}" for value in ticks])
    ax.set_yticks([0.02, 0.05, 0.10, 0.20, 0.50])
    bx.set(ylabel="Per-neuron dimension", yscale="log")
    bx.set_yticks([0.02, 0.04, 0.08, 0.16, 0.24])
    cx.set(ylabel=r"E dimensions / network size   $D_E/N$")
    cx.set_yticks([0.015, 0.025, 0.035, 0.045])
    dx.set(ylabel=r"Squared covariance per pair $\times\,N$")
    dx.set_yticks([0.4, 0.6, 0.8, 1.0])
    for axis in (bx, cx, dx):
        axis.set(xlabel=r"Excitatory fraction $f_E$", xlim=(0.49, 0.98 if axis is dx else 0.96))
        axis.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 0.95], ["0.5", "0.6", "0.7", "0.8", "0.9", "0.95"])
    for axis in (ax, bx, cx):
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=1 if axis is cx else 0))
        axis.yaxis.set_minor_locator(NullLocator())
    for axis in (ax, bx, cx, dx):
        axis.grid(axis="y", color=LIGHT, lw=0.5)

    handles = [Line2D([], [], color=c, ls=s, label=l) for c, s, l in zip(colors, styles, labels)]
    handles += cohort_handles({key[2] for key in groups}, has_dense=any(k[3]=="dense" for k in groups))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.51, 0.009),
               ncol=4, columnspacing=1.5, handletextpad=0.45, labelspacing=.35)
    save_figure(fig, HERE / "population_pr", formats=("pdf", "png"))
    print(f"Conditional D_E/N peaks at {exc_total.max():.8f} for f_E={f_star:.8f}")
    print(f"Per-pair squared covariance minima occur at f_E={minima[0]:.8f} (EE) and {minima[1]:.8f} (II)")


if __name__ == "__main__":
    main()
