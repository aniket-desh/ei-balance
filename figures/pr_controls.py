"""Plot the original and dense network sweeps against the conditional theory.

Run with --source PATH to recompute from the unpacked project or source ZIP.
Run without arguments to redraw from pr_controls.json and the dense-sweep records.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import Path as ZipPath, ZipFile

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "ei-matplotlib"))

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "experiments/figures"))
from paper_style import (DIMENSION_CMAP, JOINT, MUTED, SLATE, SIZE_MARKERS as MARKERS,
                         apply_style, panel_label, save_figure)

LEGACY_PACKAGE = "recovered/bundle/ei_full/ei_balance_handoff/src/eicavity"
DATA = OUT / "pr_controls.json"


def source_layout(root):
    """Find the package and experiments in a checkout or original handoff."""
    if (root / "src/eicavity/__init__.py").is_file():
        return root / "src/eicavity", root / "experiments"
    return root / LEGACY_PACKAGE, root / "next_round"


def compute(source_path):
    from sweep_records import dense_records

    def calculate(package_parent, experiments):
        def read(name):
            return (experiments / name.removeprefix("experiments/")).read_bytes()

        sys.path.insert(0, str(package_parent))
        try:
            return calculate_grid(read, source_path.name), dense_records(read)
        finally:
            sys.path.pop(0)

    source_path = source_path.resolve()
    if source_path.is_dir():
        package, experiments = source_layout(source_path)
        return calculate(package.parent, experiments)

    with ZipFile(source_path) as archive, TemporaryDirectory(prefix="ei-pr-") as tmp:
        root = ZipPath(archive)
        # GitHub nests a source snapshot under a single top-level directory.
        children = list(root.iterdir())
        if len(children) == 1 and children[0].is_dir():
            root = children[0]
        source, experiments = source_layout(root)
        package = Path(tmp) / "eicavity"
        package.mkdir()
        for module in ("__init__", "composition", "degree", "multitype", "pair", "single_site"):
            filename = module + ".py"
            (package / filename).write_bytes((source / filename).read_bytes())
        return calculate(package.parent, experiments)


def calculate_grid(read, archive_name):
    from eicavity.composition import dimension_at_fraction, energy_tensor
    from eicavity.multitype import solve_populations

    degree = json.loads(read("experiments/degree/theory_curve.json"))
    degree_ref = json.loads(read("experiments/degree/predictions.json"))
    comp_ref = json.loads(read("experiments/composition/composition_theory_targets.json"))
    runs = json.loads(read("experiments/figures/campaign_figure_values.json"))

    f_ref = np.array(degree["fractions"], dtype=float)
    G = np.array(degree["G"], dtype=float)
    W = G * f_ref  # Scale columns by the presynaptic fractions.
    mean_rates = degree_ref["mean_rates"]
    chi_values = np.unique(np.r_[np.linspace(0, 2, 41), [r["chi"] for r in degree["rows"]]].round(10))
    fractions = np.linspace(0.5, 0.95, 91).round(10)
    theory = []

    for chi in chi_values:
        # At a fixed chi, keeping W and the mean rates fixed preserves the
        # conditional one-site law as composition changes. Solve it once.
        model = solve_populations(G, fractions=f_ref, mean_rates=mean_rates,
                                  dt=0.08, horizon=100, degree_dispersion=chi)
        local = model.conditional(n_hermite=96)
        weights = local["weights"]
        variance = local["covariance"][..., 0]
        mean_variance = variance @ weights
        variance_second = variance**2 @ weights
        radius = float(np.max(np.abs(np.linalg.eigvals((local["alpha"]**2 @ weights)[:, None] * W))))
        if radius >= 1:
            raise RuntimeError(f"Unstable covariance resolvent at chi={chi:g}")
        tensor = energy_tensor(W, local, dt=0.08, max_frequency=12)
        dimensions, energies, means = [], [], []
        for f_e in fractions:
            f = np.array([f_e, 1 - f_e])
            prediction = dimension_at_fraction(tensor, mean_variance, variance_second, f)
            dimensions.append(prediction["dimension_fraction"])
            energies.append(prediction["covariance_energy"])
            means.append(float(f @ mean_variance))
        theory.append({"chi": float(chi), "d": dimensions, "nu": means, "energy": energies,
                       "response_radius": radius, "bvp_residual": model.bvp_residual})

    d = np.array([row["d"] for row in theory])
    nu = np.array([row["nu"] for row in theory])
    energy = np.array([row["energy"] for row in theory])
    if not (np.isfinite(d).all() and (0 < d).all() and (d <= 1).all() and (energy > 0).all()):
        raise RuntimeError("Invalid value in theory grid")
    np.testing.assert_allclose(d, nu**2 / energy, rtol=2e-13)
    i80 = index(fractions, 0.8)
    i_chi = [index(chi_values, r["chi"]) for r in degree["rows"]]
    for quantity, values, expected in (
        ("d", d, [r["prediction"]["dimension_fraction"] for r in degree["rows"]]),
        ("nu", nu, [r["nu"] for r in degree["rows"]]),
        ("energy", energy, [r["prediction"]["covariance_energy"] for r in degree["rows"]]),
    ):
        np.testing.assert_allclose(values[i_chi, i80], expected, rtol=2e-6, err_msg=quantity)
    for f, expected in comp_ref["one_site_reference"]["predictions"].items():
        np.testing.assert_allclose(
            d[index(chi_values, 1), index(fractions, float(f))],
            expected["dimension_fraction"], rtol=2e-6)
    print(f"Grid {len(chi_values)} × {len(fractions)}; matches 13 archived degree points and both Bernoulli compositions.")

    observations = []
    for run in runs:
        config = run["config"]
        experiment = run["job"]["experiment"]
        observations.append({
            "name": run["name"], "experiment": experiment, "N": config["n"],
            "fraction_e": config["fraction_e"], "chi": config.get("degree_dispersion", 1),
            "seed": config["seed"],
            **{key: run["primary"][key] for key in ("d", "nu", "energy")},
        })
    for experiment, expected in (("degree", 24), ("composition", 40)):
        assert sum(r["experiment"] == experiment for r in observations) == expected
    return {
        "archive": archive_name,
        "control": "Fixed aggregate variance G diag(f) and mean J0 diag(f); fixed target mean rates. Recompute the conditional local law as chi changes.",
        "W": W.tolist(),
        "aggregate_mean_connectivity": (np.array(degree["J0"]) * f_ref).tolist(),
        "target_mean_rates": mean_rates,
        "quadrature": {"dt": 0.08, "horizon": 100, "n_hermite": 96, "max_frequency": 12},
        "simulation_estimator": {"duration": 8000, "block_length": 500, "minimum_empty_gap": 0},
        "fraction_e": fractions.tolist(), "theory_rows": theory, "network_observations": observations,
    }


def index(values, target):
    return int(np.flatnonzero(np.isclose(values, target, rtol=0, atol=1e-9)).item())


def setup_panel(ax, letter, xlabel, ylabel):
    ax.set(xlabel=xlabel, ylabel=ylabel)
    panel_label(ax, letter)


def percent_axis(ax):
    ax.set_yticks([0, 0.04, 0.08, 0.12, 0.16])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def plot(data, dense=None):
    from sweep_records import cohort_handles, dense_records, draw_group, point_offsets

    apply_style()
    theory = data["theory_rows"]
    chi = np.array([row["chi"] for row in theory])
    f = np.array(data["fraction_e"])
    d = np.array([row["d"] for row in theory])
    nu = np.array([row["nu"] for row in theory])
    energy = np.array([row["energy"] for row in theory])
    if dense is None:
        dense = dense_records()
    records = [dict(r, cohort="original") for r in data["network_observations"]]
    for row in dense:
        cfg, primary = row["config"], row["primary"]
        records.append({"name": row["name"], "cohort": "dense", "experiment": row["job"]["experiment"],
                        "N": cfg["n"], "seed": cfg["seed"], "fraction_e": cfg["fraction_e"],
                        "chi": cfg.get("degree_dispersion", 1) if row["job"]["experiment"]=="degree" else 1,
                        "d": primary["d"], "nu": primary["nu"], "energy": primary["energy"]})

    def samples(ax, which):
        groups = {}
        for record in records:
            if record["experiment"] != which:
                continue
            x = np.sqrt(record["chi"]) if which == "degree" else record["fraction_e"]
            groups.setdefault((x, record["N"], record["cohort"]), []).append(record["d"])
        offsets = point_offsets(groups, .026 if which=="degree" else .009)
        for key, values in groups.items():
            x, n, cohort = key
            draw_group(ax, x+offsets[key], values, MARKERS[n], JOINT, cohort,
                       jitter=.004 if which=="degree" else .0013)
    degree = [r for r in records if r["experiment"] == "degree"]
    ref = index(f, 0.8)
    fig, ((ax, bx), (cx, dx)) = plt.subplots(2, 2, figsize=(8.15, 5.65), layout="constrained")

    setup_panel(ax, "A", r"Degree dispersion $\chi$ (square-root scale)", r"Joint dimension $D/N$")
    ax.plot(np.sqrt(chi), d[:, ref], color=JOINT)
    samples(ax, "degree")
    ax.set(xlim=(-.065, np.sqrt(2)+.065), ylim=(0, 1.08*max(d[:,ref].max(),max(r["d"] for r in degree))))
    ticks = [0, .05, .2, .5, 1, 1.5, 2]
    ax.set_xticks(np.sqrt(ticks), [f"{value:g}" for value in ticks])
    percent_axis(ax)
    fig.legend(handles=cohort_handles({r["N"] for r in records}, has_dense=bool(dense)), loc="outside lower center",
               ncol=5, columnspacing=1.2, handletextpad=.45)

    setup_panel(bx, "B", r"Excitatory fraction $f_E$", r"Joint dimension $D/N$")
    for value, color, ls in ((0, JOINT, "-"), (0.5, SLATE, "--"), (1, MUTED, "-."), (2, JOINT, ":")):
        bx.plot(f, d[index(chi, value)], ls=ls, color=color, label=fr"$\chi={value:g}$")
    samples(bx, "composition")
    bx.set(xlim=(f[0], f[-1]), ylim=(0, 0.19))
    bx.set_xticks([.5, .6, .7, .8, .9, .95])
    percent_axis(bx)
    bx.legend(loc="center left", ncol=2, columnspacing=1.1, fontsize=7.8)

    setup_panel(cx, "C", r"Degree dispersion $\chi$", r"Excitatory fraction $f_E$")
    heatmap = cx.pcolormesh(chi, f, 100*d.T, norm=LogNorm(100*d.min(), 100*d.max()),
                            cmap=DIMENSION_CMAP, shading="nearest", rasterized=True)
    colorbar = fig.colorbar(heatmap, ax=cx, pad=0.025, aspect=30)
    colorbar.set_label(r"$D/N$ (%)")
    colorbar.set_ticks([1.5, 2, 4, 8, 16], labels=["1.5", "2", "4", "8", "16"])
    colorbar.ax.minorticks_off()
    for cohort, marker_size, fill in [("original", 35, "white"), ("dense", 9, JOINT)]:
        tested = sorted({(r["chi"], r["fraction_e"]) for r in records if r["cohort"] == cohort})
        if tested:
            cx.scatter(*np.array(tested).T, facecolors=fill, edgecolors=JOINT,
                       linewidths=.8, marker="D", s=marker_size, clip_on=False)
    cx.set(xlim=(chi[0], chi[-1]), ylim=(f[0], f[-1]))
    cx.set_yticks([.5, .6, .7, .8, .9, .95])
    cx.set_xticks([0, .5, 1, 1.5, 2])

    setup_panel(dx, "D", r"Degree dispersion $\chi$",
                r"Relative to $\chi=0$")
    terms = [
        (nu[:, ref]**2, r"Squared mean variance $\nu^2$", MUTED, "-"),
        (energy[:, ref], r"Covariance energy $\mathcal{E}$", SLATE, "-."),
        (d[:, ref], r"Dimension $\nu^2/\mathcal{E}$", JOINT, "--"),
    ]
    for values, label, color, style in terms:
        dx.plot(chi, values / values[0], label=label, color=color, ls=style, lw=2)
    dx.set(xlim=(0, 2), yscale="log", ylim=(0.075, 2.4))
    dx.set_yticks([0.1, 0.2, 0.5, 1, 2], ["0.1×", "0.2×", "0.5×", "1×", "2×"])
    dx.set_xticks([0, .5, 1, 1.5, 2])
    dx.legend(loc="center right", bbox_to_anchor=(1, .59), fontsize=7.8)
    dx.minorticks_off()
    save_figure(fig, OUT / "PR vs degree dispersion and E-I composition")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", "--archive", type=Path, help="Recompute from the unpacked project or source ZIP")
    args = parser.parse_args()
    if args.source:
        data, dense = compute(args.source)
        DATA.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    else:
        data = json.loads(DATA.read_text())
        dense = None
    plot(data, dense=dense)
