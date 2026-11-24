"""Control and bookkeeping figures: invariance of the one-site law along the composition path,
realized in-degree dispersion, operating-point spread against size, chaos across all cells, and timing.

    python experiments/paper_campaign/plots/checks.py
"""
import numpy as np

from common import (CAMPAIGN, CROSS, E, GOLD, GRAY, I, JOINT, LIGHT, SEQ, column, fluctuating, groups, letters, load,
                    mean_se, panels, save, select, sqrt_axis, style, sweep)

G0 = np.array([[16., 9.], [25., 12.]])
J0 = np.array([[1., -6.], [2., -10.]])


def composition_local(rows):
    comp = fluctuating(select(rows, family="composition"))
    spec = [(("nu_E", "nu_I"), r"rate variance $\nu_a$"), (("q_E", "q_I"), r"temporal input variance $q_a$"),
            (("Delta_E", "Delta_I"), r"static input variance $\Delta_a$"), (("A_E", "A_I"), r"squared gain $A_a$"),
            (("rate_E", "rate_I"), r"mean rate $\bar m_a$")]
    fig, ax = panels(1, 5, width=7.2, height=2.15)
    for a, (names, ylabel) in zip(ax, spec):
        for name, color in zip(names, (E, I)):
            if "th_" + name in comp[0]:
                a.axhline(comp[0]["th_" + name], color=color, lw=1.3, zorder=2)
            elif name.startswith("rate"):
                a.axhline(.3, color=JOINT, lw=1.3, zorder=2)
            sweep(a, comp, "fraction_e", name, color=color, dots=False, ms=3)
        a.set_xlabel(r"$f_E$")
        a.set_ylabel(ylabel)
        a.set_xlim(.47, .97)
    letters(ax)
    save(fig, "check_composition_local_laws",
         "Along the composition path the one-site statistics should not change. Mean temporal rate variance, temporal "
         "input variance, static input variance, squared gain and mean rate of each population (E blue, I red) against "
         "excitatory fraction, with the single fraction-independent prediction as a horizontal line. N = 2,000, 48 "
         "networks per fraction; bars are one across-network standard error.")


def realized_dispersion(rows):
    """chi_ab = Var(K_ab) / (M p (1-p)) from each neuron's realized in-degree."""
    deg = select(rows, family="degree")
    p = J0**2 / (J0**2 + G0)
    fig, ax = panels(1, 2, width=5.2, height=2.5)
    labels = [["E to E", "I to E"], ["E to I", "I to I"]]
    colors = [[E, CROSS], [GOLD, I]]
    for chi, cell in groups(deg, "chi").items():
        acc = np.zeros((2, 2, len(cell[:6])))
        for k, r in enumerate(cell[:6]):
            with np.load(CAMPAIGN / "arrays" / f"{r['name']}.npz") as z:
                ne = int(z["n_e"])
                d = z["indegree"].astype(float)
            pools = (ne, len(d) - ne)
            for a, sl in enumerate((slice(0, ne), slice(ne, None))):
                for b in range(2):
                    acc[a, b, k] = d[sl, b].var() / (pools[b] * p[a, b] * (1 - p[a, b]))
        for a in range(2):
            for b in range(2):
                m = acc[a, b].mean()
                ax[0].plot(chi, m, "o", color=colors[a][b], ms=3.2, label=labels[a][b] if chi == 0 else None)
                ax[1].plot(chi, m - chi, "o", color=colors[a][b], ms=3.2)
    grid = np.linspace(0, 3, 50)
    ax[0].plot(grid, grid, color=GRAY, lw=.8, ls=(0, (4, 2)), zorder=0)
    ax[0].set_ylabel(r"realized dispersion $\mathrm{Var}K_{ab}/[Mp_{ab}(1-p_{ab})]$")
    ax[1].axhline(0, color=GRAY, lw=.7)
    ax[1].set_ylabel("realized minus requested")
    for a in ax:
        a.set_xlabel(r"requested dispersion $\chi$")
    ax[0].legend()
    letters(ax)
    save(fig, "check_realized_degree_dispersion",
         "The degree sampler delivers the requested dispersion. (a) Variance of the realized in-degree in each of the "
         "four connectivity blocks, divided by the Bernoulli count variance, against the requested dispersion (mean "
         "over six networks; the dashed line is equality). (b) The difference, which at zero dispersion is the "
         "stochastic rounding of non-integer mean counts. N = 2,000.")


def chaos_all(rows):
    live = [r for r in rows if np.isfinite(r["lyapunov"])]
    fig, ax = panels(1, 3, height=2.5)
    for family, color in (("degree", JOINT), ("composition", CROSS), ("size", GOLD), ("generality", I)):
        cell = select(live, family=family)
        ax[0].plot(column(cell, "th_response_radius"), column(cell, "lyapunov"), ".", color=color, ms=3, alpha=.6, mew=0, label=family)
        ax[1].plot(column(cell, "q_E"), column(cell, "lyapunov"), ".", color=color, ms=3, alpha=.6, mew=0)
        fl = fluctuating(cell)
        ax[2].plot(column(fl, "lyapunov"), column(fl, "d"), ".", color=color, ms=3, alpha=.6, mew=0)
    ax[0].set_xlabel(r"predicted radius of $\mathrm{diag}(A)\,W$")
    ax[0].set_ylabel(r"largest Lyapunov exponent $\lambda_{\max}$")
    ax[0].legend(markerscale=2)
    ax[1].set_xlabel(r"measured temporal input variance $q_E$")
    ax[1].set_ylabel(r"$\lambda_{\max}$")
    ax[1].set_xscale("log")
    ax[2].set_xlabel(r"$\lambda_{\max}$")
    ax[2].set_ylabel(r"$D/N$")
    ax[2].set_yscale("log")
    for a in ax[:2]:
        a.axhline(0, color=GRAY, lw=.7)
    letters(ax)
    save(fig, "check_lyapunov_all",
         f"Largest finite-time Lyapunov exponent of all {len(live)} networks where it was measured (one tangent vector, "
         "2,000 leak times after burn-in), against (a) the predicted spectral radius of the population response "
         "matrix and (b) the measured temporal input variance of E cells; (c) dimension against the exponent. Colors "
         "mark the experiment family. One network at a fixed point has a negative exponent and three are zero "
         "within their block-to-block scatter (summary.json lists them); the rest are positive in every time block.")


def spread_and_timing(rows):
    fig, ax = panels(1, 3, height=2.5)
    # operating-point histograms at fixed in-degree for each size
    cell = [r for r in fluctuating(rows) if r["family"] in ("size", "degree") and r["chi"] == 0. and r["fraction_e"] == .8]
    for n, g in groups(cell, "n").items():
        mu = []
        for r in g[:4]:
            with np.load(CAMPAIGN / "arrays" / f"{r['name']}.npz") as z:
                ne = int(z["n_e"])
                mu.append(z["mean_x"][:ne] - z["mean_x"][:ne].mean())
        mu = np.concatenate(mu)
        hist, edges = np.histogram(mu, bins=np.linspace(-1.5, 1.5, 121), density=True)
        ax[0].plot(.5 * (edges[1:] + edges[:-1]), hist, "-", color=SEQ(.1 + .85 * np.log2(n / 500) / 5), lw=1.1,
                   label=f"{int(n):,}")
    ax[0].set_xlabel(r"E mean input minus population mean, $\chi=0$")
    ax[0].set_ylabel("density")
    ax[0].set_xlim(-1.2, 1.2)
    ax[0].legend(title=r"$N$", fontsize=6.5, title_fontsize=7.5)
    live = [r for r in rows if r["dt"] == .05 and r["duration"] == 8000.]
    for host_rows, color in ((live, JOINT),):
        x, m, se = sweep(ax[1], host_rows, "n", "sim_seconds", color=color, dots=True)
        ax[1].plot(x, m[0] * (x / x[0])**2, color=GRAY, lw=.8, ls=(0, (4, 2)))
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"network size $N$")
    ax[1].set_ylabel("simulation seconds per network")
    ax[1].text(.05, .9, r"dashed: $\propto N^2$", transform=ax[1].transAxes, fontsize=7, color=GRAY)
    counts = {}
    for r in rows:
        counts[r["family"]] = counts.get(r["family"], 0) + 1
    ax[2].bar(range(len(counts)), list(counts.values()), color=[JOINT, GOLD, CROSS, GRAY, I][:len(counts)], width=.6)
    ax[2].set_xticks(range(len(counts)))
    ax[2].set_xticklabels(list(counts), rotation=30, ha="right")
    ax[2].set_ylabel("networks")
    letters(ax)
    save(fig, "check_spread_and_timing",
         "(a) Distribution of excitatory mean inputs at fixed in-degree, centered on the population mean, for each "
         "network size (four networks pooled per size); the leading theory predicts a point mass. (b) Simulation time "
         "per network on one core of the rented machines (8,000 leak times, step 0.05), with a quadratic guide. (c) "
         "Networks per experiment family.")


def main():
    style()
    rows = load()
    for figure in (composition_local, realized_dispersion, chaos_all, spread_and_timing):
        figure(rows)


if __name__ == "__main__":
    main()
