"""Additional views: finite-size exponents, averaging controls along composition, network-level scatter,
and a one-page overview assembled from the campaign.

    python experiments/paper_campaign/plots/extra.py
"""
import json
import sys

import numpy as np

from common import (CAMPAIGN, CROSS, E, GOLD, GRAY, I, JOINT, LIGHT, SEQ, column, fluctuating, groups, letters, load,
                    mean_se, panels, save, select, sqrt_axis, style, sweep, theory_line)
from size import CONDITIONS, condition_rows

sys.path.insert(0, str(CAMPAIGN))
import theory


def size_loglog(rows):
    """Absolute relative error against N on log axes at fixed in-degree, where every residual is resolved."""
    cell = condition_rows(rows, CONDITIONS[0][1])
    names = [("d", r"$D/N$", JOINT, "o"), ("energy", r"$\mathcal{E}$", CROSS, "s"), ("P_EE", r"$P_{EE}$", E, "D"),
             ("nu", r"$\nu$", GOLD, "^"), ("A_E", r"$A_E$", I, "v"), ("q_E", r"$q_E$", GRAY, "<")]
    fig, ax = panels(1, 3, height=2.6)
    fits = {}
    for name, label, color, marker in names:
        xs, m, se = [], [], []
        for n, g in groups(cell, "n").items():
            v = column(g, name) / g[0]["th_" + name] - 1
            mm, ss, k = mean_se(v)
            xs.append(n), m.append(abs(mm)), se.append(ss if k > 1 else 0)
        xs, m, se = map(np.array, (xs, m, se))
        w = 1 / np.maximum(se / m, 1e-3)**2
        slope, intercept = np.polyfit(np.log(xs), np.log(m), 1, w=np.sqrt(w))
        fits[name] = slope
        ax[0].errorbar(xs, m, yerr=se, fmt=marker, color=color, ms=3.4, elinewidth=.8, capsize=1.5,
                       label=f"{label}: $N^{{{slope:.2f}}}$")
        ax[0].plot(xs, np.exp(intercept) * xs**slope, "-", color=color, lw=.8, alpha=.8)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"network size $N$")
    ax[0].set_ylabel(r"|relative error| at $\chi=0$")
    ax[0].legend(fontsize=6.3, ncol=1, loc="upper right")
    ax[0].set_ylim(None, 3)
    # static spread of mean inputs at fixed in-degree: zero in the leading theory
    for name, color, label in (("Delta_E", E, r"$\Delta_E$"), ("Delta_I", I, r"$\Delta_I$")):
        x, m, se = sweep(ax[1], cell, "n", name, color=color, label=label, dots=False)
        slope = np.polyfit(np.log(x), np.log(m), 1)[0]
        ax[1].plot(x, m, "-", color=color, lw=.8)
        ax[1].text(x[-1] * 1.05, m[-1], f"$N^{{{slope:.2f}}}$", color=color, fontsize=7, va="center")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(400, 40000)
    ax[1].set_xlabel(r"network size $N$")
    ax[1].set_ylabel(r"static input variance at $\chi=0$")
    ax[1].legend()
    # excess pair energy in absolute units
    for name, color, label in (("P_EE", E, r"$P_{EE}$"), ("P_EI", CROSS, r"$P_{EI}$"), ("P_II", I, r"$P_{II}$")):
        x, m, se = sweep(ax[2], cell, "n", name, color=color, label=label, dots=False,
                         transform=lambda v, c, n=name: v - c[0]["th_" + n])
        ax[2].plot(x, m, "-", color=color, lw=.8)
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel(r"network size $N$")
    ax[2].set_ylabel(r"excess pair energy at $\chi=0$")
    ax[2].legend()
    letters(ax)
    save(fig, "size_fixed_degree_exponents",
         "How the fixed in-degree residuals vanish with network size. (a) Absolute relative error of the "
         "across-network mean against the large-N prediction, with weighted power-law fits (exponents in the "
         "legend). (b) Across-neuron variance of mean inputs, which the leading theory sets to zero at fixed "
         "in-degree. (c) Measured minus predicted distinct-pair energy. Bars are one across-network standard error; "
         "fits are descriptive, over the sizes shown.")
    return fits


def composition_averaging(rows):
    curves = json.loads((CAMPAIGN / "theory_curves.json").read_text())
    th, ctl = curves["composition"], curves["composition_controls"]
    col = lambda t, n: np.array([r[n] for r in t], dtype=float)
    comp = fluctuating(select(rows, family="composition"))
    fig, ax = panels(1, 3, height=2.4)
    for a, name, tname, ylabel in ((ax[0], "d", "d", r"$D/N$"), (ax[1], "dE", "dE", r"$D_E/N_E$"), (ax[2], "dI", "dI", r"$D_I/N_I$")):
        theory_line(a, col(ctl, "fraction_e"), col(ctl, f"{tname}_gains"), color=LIGHT, ls=(0, (1, 1.2)), label="gains averaged")
        theory_line(a, col(ctl, "fraction_e"), col(ctl, f"{tname}_spectra"), color=GRAY, ls=(0, (4, 1.5)), label="spectra averaged")
        theory_line(a, col(th, "fraction_e"), col(th, tname), label="conditional theory")
        sweep(a, comp, "fraction_e", name, jitter=.006)
        a.set_yscale("log")
        a.set_xlabel(r"excitatory fraction $f_E$")
        a.set_ylabel(ylabel)
        a.set_xlim(.47, .97)
    ax[0].legend(fontsize=6.5, loc="center left")
    letters(ax)
    save(fig, "composition_averaging_controls",
         "Premature averaging fails along the whole composition path. Joint (a), E-only (b) and I-only (c) dimension "
         "per neuron against excitatory fraction, with the conditional prediction and the two approximations that "
         "average spectra or gains before taking second moments. N = 2,000, 48 seed-matched networks per fraction.")


def network_scatter(rows):
    deg = fluctuating(select(rows, family="degree"))
    fig, ax = panels(1, 3, height=2.5)
    chis = np.array([r["chi"] for r in deg])
    colors = SEQ(.1 + .85 * chis / 3)
    x = np.array([r["nu"] / r["th_nu"] for r in deg])
    y = np.array([r["energy"] / r["th_energy"] for r in deg])
    d = np.array([r["d"] / r["th_d"] for r in deg])
    ax[0].scatter(x, y, c=colors, s=5, linewidths=0, rasterized=True)
    grid = np.linspace(x.min(), x.max(), 50)
    ax[0].plot(grid, grid**2, color=GRAY, lw=.8, ls=(0, (4, 2)))
    ax[0].text(.05, .93, r"dashed: $\mathcal{E}\propto\nu^2$ (unchanged $D/N$)", transform=ax[0].transAxes, fontsize=6.8, color=GRAY)
    ax[0].set_xlabel(r"$\nu$ / predicted")
    ax[0].set_ylabel(r"$\mathcal{E}$ / predicted")
    ax[1].scatter(np.array([r["P_EE"] / r["th_P_EE"] for r in deg]), d, c=colors, s=5, linewidths=0, rasterized=True)
    ax[1].set_xlabel(r"$P_{EE}$ / predicted")
    ax[1].set_ylabel(r"$D/N$ / predicted")
    ax[2].scatter(np.array([r["A_E"] / r["th_A_E"] for r in deg]), d, c=colors, s=5, linewidths=0, rasterized=True)
    ax[2].set_xlabel(r"$A_E$ / predicted")
    ax[2].set_ylabel(r"$D/N$ / predicted")
    for a in ax:
        a.axhline(1, color=LIGHT, lw=.6, zorder=0)
        a.axvline(1, color=LIGHT, lw=.6, zorder=0)
        a.set_xscale("log")
        a.set_yscale("log")
    letters(ax)
    save(fig, "degree_network_scatter",
         "Network-by-network residuals along the degree sweep (360 networks; light to dark is dispersion 0 to 3). "
         "(a) Covariance energy against rate variance, each divided by its prediction; networks move along the dashed "
         "curve when amplitude changes without changing dimension. (b,c) Dimension residual against the residual of "
         "the E-E pair energy and of the E squared gain. N = 2,000.")


def overview(rows):
    """Candidate summary figure: local law, degree curve with controls, size scaling, composition, lag slice."""
    curves = json.loads((CAMPAIGN / "theory_curves.json").read_text())
    fig, ax = panels(2, 3, height=4.5)
    # (a) conditional variance law at chi = 1
    deg1 = fluctuating(select(rows, family="degree", chi=1.))[:8]
    summary, th = theory.solve(1., .3, 1.)
    for a_, (pop, color) in enumerate((("E", E), ("I", I))):
        mu, v = [], []
        for r in deg1:
            with np.load(CAMPAIGN / "arrays" / f"{r['name']}.npz") as z:
                ne = int(z["n_e"])
                sl = slice(0, ne) if a_ == 0 else slice(ne, None)
                mu.append(z["mean_x"][sl]), v.append(z["variance_rate"][sl])
        mu, v = np.concatenate(mu), np.concatenate(v)
        pick = np.random.default_rng(0).choice(len(mu), min(3000, len(mu)), replace=False)
        ax[0, 0].plot(mu[pick], v[pick], ".", color=color, ms=1.6, alpha=.2, mew=0, rasterized=True)
        ax[0, 0].plot(th["mu"][a_], th["covariance"][a_, :, 0], "-", color=color, lw=1.3, label=pop)
    ax[0, 0].set_xlim(-9, 4)
    ax[0, 0].set_xlabel(r"time-averaged input $\mu_i$")
    ax[0, 0].set_ylabel(r"temporal rate variance $v_i$")
    ax[0, 0].legend()
    # (b) degree curve with controls
    td = curves["degree"]
    chi = np.array([r["chi"] for r in td])
    col = lambda n: np.array([r[n] for r in td], dtype=float)
    theory_line(ax[0, 1], chi, col("d_gains"), color=LIGHT, ls=(0, (1, 1.2)), label="gains averaged")
    theory_line(ax[0, 1], chi, col("d_spectra"), color=GRAY, ls=(0, (4, 1.5)), label="spectra averaged")
    theory_line(ax[0, 1], chi, col("d"), label="conditional theory")
    sweep(ax[0, 1], fluctuating(select(rows, family="degree")), "chi", "d", dots=False)
    sqrt_axis(ax[0, 1])
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_ylim(6e-3, .4)
    ax[0, 1].set_xlabel(r"degree dispersion $\chi$")
    ax[0, 1].set_ylabel(r"$D/N$")
    ax[0, 1].legend(fontsize=6.3, loc="lower left")
    # (c) finite-size residual of D/N
    colors = [SEQ(.15), SEQ(.4), SEQ(.62), SEQ(.95), GOLD]
    ax[0, 2].axhline(0, color=GRAY, lw=.7, ls=(0, (4, 2)))
    for (_, where, label), color, marker in zip(CONDITIONS, colors, "osD^v"):
        cell = condition_rows(rows, where)
        x, m, se = sweep(ax[0, 2], cell, "n", "d", color=color, marker=marker, label=label, dots=False,
                         transform=lambda v, c: v / c[0]["th_d"] - 1, xmap=lambda n: n**-.5, ms=3.2, stat="median")
        ax[0, 2].plot(x, m, "-", color=color, lw=.7)
    ax[0, 2].set_xlim(0, .048)
    ax[0, 2].set_ylim(-.42, .15)
    ax[0, 2].set_xlabel(r"$N^{-1/2}$")
    ax[0, 2].set_ylabel(r"relative error of $D/N$")
    ax[0, 2].legend(fontsize=6.2, loc="lower left")
    # (d,e) composition: total E dimension and pair energies
    tc = curves["composition"]
    f = np.array([r["fraction_e"] for r in tc])
    cc = lambda n: np.array([r[n] for r in tc], dtype=float)
    comp = fluctuating(select(rows, family="composition"))
    theory_line(ax[1, 0], f, cc("dE_total"), color=E)
    sweep(ax[1, 0], comp, "fraction_e", "dE_total", color=E, dots=False)
    ax[1, 0].set_xlabel(r"excitatory fraction $f_E$")
    ax[1, 0].set_ylabel(r"total E dimension $D_E/N$")
    for name, color, label in (("P_EE", E, r"$P_{EE}$"), ("P_II", I, r"$P_{II}$")):
        theory_line(ax[1, 1], f, cc(name), color=color)
        sweep(ax[1, 1], comp, "fraction_e", name, color=color, label=label, dots=False)
    ax[1, 1].set_ylim(.4, 1.3)
    ax[1, 1].set_xlabel(r"excitatory fraction $f_E$")
    ax[1, 1].set_ylabel(r"distinct-pair energy")
    ax[1, 1].legend()
    for a in ax[1, :2]:
        a.set_xlim(.47, .97)
    # (f) equal-lag pair moment at chi = 1
    lag = theory.lag_map(1., .3, 1.)
    maps = {k: [] for k in ("psi_EE", "psi_EI", "psi_II")}
    for r in fluctuating(select(rows, family="degree", chi=1.)):
        with np.load(CAMPAIGN / "arrays" / f"{r['name']}.npz") as z:
            if "psi_EE" in z.files:
                for k in maps:
                    maps[k].append(np.diag(z[k]))
    for (key, a_, b_, label), color in zip((("psi_EE", 0, 0, "E-E"), ("psi_EI", 0, 1, "E-I"), ("psi_II", 1, 1, "I-I")), (E, CROSS, I)):
        if maps[key]:
            m = np.mean(maps[key], 0)
            se = np.std(maps[key], 0, ddof=1) / np.sqrt(len(maps[key]))
            ax[1, 2].plot(lag["lags"], np.diag(lag["psi"][a_, b_]), "-", color=color, lw=1.3)
            ax[1, 2].errorbar(lag["lags"][::2], m[::2], yerr=se[::2], fmt="o", color=color, ms=2.8, elinewidth=.7, label=label)
    ax[1, 2].set_xlabel(r"lag $\tau$ (leak times)")
    ax[1, 2].set_ylabel(r"$N\langle C_{ij}(\tau)^2\rangle_{i\neq j}$")
    ax[1, 2].legend()
    letters(ax)
    save(fig, "overview",
         "Conditional theory against 1,694 simulated networks, one page. (a) Each neuron's rate variance against its "
         "mean input at Bernoulli dispersion (dots; eight networks) with the conditional law (lines). (b) Dimension "
         "per neuron against degree dispersion, N = 2,000, with the conditional prediction and the two "
         "premature-averaging approximations. (c) Relative error of D/N against inverse square root of network size "
         "at five settings, N = 500 to 16,000 (medians with bootstrap standard errors). (d) Total E dimension per network neuron against excitatory fraction. "
         "(e) Squared covariance per distinct E-E and I-I pair against excitatory fraction. (f) Equal-lag pair moment "
         "against lag for E-E, E-I and I-I pairs. Lines are parameter-only predictions; markers are across-network "
         "means with one standard error.")


def main():
    style()
    rows = load()
    fits = size_loglog(rows)
    print("fixed in-degree exponents", {k: round(v, 3) for k, v in fits.items()})
    composition_averaging(rows)
    network_scatter(rows)
    overview(rows)


if __name__ == "__main__":
    main()
