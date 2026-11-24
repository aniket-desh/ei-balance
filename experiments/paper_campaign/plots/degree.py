"""Degree-dispersion sweep at 80:20 E:I, N = 2000: 15 dispersions x 24 networks.

    python experiments/paper_campaign/plots/degree.py

Every panel: solid line = parameter-only conditional theory; faint dots = networks;
markers = across-network mean with one standard error.
"""
import json

import numpy as np

from common import (CAMPAIGN, CROSS, E, GOLD, GRAY, I, JOINT, LIGHT, column, fluctuating, groups, letters, load, mean_se,
                    panels, save, select, sqrt_axis, style, sweep, theory_line)

XL = r"degree dispersion $\chi$"
N_NOTE = "N = 2,000, 80:20 E:I, 24 networks per dispersion, 8,000 recorded leak times."


def data():
    rows = fluctuating(select(load(), family="degree"))
    th = json.loads((CAMPAIGN / "theory_curves.json").read_text())["degree"]
    return rows, th, np.array([r["chi"] for r in th])


def tcol(th, name):
    return np.array([r[name] for r in th], dtype=float)


def dimension(rows, th, chi):
    fig, ax = panels(1, 3, height=2.35)
    for a, logy in zip(ax[:2], (False, True)):
        theory_line(a, chi, tcol(th, "d_gains"), color=LIGHT, ls=(0, (1, 1.2)), label="gains averaged")
        theory_line(a, chi, tcol(th, "d_spectra"), color=GRAY, ls=(0, (4, 1.5)), label="spectra averaged")
        theory_line(a, chi, tcol(th, "d"), label="conditional theory")
        sweep(a, rows, "chi", "d", label="networks")
        sqrt_axis(a)
        a.set_xlabel(XL)
        a.set_ylabel(r"dimension per neuron $D/N$")
        if logy:
            a.set_yscale("log")
            a.set_ylim(6e-3, .6)
        else:
            a.set_ylim(0, .26)
    ax[0].legend(loc="upper right")
    ax[2].axhline(1, color=GRAY, lw=.7, ls=(0, (4, 2)))
    sweep(ax[2], rows, "chi", "d", transform=lambda v, cell: v / cell[0]["th_d"])
    sqrt_axis(ax[2])
    ax[2].set_xlabel(XL)
    ax[2].set_ylabel(r"measured / predicted $D/N$")
    letters(ax)
    save(fig, "degree_dimension",
         "Degree dispersion lowers activity dimension along the conditional prediction. (a) Joint dimension per "
         "neuron against in-degree dispersion; the solid line is the parameter-only conditional theory, the dashed "
         "and dotted lines average the conditional spectra or the gains before taking second moments. (b) The same "
         "on a logarithmic axis. (c) Ratio of each network's dimension to the prediction. Dots are networks, markers "
         "are means with one across-network standard error. " + N_NOTE)


def ingredients(rows, th, chi):
    fig, ax = panels(2, 3, height=4.3)
    spec = [(ax[0, 0], [("nu_E", E, r"$\nu_E$"), ("nu_I", I, r"$\nu_I$")], r"temporal rate variance $\nu_a$", 1),
            (ax[0, 1], [("E_EE", E, r"$\mathcal{E}_{EE}$"), ("E_EI", CROSS, r"$2\mathcal{E}_{EI}$"),
                        ("E_II", I, r"$\mathcal{E}_{II}$")], r"covariance energy per neuron", 1),
            (ax[0, 2], [("P_EE", E, r"$P_{EE}$"), ("P_EI", CROSS, r"$P_{EI}$"), ("P_II", I, r"$P_{II}$")],
             r"distinct-pair energy $N\langle C_{ij}^2\rangle$", 1),
            (ax[1, 0], [("Delta_E", E, r"$\Delta_E$"), ("Delta_I", I, r"$\Delta_I$")],
             r"static input variance $\Delta_a$", 1),
            (ax[1, 1], [("q_E", E, r"$q_E$"), ("q_I", I, r"$q_I$")], r"temporal input variance $q_a$", 1),
            (ax[1, 2], [("A_E", E, r"$A_E$"), ("A_I", I, r"$A_I$")], r"squared gain $A_a=\langle\alpha^2\rangle$", 1)]
    for a, series, ylabel, _ in spec:
        for name, color, label in series:
            factor = 2. if name == "E_EI" else 1.
            theory_line(a, chi, factor * tcol(th, name), color=color)
            sweep(a, rows, "chi", name, color=color, label=label, transform=lambda v, c, f=factor: f * v)
        sqrt_axis(a)
        a.set_xlabel(XL)
        a.set_ylabel(ylabel)
        a.legend(loc="best", ncol=1)
    letters(ax)
    save(fig, "degree_ingredients",
         "The ingredients of the dimension along the degree sweep. (a) Mean temporal rate variance of each "
         "population. (b) Covariance energies per network neuron; the E-I block is counted twice. (c) N times the "
         "mean squared covariance over distinct pairs. (d) Across-neuron variance of mean input, estimated across "
         "disjoint 500-time windows. (e) Temporal input variance. (f) Mean squared time-averaged gain. Lines are "
         "parameter-only predictions. " + N_NOTE)


def ratios(rows, th, chi):
    names = [("nu", r"$\nu$"), ("energy", r"$\mathcal{E}$"), ("d", r"$D/N$"), ("dE", r"$D_E/N_E$"),
             ("dI", r"$D_I/N_I$"), ("P_EE", r"$P_{EE}$"), ("P_EI", r"$P_{EI}$"), ("P_II", r"$P_{II}$"),
             ("q_E", r"$q_E$"), ("q_I", r"$q_I$"), ("A_E", r"$A_E$"), ("A_I", r"$A_I$")]
    fig, ax = panels(3, 4, height=5.4, sharex=True)
    for a, (name, label) in zip(ax.ravel(), names):
        a.axhline(1, color=GRAY, lw=.7, ls=(0, (4, 2)))
        color = E if name.endswith("_E") or name in ("dE", "P_EE") else I if name.endswith("_I") or name in ("dI", "P_II") \
            else CROSS if name == "P_EI" else JOINT
        sweep(a, rows, "chi", name, color=color, transform=lambda v, c, n=name: v / c[0]["th_" + n], ms=3)
        sqrt_axis(a, ticks=(0, .5, 1, 2, 3))
        a.set_title(label)
    for a in ax[-1]:
        a.set_xlabel(XL)
    for a in ax[:, 0]:
        a.set_ylabel("measured / predicted")
    letters(ax)
    save(fig, "degree_ratios",
         "Measured-to-predicted ratios along the degree sweep. Each panel divides one network statistic by its "
         "parameter-only prediction at the same dispersion; the dashed line marks agreement. " + N_NOTE)


def population_dimensions(rows, th, chi):
    fig, ax = panels(1, 2, width=5.2, height=2.4)
    for name, color, label in (("dE", E, r"$D_E/N_E$"), ("dI", I, r"$D_I/N_I$"), ("d", JOINT, r"$D/N$")):
        for a in ax:
            theory_line(a, chi, tcol(th, name), color=color)
            sweep(a, rows, "chi", name, color=color, label=label)
    for a in ax:
        sqrt_axis(a)
        a.set_xlabel(XL)
        a.set_ylabel("dimension per neuron")
    ax[1].set_yscale("log")
    ax[0].legend()
    letters(ax)
    save(fig, "degree_population_dimensions",
         "E-only, I-only and joint dimension per corresponding neuron along the degree sweep, on linear (a) and "
         "logarithmic (b) axes. " + N_NOTE)


def factors(rows, th, chi):
    fig, ax = panels(1, 3, height=2.35)
    for a, name, label in ((ax[0], "het_factor", r"variance heterogeneity $\nu^2/\mathcal{E}_{\rm diag}$"),
                           (ax[1], "pair_factor", r"pair correlations $\mathcal{E}_{\rm diag}/\mathcal{E}$")):
        theory_line(a, chi, tcol(th, name))
        sweep(a, rows, "chi", name)
        sqrt_axis(a)
        a.set_xlabel(XL)
        a.set_ylabel(label)
        a.set_yscale("log")
    for name, color, label in (("het_factor", GOLD, "variance heterogeneity"), ("pair_factor", CROSS, "pair correlations")):
        ref = tcol(th, name)[0]
        theory_line(ax[2], chi, tcol(th, name) / ref, color=color)
        base = np.mean(column(select(rows, chi=0.), name))
        sweep(ax[2], rows, "chi", name, color=color, label=label, transform=lambda v, c, b=base: v / b, dots=False)
    sqrt_axis(ax[2])
    ax[2].set_yscale("log")
    ax[2].set_xlabel(XL)
    ax[2].set_ylabel(r"factor relative to $\chi=0$")
    ax[2].legend()
    letters(ax)
    save(fig, "degree_factors",
         "Two routes to low dimension. D/N factors into a variance-heterogeneity term and a pair-correlation term "
         "(manuscript Eq. degreefactors). (a,b) Each factor against dispersion. (c) Both relative to their value at "
         "fixed in-degree; measured factors are normalized by the measured mean at zero dispersion. " + N_NOTE)


def residual_partition(rows, th, chi):
    """log(d/d0) = 2 log(nu/nu0) - sum_b (E_b - E_b0) log(E/E0)/(E - E0), per network."""
    terms = {k: [] for k in ("rate variance", "E-E", "2 x E-I", "I-I", "total")}
    xs = []
    for value, cell in groups(rows, "chi").items():
        parts = {k: [] for k in terms}
        for r in cell:
            e, e0 = r["energy"], r["th_energy"]
            coef = np.log(e / e0) / (e - e0) if abs(e - e0) > 1e-12 else 1 / e0
            parts["rate variance"].append(2 * np.log(r["nu"] / r["th_nu"]))
            parts["E-E"].append(-(r["E_EE"] - r["th_E_EE"]) * coef)
            parts["2 x E-I"].append(-2 * (r["E_EI"] - r["th_E_EI"]) * coef)
            parts["I-I"].append(-(r["E_II"] - r["th_E_II"]) * coef)
            parts["total"].append(np.log(r["d"] / r["th_d"]))
            assert abs(sum(parts[k][-1] for k in ("rate variance", "E-E", "2 x E-I", "I-I")) - parts["total"][-1]) < 1e-9
        xs.append(value)
        for k in terms:
            terms[k].append(mean_se(parts[k])[:2])
    fig, ax = panels(1, 2, height=2.5)
    colors = {"rate variance": GOLD, "E-E": E, "2 x E-I": CROSS, "I-I": I, "total": JOINT}
    for k, values in terms.items():
        m, se = np.array(values).T
        ax[0].errorbar(xs, m, yerr=se, fmt="o-" if k != "total" else "s-", color=colors[k], ms=3.2, lw=1,
                       elinewidth=.8, capsize=1.5, label=k)
    ax[0].axhline(0, color=GRAY, lw=.7)
    sqrt_axis(ax[0])
    ax[0].set_xlabel(XL)
    ax[0].set_ylabel(r"contribution to $\log(d/d_{\rm theory})$")
    ax[0].legend(ncol=2, loc="lower right")
    # distinct-pair versus diagonal part of the E-E term
    for name, label, ls in (("pair", "distinct E-E pairs", "o-"), ("diag", "E diagonal", "s--")):
        ms_, ses = [], []
        for value, cell in groups(rows, "chi").items():
            v = []
            for r in cell:
                e, e0 = r["energy"], r["th_energy"]
                coef = np.log(e / e0) / (e - e0)
                dm, d0 = r["diag_E"], r["th_diag_E"]
                v.append(-((r["E_EE"] - dm) - (r["th_E_EE"] - d0)) * coef if name == "pair" else -(dm - d0) * coef)
            m, se, _ = mean_se(v)
            ms_.append(m), ses.append(se)
        ax[1].errorbar(xs, ms_, yerr=ses, fmt=ls, color=E, ms=3.2, lw=1, elinewidth=.8, capsize=1.5, label=label,
                       mfc=E if name == "pair" else "white")
    ax[1].axhline(0, color=GRAY, lw=.7)
    sqrt_axis(ax[1])
    ax[1].set_xlabel(XL)
    ax[1].set_ylabel("E-E contribution")
    ax[1].legend()
    letters(ax)
    save(fig, "degree_residual_partition",
         "Which covariance terms carry the dimension error. Each network's log dimension error is partitioned exactly "
         "into a rate-variance term and the three block-energy terms (experiments/degree_residual). (a) Mean "
         "contribution of each term; squares give the total. (b) The E-E term split into distinct pairs and the "
         "same-neuron diagonal. Bars are one across-network standard error. " + N_NOTE)


def dynamics(rows, th, chi):
    fig, ax = panels(1, 3, height=2.35)
    theory_line(ax[0], chi, tcol(th, "overlap"))
    theory_line(ax[0], chi, tcol(th, "overlap_gains"), color=LIGHT, ls=(0, (1, 1.2)), label="gains averaged")
    sweep(ax[0], rows, "chi", "overlap")
    ax[0].set_ylabel(r"temporal Gram alignment $\mathcal{O}_{EI}$")
    ax[0].legend(loc="lower right")
    sweep(ax[1], rows, "chi", "lyapunov", color=GOLD)
    ax[1].axhline(0, color=GRAY, lw=.7)
    ax[1].set_ylabel(r"largest Lyapunov exponent $\lambda_{\max}$")
    theory_line(ax[2], chi, tcol(th, "response_radius"), label=r"radius of $\mathrm{diag}(A)\,W$")
    ax[2].axhline(1, color=GRAY, lw=.7, ls=(0, (4, 2)))
    ax[2].set_ylabel("predicted response radius")
    ax[2].set_ylim(.7, 1.02)
    for a in ax:
        sqrt_axis(a)
        a.set_xlabel(XL)
    letters(ax)
    save(fig, "degree_dynamics",
         "Alignment and chaos along the degree sweep. (a) Normalized E-I covariance energy (linear CKA of the two "
         "temporal Gram matrices) with its prediction. (b) Largest finite-time Lyapunov exponent of the midpoint map, "
         "six networks per dispersion, 2,000 leak times after burn-in. (c) Predicted spectral radius of the "
         "population response matrix; the averaging proposition assumes it is below one. " + N_NOTE)


def spread(rows, th, chi):
    fig, ax = panels(1, 3, height=2.35)
    xs, cv, skew = [], [], []
    for value, cell in groups(rows, "chi").items():
        v = column(cell, "d")
        xs.append(value)
        cv.append(v.std(ddof=1) / v.mean())
        skew.append(np.mean(((v - v.mean()) / v.std(ddof=1))**3))
        ax[0].plot(np.sort(v) / cell[0]["th_d"], (np.arange(len(v)) + .5) / len(v), "-",
                   color=__import__("common").SEQ(value / 3), lw=1)
    ax[0].axvline(1, color=GRAY, lw=.7, ls=(0, (4, 2)))
    ax[0].set_xlabel(r"$D/N$ relative to prediction")
    ax[0].set_ylabel("fraction of networks")
    ax[0].set_title(r"light to dark: $\chi=0$ to 3", fontsize=7.5)
    ax[1].plot(xs, cv, "o-", color=JOINT, ms=3.2)
    ax[1].set_ylabel("across-network CV of $D/N$")
    ax[2].axhline(1, color=GRAY, lw=.7, ls=(0, (4, 2)))
    sweep(ax[2], rows, "chi", "d", color=GRAY, label="T = 4,000 / 8,000", dots=False,
          transform=lambda v, c: column(c, "d_T4000") / v)
    sweep(ax[2], rows, "chi", "d", color=GOLD, label="one empty block / none", dots=False, marker="s",
          transform=lambda v, c: column(c, "d_gap1") / v)
    ax[2].set_ylabel("ratio of dimension estimates")
    ax[2].legend()
    for a in ax[1:]:
        sqrt_axis(a)
        a.set_xlabel(XL)
    letters(ax)
    save(fig, "degree_spread_and_sensitivity",
         "Network-to-network spread and recording sensitivity along the degree sweep. (a) Empirical distribution of "
         "D/N over 24 networks at each dispersion, divided by the prediction. (b) Coefficient of variation across "
         "networks. (c) Ratio of the dimension estimated from the first 4,000 leak times, or with one empty window "
         "between estimator windows, to the primary estimate. " + N_NOTE)


def readout(rows, th, chi):
    fig, ax = panels(1, 2, width=5.2, height=2.4)
    sweep(ax[0], rows, "chi", "PR_E100", color=E, label="100 E cells")
    sweep(ax[0], rows, "chi", "PR_I100", color=I, label="100 I cells")
    ax[0].set_ylabel("PR of a 100-cell readout")
    ax[0].legend()
    for name, color in (("dE", E), ("dI", I)):
        theory_line(ax[1], chi, tcol(th, name) * (1600 if name == "dE" else 400), color=color)
        sweep(ax[1], rows, "chi", name, color=color, transform=lambda v, c, k=(1600 if name == "dE" else 400): v * k)
    ax[1].set_ylabel(r"whole-population dimension $D_a$")
    ax[1].set_yscale("log")
    for a in ax:
        sqrt_axis(a)
        a.set_xlabel(XL)
    letters(ax)
    save(fig, "degree_readout",
         "Fixed-size readouts along the degree sweep. (a) Participation ratio of 100 uniformly sampled E or I cells "
         "(mean of eight readouts per network). (b) Dimension of each whole population, 1,600 E and 400 I cells, "
         "with predictions. " + N_NOTE)


def main():
    style()
    args = data()
    for figure in (dimension, ingredients, ratios, population_dimensions, factors, residual_partition, dynamics,
                   spread, readout):
        figure(*args)


if __name__ == "__main__":
    main()
