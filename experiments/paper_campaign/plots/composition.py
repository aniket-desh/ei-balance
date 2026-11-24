"""Composition sweep at fixed J0 F and G F, Bernoulli edges, N = 2000: 13 fractions x 48 matched seeds.

    python experiments/paper_campaign/plots/composition.py

The same 48 seed labels are used at every fraction, so differences between two
fractions are paired by seed. Readouts are uniform K-cell subsets, nested in K.
"""
import json

import numpy as np

from common import (CAMPAIGN, CROSS, E, GOLD, GRAY, I, JOINT, LIGHT, SEQ, column, fluctuating, groups, letters, load, mean_se,
                    panels, save, select, style, sweep, t95, theory_line)

XL = r"excitatory fraction $f_E$"
NOTE = "N = 2,000, Bernoulli edges, 48 seed-matched networks per fraction, 8,000 recorded leak times."
SIZES = (25, 50, 100, 200)


def data():
    rows = fluctuating(select(load(), family="composition"))
    curves = json.loads((CAMPAIGN / "theory_curves.json").read_text())
    return rows, curves


def col(th, name):
    return np.array([np.nan if r.get(name) is None else r[name] for r in th], dtype=float)


def frac_axis(ax):
    ax.set_xlim(.47, .97)
    ax.set_xticks([.5, .6, .7, .8, .9])
    ax.set_xlabel(XL)


def moments(rows, curves):
    th = curves["composition"]
    f = col(th, "fraction_e")
    fig, ax = panels(1, 4, width=7.2, height=2.2)
    for a, (name, label) in zip(ax, [("nu", r"$\nu$"), ("energy", r"$\mathcal{E}$"), ("d", r"$D/N$")]):
        a.axhline(1, color=GRAY, lw=.7, ls=(0, (4, 2)))
        sweep(a, rows, "fraction_e", name, transform=lambda v, c, n=name: v / c[0]["th_" + n], jitter=.006)
        a.set_ylabel(f"measured / predicted {label}")
        frac_axis(a)
    ctl = curves["composition_controls"]
    theory_line(ax[3], col(ctl, "fraction_e"), col(ctl, "overlap_gains"), color=LIGHT, ls=(0, (1, 1.2)),
                label="gains averaged")
    theory_line(ax[3], f, col(th, "overlap"), label="conditional theory")
    sweep(ax[3], rows, "fraction_e", "overlap", jitter=.006)
    ax[3].set_ylabel(r"temporal Gram alignment $\mathcal{O}_{EI}$")
    ax[3].legend(loc="lower left")
    frac_axis(ax[3])
    letters(ax)
    save(fig, "composition_moments",
         "Composition changes temporal alignment while the predicted moments stay accurate. (a-c) Mean rate variance, "
         "covariance energy and joint dimension divided by their predictions. (d) E-I temporal Gram alignment with "
         "its prediction; the dotted line averages the gains first. " + NOTE)


def dimensions(rows, curves):
    th = curves["composition"]
    f = col(th, "fraction_e")
    fig, ax = panels(1, 3, height=2.4)
    for name, color, label in (("dE", E, r"$D_E/N_E$"), ("dI", I, r"$D_I/N_I$"), ("d", JOINT, r"$D/N$")):
        theory_line(ax[0], f, col(th, name), color=color)
        sweep(ax[0], rows, "fraction_e", name, color=color, label=label, jitter=.006)
    ax[0].set_ylabel("dimension per neuron")
    ax[0].set_yscale("log")
    ax[0].legend(loc="upper left")
    theory_line(ax[1], f, col(th, "dE_total"), color=E)
    sweep(ax[1], rows, "fraction_e", "dE_total", color=E, jitter=.006)
    k = np.nanargmax(col(th, "dE_total"))
    ax[1].plot(f[k], col(th, "dE_total")[k], "D", mfc="white", mec=E, ms=5, zorder=6)
    ax[1].set_ylabel(r"E dimension per network neuron $D_E/N$")
    ax[1].set_ylim(.018, .046)
    theory_line(ax[2], f, (1 - f) * col(th, "dI"), color=I)
    sweep(ax[2], rows, "fraction_e", "dI", color=I, jitter=.006, transform=lambda v, c: v * (1 - c[0]["fraction_e"]))
    ax[2].set_ylabel(r"I dimension per network neuron $D_I/N$")
    for a in ax:
        frac_axis(a)
    letters(ax)
    save(fig, "composition_dimensions",
         "Per-neuron and total population dimensions against excitatory fraction. (a) E-only, I-only and joint "
         "dimension per corresponding neuron. (b) Total E dimension divided by network size; the hollow diamond marks "
         "the predicted maximum. (c) Total I dimension divided by network size. " + NOTE)


def pairs(rows, curves):
    th = curves["composition"]
    f = col(th, "fraction_e")
    fig, ax = panels(1, 3, height=2.4)
    for a, name, color in zip(ax, ("P_EE", "P_II", "P_EI"), (E, I, CROSS)):
        theory_line(a, f, col(th, name), color=color)
        sweep(a, rows, "fraction_e", name, color=color, jitter=.006)
        if name != "P_EI":
            k = np.nanargmin(col(th, name))
            a.plot(f[k], col(th, name)[k], "D", mfc="white", mec=color, ms=5, zorder=6)
        a.set_ylabel({"P_EE": r"$P_{EE}$", "P_II": r"$P_{II}$", "P_EI": r"$P_{EI}$"}[name]
                     + r" $=N\langle C_{ij}^2\rangle_{i\neq j}$")
        lo, hi = np.nanmin(col(th, name)), np.nanpercentile(col(th, name), 97)
        a.set_ylim(.6 * lo, 1.6 * hi)
        frac_axis(a)
    letters(ax)
    save(fig, "composition_pair_energy",
         "Squared covariance per distinct pair against excitatory fraction, for E-E (a), I-I (b) and E-I (c) pairs. "
         "Hollow diamonds mark the predicted minima of the within-population pair energies. " + NOTE)


def readout_curves(rows, curves):
    th = curves["readout"]
    f = col(th, "fraction_e")
    fig, ax = panels(2, 4, width=7.2, height=4.0, sharex=True)
    for row, pop, color in ((0, "E", E), (1, "I", I)):
        for a, k in zip(ax[row], SIZES):
            theory_line(a, f, col(th, f"R_{pop}{k}"), color=color)
            cell = [r for r in rows if np.isfinite(r.get(f"PR_{pop}{k}", np.nan))]
            sweep(a, cell, "fraction_e", f"PR_{pop}{k}", color=color, jitter=.006)
            a.set_title(f"{k} {pop} cells")
            a.set_xlim(.47, .97)
            a.set_xticks([.5, .6, .7, .8, .9])
        ax[row, 0].set_ylabel(f"PR of a K-cell {pop} readout")
    for a in ax[1]:
        a.set_xlabel(XL)
    letters(ax)
    save(fig, "composition_readout_curves",
         "Dimension of a fixed-size readout against excitatory fraction. Top: K excitatory cells; bottom: K inhibitory "
         "cells. Lines give the ratio-of-moments prediction R_K (experiments/readout_theory); markers are means over "
         "48 networks of the mean PR of eight readouts. An I readout of size K exists only where the network has at "
         "least K inhibitory cells. " + NOTE)


def readout_size(rows, curves):
    th = curves["readout"]
    fs = (.6, .8, .88, .94)
    fig, ax = panels(1, 2, width=5.4, height=2.5)
    for a, pop in zip(ax, "EI"):
        for f in fs:
            color = SEQ((f - .5) / .5)
            cell = select(rows, fraction_e=f)
            ks, m, se = [], [], []
            for k in SIZES:
                v = column(cell, f"PR_{pop}{k}")
                if np.isfinite(v).sum() > 2:
                    mm, ss, _ = mean_se(v)
                    ks.append(k), m.append(mm / k), se.append(ss / k)
            a.errorbar(ks, m, yerr=se, fmt="o", color=color, ms=3.4, elinewidth=.8, capsize=1.5, label=f"$f_E={f:g}$")
            row = min(th, key=lambda r: abs(r["fraction_e"] - f))
            tk = [k for k in SIZES if row[f"R_{pop}{k}"] is not None]
            a.plot(tk, [row[f"R_{pop}{k}"] / k for k in tk], "-", color=color, lw=1)
        a.set_xscale("log")
        a.set_xticks(SIZES)
        a.set_xticklabels([str(k) for k in SIZES])
        a.minorticks_off()
        a.set_xlabel(f"readout size K ({pop} cells)")
        a.set_ylabel("PR / K")
    ax[0].legend()
    letters(ax)
    save(fig, "composition_readout_size",
         "Readout dimension per readout cell against readout size, for E (a) and I (b) readouts at four excitatory "
         "fractions. Lines are the ratio-of-moments prediction. " + NOTE)


def paired(rows, curves):
    """Seed-paired change relative to f_E = 0.80 with two-sided 95% t intervals."""
    th = curves["composition"]
    ro = curves["readout"]
    f = col(th, "fraction_e")
    ref = {r["seed"]: r for r in select(rows, fraction_e=.8)}
    spec = [("PR_E100", r"$\Delta$ PR, 100 E cells", E, col(ro, "R_E100")),
            ("PR_I100", r"$\Delta$ PR, 100 I cells", I, col(ro, "R_I100")),
            ("P_EE", r"$\Delta P_{EE}$", E, col(th, "P_EE")), ("P_II", r"$\Delta P_{II}$", I, col(th, "P_II")),
            ("dE_total", r"$\Delta\,D_E/N$", E, col(th, "dE_total")), ("d", r"$\Delta\,D/N$", JOINT, col(th, "d"))]
    k80 = np.argmin(abs(f - .8))
    fig, ax = panels(2, 3, height=4.3, sharex=True)
    for a, (name, label, color, tcurve) in zip(ax.ravel(), spec):
        a.axhline(0, color=GRAY, lw=.7)
        a.plot(f, tcurve - tcurve[k80], "-", color=color, lw=1.3)
        xs, m, half = [], [], []
        for value, cell in groups(rows, "fraction_e").items():
            diff = np.array([r[name] - ref[r["seed"]][name] for r in cell if r["seed"] in ref])
            diff = diff[np.isfinite(diff)]
            if len(diff) < 3 or np.isclose(value, .8):
                continue
            mm, se, n = mean_se(diff)
            xs.append(value), m.append(mm), half.append(t95(n) * se)
        a.errorbar(xs, m, yerr=half, fmt="o", color=color, ms=3.4, elinewidth=.8, capsize=1.6)
        a.plot([.8], [0], "o", mfc="white", mec=color, ms=3.4)
        a.set_ylabel(label)
        a.set_xlim(.47, .97)
        if len(m):
            lo = min(np.min(np.array(m) - half), np.nanmin((tcurve - tcurve[k80])[(f > .49) & (f < .95)]))
            hi = max(np.max(np.array(m) + half), np.nanmax((tcurve - tcurve[k80])[(f > .49) & (f < .95)]))
            a.set_ylim(lo - .12 * (hi - lo), hi + .12 * (hi - lo))
    for a in ax[1]:
        a.set_xlabel(XL)
    letters(ax)
    save(fig, "composition_paired_changes",
         "Seed-paired changes relative to 80% excitatory cells. Each marker is the mean over matched seeds of the "
         "difference between a network at the plotted fraction and the same-seed network at 0.80; bars are two-sided "
         "95% Student-t intervals. Lines are the predicted differences. (a,b) 100-cell E and I readout PR. (c,d) "
         "Distinct-pair energies. (e) Total E dimension per network neuron. (f) Joint dimension per neuron. " + NOTE)


def shares(rows, curves):
    th = curves["composition"]
    f = col(th, "fraction_e")
    fig, ax = panels(1, 3, height=2.35)
    for name, color, label, factor in (("E_EE", E, "E-E", 1), ("E_EI", CROSS, "2 x E-I", 2), ("E_II", I, "I-I", 1)):
        theory_line(ax[0], f, factor * col(th, name) / col(th, "energy"), color=color)
        sweep(ax[0], rows, "fraction_e", name, color=color, label=label, dots=False,
              transform=lambda v, c, k=factor: k * v / column(c, "energy"))
    ax[0].set_ylabel("share of covariance energy")
    ax[0].legend()
    sweep(ax[1], rows, "fraction_e", "lyapunov", color=GOLD, jitter=.006)
    ax[1].axhline(0, color=GRAY, lw=.7)
    ax[1].set_ylabel(r"largest Lyapunov exponent $\lambda_{\max}$")
    ax[2].axhline(1, color=GRAY, lw=.7, ls=(0, (4, 2)))
    sweep(ax[2], rows, "fraction_e", "d", color=GRAY, label="T = 4,000 / 8,000", dots=False,
          transform=lambda v, c: column(c, "d_T4000") / v)
    sweep(ax[2], rows, "fraction_e", "d", color=GOLD, label="one empty block / none", dots=False, marker="s",
          transform=lambda v, c: column(c, "d_gap1") / v)
    ax[2].set_ylabel("ratio of dimension estimates")
    ax[2].legend()
    for a in ax:
        frac_axis(a)
    letters(ax)
    save(fig, "composition_shares_and_dynamics",
         "(a) Allocation of covariance energy to E-E, E-I (counted twice) and I-I blocks with predictions. (b) "
         "Largest finite-time Lyapunov exponent, six networks per fraction. (c) Sensitivity of the dimension estimate "
         "to recording length and to an empty window between estimator windows. " + NOTE)


def ratio_vs_mean(rows, curves):
    fig, ax = panels(1, 2, width=5.2, height=2.4)
    for a, pop, color in zip(ax, "EI", (E, I)):
        a.axhline(0, color=GRAY, lw=.7)
        for k, marker in zip((25, 100), "os"):
            cell = [r for r in rows if np.isfinite(r.get(f"PR_{pop}{k}", np.nan))]
            sweep(a, cell, "fraction_e", f"PR_{pop}{k}", color=color, marker=marker, dots=False, label=f"K = {k}",
                  fill="full" if k == 100 else "none",
                  transform=lambda v, c, p=pop, kk=k: (v - column(c, f"R_{p}{kk}")) / kk)
        a.set_ylabel(f"(mean PR $-$ ratio of moments) / K, {pop}")
        a.legend()
        frac_axis(a)
    letters(ax)
    save(fig, "composition_ratio_approximation",
         "Size of the ratio-of-moments approximation. For each network, the mean of eight readout PRs minus the ratio "
         "of the subset-averaged squared trace to the subset-averaged energy, per readout cell. " + NOTE)


def main():
    style()
    rows, curves = data()
    for figure in (moments, dimensions, pairs, readout_curves, readout_size, paired, shares, ratio_vs_mean):
        figure(rows, curves)


if __name__ == "__main__":
    main()
