"""Finite-size series: N = 500 to 8000 at four dispersions (80:20) and at 60:40 Bernoulli.

    python experiments/paper_campaign/plots/size.py

N = 2000 points come from the degree and composition families (their first
networks at the same settings); every other size has its own fresh networks.
"""
import numpy as np

from common import (CROSS, E, GOLD, GRAY, I, JOINT, SEQ, column, groups, letters, load, mean_se, panels, save,
                    select, style, sweep, theory_line)

CONDITIONS = [("chi0", dict(chi=0., fraction_e=.8), r"$\chi=0$"), ("chi0.5", dict(chi=.5, fraction_e=.8), r"$\chi=0.5$"),
              ("chi1", dict(chi=1., fraction_e=.8), r"$\chi=1$"), ("chi2", dict(chi=2., fraction_e=.8), r"$\chi=2$"),
              ("f0.6", dict(chi=1., fraction_e=.6), r"60:40, $\chi=1$")]
SIZES = (500, 1000, 2000, 4000, 8000, 16000)
NOTE = ("Networks per size: 48, 32, 24 (48 at 60:40), 12 and 6 for N = 500, 1,000, 2,000, 4,000 and 8,000, and 4 at "
        "N = 16,000 for dispersions 0, 1 and 2. Dots are networks; markers are across-network medians with a "
        "bootstrap standard error over fluctuating networks (medians, because at N = 500 a few weakly fluctuating "
        "networks have ill-conditioned ratios).")


def condition_rows(rows, where, keep_fixed=False):
    """Reference-coupling networks at one (chi, f_E), all sizes. Fixed-point networks are dropped unless asked."""
    out = [r for r in rows if (keep_fixed or r["fluctuating"] == 1) and r["family"] in ("size", "degree", "composition") and np.isclose(r["chi"], where["chi"])
           and np.isclose(r["fraction_e"], where["fraction_e"]) and np.isclose(r["dt"], .05)]
    if where["fraction_e"] == .8:   # keep the degree-ensemble graphs only, so one sampler per curve
        out = [r for r in out if r["family"] != "composition"]
    return out


def xaxis(ax, inverse=False):
    if inverse:
        ax.set_xlabel(r"$N^{-1/2}$")
        ax.set_xlim(0, .048)
        top = ax.secondary_xaxis("top", functions=(lambda v: v, lambda v: v))
        top.set_xticks([n**-.5 for n in SIZES])
        top.set_xticklabels([f"{n:,}" if n in (500, 2000, 16000) else "" for n in SIZES], fontsize=6.5)
        top.tick_params(length=2)
    else:
        ax.set_xscale("log")
        ax.set_xticks(list(SIZES))
        ax.set_xticklabels([{500: "500", 2000: "2k", 8000: "8k"}.get(n, "") for n in SIZES])
        ax.minorticks_off()
        ax.set_xlabel(r"network size $N$")


def dimension(rows):
    fig, ax = panels(1, 5, width=7.2, height=2.1)
    for a, (_, where, label) in zip(ax, CONDITIONS):
        cell = condition_rows(rows, where)
        if not cell:
            continue
        a.axhline(cell[0]["th_d"], color=JOINT, lw=1.3, zorder=2)
        sweep(a, cell, "n", "d", stat="median")
        xaxis(a)
        a.set_title(label)
    ax[0].set_ylabel(r"dimension per neuron $D/N$")
    letters(ax)
    save(fig, "size_dimension",
         "Dimension per neuron against network size at five settings. The horizontal line is the large-N "
         "conditional prediction. " + NOTE)


def scaling(rows):
    stats = [("d", r"$D/N$"), ("nu", r"$\nu$"), ("energy", r"$\mathcal{E}$")]
    fig, ax = panels(1, 3, height=2.6)
    colors = [SEQ(.15), SEQ(.4), SEQ(.62), SEQ(.95), GOLD]
    for a, (name, label) in zip(ax, stats):
        a.axhline(0, color=GRAY, lw=.7, ls=(0, (4, 2)))
        for (_, where, clabel), color, marker in zip(CONDITIONS, colors, "osD^v"):
            cell = condition_rows(rows, where)
            if not cell:
                continue
            x, m, se = sweep(a, cell, "n", name, color=color, marker=marker, label=clabel, dots=False,
                             transform=lambda v, c, n=name: v / c[0]["th_" + n] - 1, xmap=lambda n: n**-.5, ms=3.4,
                             stat="median")
            a.plot(x, m, "-", color=color, lw=.7, alpha=.7)
        xaxis(a, inverse=True)
        a.set_ylabel(f"relative error of {label}")
    ax[0].legend(ncol=2, loc="lower left", fontsize=6.8)
    letters(ax)
    save(fig, "size_scaling",
         "Finite-size residuals of the dimension and its two ingredients. Relative error of the across-network median "
         "against the large-N prediction, plotted against the inverse square root of network size (top axis: N). A "
         "leading finite-size correction would appear as a straight line through the origin. Bars are one "
         "across-network standard error. " + NOTE)


def ingredients(rows):
    names = [("dE", r"$D_E/N_E$"), ("dI", r"$D_I/N_I$"), ("P_EE", r"$P_{EE}$"), ("P_EI", r"$P_{EI}$"),
             ("P_II", r"$P_{II}$"), ("q_E", r"$q_E$"), ("q_I", r"$q_I$"), ("A_E", r"$A_E$"), ("A_I", r"$A_I$"),
             ("overlap", r"$\mathcal{O}_{EI}$")]
    fig, ax = panels(2, 5, width=7.2, height=4.1, sharex=True)
    colors = [SEQ(.15), SEQ(.4), SEQ(.62), SEQ(.95), GOLD]
    for a, (name, label) in zip(ax.ravel(), names):
        a.axhline(0, color=GRAY, lw=.7, ls=(0, (4, 2)))
        for (_, where, clabel), color, marker in zip(CONDITIONS, colors, "osD^v"):
            cell = condition_rows(rows, where)
            if not cell:
                continue
            x, m, se = sweep(a, cell, "n", name, color=color, marker=marker, label=clabel, dots=False,
                             transform=lambda v, c, n=name: v / c[0]["th_" + n] - 1, xmap=lambda n: n**-.5, ms=3,
                             stat="median")
            a.plot(x, m, "-", color=color, lw=.7, alpha=.7)
        a.set_title(label)
        a.set_xlim(0, .048)
    for a in ax[-1]:
        a.set_xlabel(r"$N^{-1/2}$")
    for a in ax[:, 0]:
        a.set_ylabel("relative error")
    ax[0, 0].legend(fontsize=6.3, ncol=1)
    letters(ax)
    save(fig, "size_ingredients",
         "Finite-size residuals of population dimensions, distinct-pair energies, local input and gain statistics "
         "and E-I alignment, as relative errors of the across-network median against the prediction. " + NOTE)


def fold_change(rows):
    fig, ax = panels(1, 2, width=5.2, height=2.4)
    lo, hi = condition_rows(rows, CONDITIONS[0][1]), condition_rows(rows, CONDITIONS[3][1])
    xs, fold, err = [], [], []
    for n in SIZES:
        a, b = column(select(lo, n=float(n)), "d"), column(select(hi, n=float(n)), "d")
        if len(a) and len(b):
            ma, sa, _ = mean_se(a)
            mb, sb, _ = mean_se(b)
            xs.append(n), fold.append(ma / mb)
            err.append(ma / mb * np.hypot(sa / ma, sb / mb))   # delta-method SE of a ratio of independent means
    if lo and hi:
        ax[0].axhline(lo[0]["th_d"] / hi[0]["th_d"], color=JOINT, lw=1.3)
    ax[0].errorbar(xs, fold, yerr=err, fmt="o", color=JOINT, ms=3.6, elinewidth=.8, capsize=1.6)
    xaxis(ax[0])
    ax[0].set_ylabel(r"fold change $d(\chi{=}0)\,/\,d(\chi{=}2)$")
    for (_, where, label), color, marker in zip(CONDITIONS, [SEQ(.15), SEQ(.4), SEQ(.62), SEQ(.95), GOLD], "osD^v"):
        cell = condition_rows(rows, where)
        xs, cv = [], []
        for n, g in groups(cell, "n").items():
            v = column(g, "d")
            if len(v) > 3:
                xs.append(n), cv.append(v.std(ddof=1) / v.mean())
        ax[1].plot(xs, cv, marker + "-", color=color, ms=3.2, lw=.8, label=label)
    ref = np.array([500, 16000])
    ax[1].plot(ref, .12 * (ref / 500.)**-.5, color=GRAY, lw=.8, ls=(0, (4, 2)))
    ax[1].text(2600, .058, r"$\propto N^{-1/2}$", color=GRAY, fontsize=7.5)
    ax[1].set_yscale("log")
    xaxis(ax[1])
    ax[1].set_ylabel("across-network CV of $D/N$")
    ax[1].legend(fontsize=6.5)
    letters(ax)
    save(fig, "size_fold_change_and_spread",
         "(a) Ratio of mean dimension at fixed in-degree to that at dispersion 2, against network size; the line is "
         "the predicted 10.34-fold change and bars are delta-method standard errors. (b) Coefficient of variation of "
         "D/N across networks; the dashed line has slope -1/2. " + NOTE)


def chaos(rows):
    fig, ax = panels(1, 2, width=5.2, height=2.4)
    for (_, where, label), color, marker in zip(CONDITIONS, [SEQ(.15), SEQ(.4), SEQ(.62), SEQ(.95), GOLD], "osD^v"):
        cell = [r for r in condition_rows(rows, where) if np.isfinite(r["lyapunov"])]
        if cell:
            x, m, se = sweep(ax[0], cell, "n", "lyapunov", color=color, marker=marker, label=label, dots=False)
            ax[0].plot(x, m, "-", color=color, lw=.7)
        cell = condition_rows(rows, where, keep_fixed=True)
        xs, frac = [], []
        for n, g in groups(cell, "n").items():
            xs.append(n), frac.append(np.mean(column(g, "fluctuating") == 0))
        ax[1].plot(xs, frac, marker + "-", color=color, ms=3.2, lw=.8, label=label)
    ax[0].axhline(0, color=GRAY, lw=.7)
    ax[0].set_ylabel(r"largest Lyapunov exponent $\lambda_{\max}$")
    ax[0].legend(fontsize=6.5)
    ax[1].set_ylabel("fraction settled to a fixed point")
    ax[1].set_ylim(-.02, .5)
    for a in ax:
        xaxis(a)
    letters(ax)
    save(fig, "size_chaos",
         "(a) Largest finite-time Lyapunov exponent against network size (up to six networks per point; none at "
         "N = 8,000 or above). (b) Fraction of networks that ended the recording at a stable fixed point (final drift "
         "below 1e-10); these have no stationary PR and are excluded from every other panel. " + NOTE)


def main():
    style()
    rows = load()
    for figure in (dimension, scaling, ingredients, fold_change, chaos):
        figure(rows)


if __name__ == "__main__":
    main()
