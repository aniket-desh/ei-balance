"""Recording-protocol checks (T = 32000, block length, gap, time step) and other coupling strengths and rates.

    python experiments/paper_campaign/plots/record_generality.py
"""
import json

import numpy as np

from common import (CAMPAIGN, CROSS, E, GOLD, GRAY, I, JOINT, LIGHT, SEQ, column, fluctuating, groups, letters,
                    load, mean_se, panels, save, select, style, sweep, t95, theory_line)

CELLS = [(dict(chi=0., fraction_e=.8), r"$\chi=0$"), (dict(chi=1., fraction_e=.8), r"$\chi=1$"),
         (dict(chi=2., fraction_e=.8), r"$\chi=2$"), (dict(chi=1., fraction_e=.6), r"60:40")]
RNOTE = "Eight networks per setting, N = 2,000; bars are one across-network standard error."


def per_network(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r["name"], {})[r[key]] = r
    return out


def duration(windows, networks):
    long = [w for w in windows if w["family"] == "record" and w["duration"] == 32000.]
    fig, ax = panels(2, 4, width=7.2, height=4.0, sharex=True)
    for j, (where, label) in enumerate(CELLS):
        cell = select(long, **where)
        th = select(networks, family="record", **where)
        if not cell:
            continue
        for block, color, marker in ((500., JOINT, "o"), (1000., CROSS, "s"), (2000., GOLD, "D")):
            sub = select(cell, block=block, gap=0.)
            x, m, se = sweep(ax[0, j], sub, "window_duration", "d", color=color, marker=marker, dots=False,
                             label=f"B = {int(block):,}", ms=3.2)
            ax[0, j].plot(x, m, "-", color=color, lw=.7)
            x, m, se = sweep(ax[1, j], sub, "window_duration", "classic_d", color=color, marker=marker, dots=False,
                             fill="none", ms=3.2,
                             transform=lambda v, c: v / column(c, "d"))
            ax[1, j].plot(x, m, "-", color=color, lw=.7)
        if th:
            ax[0, j].axhline(th[0]["th_d"], color=GRAY, lw=1, ls=(0, (4, 2)))
        ax[1, j].axhline(1, color=GRAY, lw=.7)
        ax[0, j].set_title(label)
        ax[0, j].set_xscale("log")
        ax[1, j].set_xlabel("recorded leak times T")
        ax[1, j].set_xticks([2000, 8000, 32000])
        ax[1, j].set_xticklabels(["2,000", "8,000", "32,000"])
        ax[1, j].minorticks_off()
    ax[0, 0].set_ylabel(r"$D/N$, mean-difference")
    ax[1, 0].set_ylabel("classic / mean-difference")
    ax[0, 0].legend(fontsize=6.5)
    letters(ax)
    save(fig, "record_duration_and_block",
         "Recording length and window length. Top: the four-window mean-difference estimate of D/N against the "
         "recorded duration used, for window lengths B = 500, 1,000 and 2,000 leak times; the dashed line is the "
         "prediction. Bottom: ratio of the separately centered estimate to the mean-difference estimate at the same "
         "duration and window length; the excess falls as the window lengthens, not as the recording lengthens. "
         + RNOTE)


def ingredients(windows):
    long = [w for w in windows if w["family"] == "record" and w["duration"] == 32000. and w["block"] == 500.
            and w["gap"] == 0.]
    fig, ax = panels(1, 4, width=7.2, height=2.2)
    for a, (where, label) in zip(ax, CELLS):
        cell = select(long, **where)
        base = {w["name"]: w for w in cell if w["window_duration"] == 32000.}
        for name, color, marker, lab in (("nu", GOLD, "o", r"$\nu$"), ("energy", CROSS, "s", r"$\mathcal{E}$"),
                                         ("d", JOINT, "D", r"$D/N$")):
            x, m, se = sweep(a, cell, "window_duration", name, color=color, marker=marker, dots=False, label=lab, ms=3.2,
                             transform=lambda v, c, n=name: v / np.array([base[r["name"]][n] for r in c]))
            a.plot(x, m, "-", color=color, lw=.7)
        a.axhline(1, color=GRAY, lw=.7)
        a.set_xscale("log")
        a.set_xticks([2000, 8000, 32000])
        a.set_xticklabels(["2,000", "8,000", "32,000"])
        a.minorticks_off()
        a.set_xlabel("recorded leak times T")
        a.set_title(label)
    ax[0].set_ylabel("estimate relative to T = 32,000")
    ax[0].legend()
    letters(ax)
    save(fig, "record_convergence",
         "Convergence of the estimators with recording length. Each network's rate variance, covariance energy and "
         "dimension from a prefix of its recording, divided by the same network's value at 32,000 leak times "
         "(window 500, no gap). " + RNOTE)


def gap_and_step(windows, networks):
    fig, ax = panels(1, 3, height=2.4)
    long = [w for w in windows if w["family"] == "record" and w["duration"] == 32000. and w["block"] == 500.]
    xs = np.arange(len(CELLS))
    for k, (T, color) in enumerate(((8000., GRAY), (32000., JOINT))):
        m, half = [], []
        for where, _ in CELLS:
            cell = select(long, window_duration=T, **where)
            by = {}
            for w in cell:
                by.setdefault(w["name"], {})[w["gap"]] = w["d"]
            diff = np.array([v[1.] / v[0.] - 1 for v in by.values() if 0. in v and 1. in v])
            mm, se, n = mean_se(diff)
            m.append(mm), half.append(t95(n) * se if n > 1 else np.nan)
        ax[0].errorbar(xs + (k - .5) * .18, m, yerr=half, fmt="o", color=color, ms=3.4, elinewidth=.8, capsize=1.6,
                       label=f"T = {int(T):,}")
    ax[0].axhline(0, color=GRAY, lw=.7)
    ax[0].set_ylabel(r"change of $D/N$ with one empty window")
    ax[0].legend()
    # time step: same graph and initial condition, different step; paired by network
    rec = fluctuating(select(networks, family="record"))
    ref = {}
    for w in windows:
        if (w["family"] == "record" and w["duration"] == 32000. and w["block"] == 500. and w["gap"] == 0.
                and w["window_duration"] == 8000.):
            ref[(w["seed"], w["chi"], w["fraction_e"])] = w
    for a, name, ylabel in ((ax[1], "d", r"relative change of $D/N$"), (ax[2], "nu", r"relative change of $\nu$")):
        for k, (dt, color) in enumerate(((.025, E), (.125, I))):
            m, half = [], []
            for where, _ in CELLS:
                cell = select(rec, dt=dt, **where)
                diff = np.array([r[name] / ref[(r["seed"], r["chi"], r["fraction_e"])][name] - 1 for r in cell
                                 if (r["seed"], r["chi"], r["fraction_e"]) in ref])
                mm, se, n = mean_se(diff)
                m.append(mm), half.append(t95(n) * se if n > 1 else np.nan)
                a.plot(np.full(len(diff), xs[len(m) - 1] + (k - .5) * .18), diff, ".", color=color, ms=2.4, alpha=.3, mew=0)
            a.errorbar(xs + (k - .5) * .18, m, yerr=half, fmt="o", color=color, ms=3.4, elinewidth=.8, capsize=1.6,
                       label=f"step {dt:g} vs 0.05")
        a.axhline(0, color=GRAY, lw=.7)
        a.set_ylabel(ylabel)
    ax[1].legend()
    for a in ax:
        a.set_xticks(xs)
        a.set_xticklabels([label for _, label in CELLS])
    letters(ax)
    save(fig, "record_gap_and_step",
         "Sensitivity to an empty window between estimator windows and to the integration step. (a) Relative change "
         "of D/N when the four estimator windows must be separated by one empty window. (b,c) Relative change of "
         "D/N and of the mean rate variance when the same graph and initial condition are integrated with a step of "
         "0.025 or 0.125 instead of 0.05 (8,000 leak times each); a different step produces a different chaotic "
         "trajectory, so individual networks scatter. Bars are two-sided 95% t intervals over eight networks; dots "
         "are networks.")


def generality(networks):
    curves = json.loads((CAMPAIGN / "theory_curves.json").read_text())
    gen = fluctuating(select(networks, family="generality"))
    ref = fluctuating([r for r in networks if r["family"] == "degree" and r["chi"] == 1.])
    for r in ref:
        r = r  # reference coupling and rate, same degree sampler
    fig, ax = panels(2, 4, width=7.2, height=4.2)
    for row, (key, xname, xlabel, rows) in enumerate((
            ("scale", "scale", r"coupling scale $s$", [r for r in gen if r["rate"] == .3] + ref),
            ("rate", "rate", r"target mean rate $\bar m$", [r for r in gen if r["scale"] == 1.] + ref))):
        th = curves[key]
        x = np.array([r[xname] for r in th])
        col = lambda n: np.array([r[n] for r in th], dtype=float)
        a = ax[row, 0]
        theory_line(a, x, col("d_gains"), color=LIGHT, ls=(0, (1, 1.2)), label="gains averaged")
        theory_line(a, x, col("d_spectra"), color=GRAY, ls=(0, (4, 1.5)), label="spectra averaged")
        theory_line(a, x, col("d"), label="conditional theory")
        sweep(a, rows, xname, "d", jitter=.01 * (x.max() - x.min()))
        med = [(v, np.median(column(c, "d"))) for v, c in groups(rows, xname).items()]
        a.plot(*zip(*med), "D", mfc="white", mec=JOINT, ms=3.6, mew=.7, zorder=6, label="median" if row == 0 else None)
        a.set_ylabel(r"$D/N$")
        a.set_yscale("log")
        a.set_ylim(.5 * np.nanmin(col("d")[col("d") > 2e-3]), 1.6 * np.nanmax(col("d_gains")))
        if row == 0:
            a.legend(fontsize=6.3, loc="lower right")
        a = ax[row, 1]
        for name, color, label in (("dE", E, r"$D_E/N_E$"), ("dI", I, r"$D_I/N_I$")):
            theory_line(a, x, col(name), color=color)
            sweep(a, rows, xname, name, color=color, label=label, dots=False)
        a.set_ylabel("population dimension per neuron")
        a.set_yscale("log")
        a.legend(fontsize=6.5)
        a = ax[row, 2]
        a.axhline(1, color=GRAY, lw=.7, ls=(0, (4, 2)))
        for name, color, marker, label in (("nu", GOLD, "o", r"$\nu$"), ("energy", CROSS, "s", r"$\mathcal{E}$"),
                                           ("d", JOINT, "D", r"$D/N$")):
            sweep(a, rows, xname, name, color=color, marker=marker, label=label, dots=False,
                  transform=lambda v, c, n=name: v / c[0]["th_" + n])
        a.set_ylabel("measured / predicted")
        a.set_ylim(.5, 2.)
        a.legend(fontsize=6.5)
        a = ax[row, 3]
        sweep(a, rows, xname, "lyapunov", color=GOLD)
        a.axhline(0, color=GRAY, lw=.7)
        a.set_ylabel(r"$\lambda_{\max}$")
        for a in ax[row]:
            a.set_xlabel(xlabel)
    letters(ax)
    save(fig, "generality_coupling_and_rate",
         "The conditional prediction away from the reference parameters. Top: coupling variance profile scaled by s "
         "at fixed mean connectivity and target rate 0.3. Bottom: target mean rate changed at the reference coupling. "
         "Columns: joint dimension with the two premature-averaging approximations (hollow diamonds are medians); "
         "population dimensions; "
         "measured-to-predicted ratios of rate variance, covariance energy and dimension; largest finite-time "
         "Lyapunov exponent (six networks). At s = 0.5, close to the transition to chaos, D/N scatters widely "
         "around a small prediction and its ratio lies partly off scale. N = 2,000, 80:20, Bernoulli-like dispersion 1, 16 networks per setting "
         "(24 at the reference); one s = 0.5 network that settled to a fixed point is excluded.")


def main():
    style()
    windows = load("windows.csv")
    networks = load()
    duration(windows, networks)
    ingredients(windows)
    gap_and_step(windows, networks)
    generality(networks)


if __name__ == "__main__":
    main()
