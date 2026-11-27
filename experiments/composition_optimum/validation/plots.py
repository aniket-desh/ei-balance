"""Figures for the composition-optimum validation, in the paper campaign's style.

    python experiments/composition_optimum/validation/plots.py

Reads networks.csv, cells.csv, results.json, structure.csv, optimum.csv, bootstrap_fstar.csv
and predictions.json (run summarize.py and analyze.py first). Writes PDF, PNG and a caption
file per figure into figures/.

Conventions (experiments/paper_campaign/plots/common.py): parameter-only theory as solid lines,
each network a faint dot, across-network means as markers with one standard error, thin open
axes, Computer Modern math. Color encodes population only (E blue, I red, E-I teal, whole
network near-black); the three target-rate pairs are told apart by gray level and marker shape,
network sizes by marker shape where they share an axis.
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "paper_campaign" / "plots"))
import common  # noqa: E402
from common import CROSS, E, GRAY, I, JOINT, LIGHT, letters, panels, style, theory_line  # noqa: E402

common.FIGURES = HERE / "figures"
save = common.save
PRED = json.loads((HERE / "predictions.json").read_text())
RES = json.loads((HERE / "results.json").read_text())
RATES = ("ref", "lo", "hi")
SHADE = {"ref": JOINT, "lo": "#5E6878", "hi": "#A1A9B6"}
MARK = {"ref": "o", "lo": "s", "hi": "^"}
LABEL = {k: rf"$(m_E, m_I) = ({PRED['rate_pairs'][k]['rates'][0]:g}, {PRED['rate_pairs'][k]['rates'][1]:g})$" for k in RATES}
NMARK = {2000: "o", 4000: "s", 8000: "D", 16000: "^"}
NOFF = {2000: -.006, 4000: -.002, 8000: .002, 16000: .006}
XL = r"excitatory fraction $f$"
BLOCKS = (("E_EE", r"$\mathcal{E}_{EE}$", E), ("E_EI", r"$\mathcal{E}_{EI}$", CROSS), ("E_II", r"$\mathcal{E}_{II}$", I))
TNAMES = ("EEE", "EIE", "IIE", "EEI", "EII", "III")
TCOLOR = {"EEE": E, "EEI": E, "EIE": CROSS, "EII": CROSS, "IIE": I, "III": I}
COHORT = ("Cohort: composition path at the reference couplings, Bernoulli edges, 8,000 recorded leak times, primary "
          "estimator window (four-window mean difference, window 500, no gap); 64 networks per cell at N = 2,000, 16 at "
          "4,000 and 8,000, 4 at 16,000 (reference rates only); every network independently generated.")


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def table(name):
    return [{k: num(v) for k, v in r.items()} for r in csv.DictReader(open(HERE / name))]


NETS = [r for r in table("networks.csv") if r["fluctuating"] == 1]
CELLS = table("cells.csv")
STRUCT = table("structure.csv")
OPT = table("optimum.csv")
SIZES = sorted({int(r["n"]) for r in OPT})


def cells(key, n):
    return sorted([c for c in CELLS if c["rates"] == key and int(c["n"]) == n], key=lambda c: c["fraction_e"])


def nets(key, n, f=None):
    return [r for r in NETS if r["rates"] == key and int(r["n"]) == n and (f is None or r["fraction_e"] == f)]


def cohort(key, n):
    return RES["cohorts"].get(f"{key}_N{n}")


def curve(key):
    c = PRED["rate_pairs"][key]["curve"]
    return np.array(c["f"]), np.array(c["d"])


def frac_axis(ax, lo=.57, hi=.965):
    ax.set_xlim(lo, hi)
    ax.set_xticks([.6, .7, .8, .9])
    ax.set_xlabel(XL)


def size_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([f"{n:,}" for n in SIZES])
    ax.minorticks_off()
    ax.set_xlim(SIZES[0] / 1.35, SIZES[-1] * 1.35)
    ax.set_xlabel(r"network size $N$")


def top_legend(fig, a, **kw):
    """Size legend above the panels, so it never covers data."""
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=len(l), fontsize=6.5, frameon=False, **kw)


def errbar(ax, x, y, lo, hi, color, marker, label=None, fill=True, ms=4, **kw):
    ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt=marker, color=color, mfc=color if fill else "white", mec=color, ms=ms,
                elinewidth=.8, capsize=1.6, capthick=.6, label=label, ls="none", **kw)


# ---------------------------------------------------------------- main figures
def main_dimension(n_show=8000):
    """D/N against f at the largest all-rates size, with fitted and predicted optima."""
    fig, ax = panels(1, 3, width=7.0, height=2.5, sharey=False)
    for a, key in zip(ax, RATES):
        f, d = curve(key)
        theory_line(a, f, d, color=JOINT, label="parameter-only theory")
        c = cohort(key, n_show)
        a.plot(c["fitted_curve"]["f"], c["fitted_curve"]["d"], color=JOINT, lw=.8, ls=(0, (4, 2)), label="structural fit", zorder=3)
        a.fill_between(c["fitted_curve"]["f"], c["curve_band"]["lo"], c["curve_band"]["hi"], color=LIGHT, lw=0, zorder=1)
        for cl in cells(key, n_show):
            v = [r["d"] for r in nets(key, n_show, cl["fraction_e"])]
            a.plot(np.full(len(v), cl["fraction_e"]), v, ".", ms=2.2, color=JOINT, alpha=.22, mew=0)
            a.errorbar(cl["fraction_e"], cl["d_mean"], yerr=cl["d_se"], fmt=MARK[key], color=JOINT, ms=3.6, elinewidth=.8, capsize=1.6, capthick=.6)
        ymax = max(d.max(), max(cl["d_mean"] + cl["d_se"] for cl in cells(key, n_show)))
        top = ymax * 1.06
        p = PRED["rate_pairs"][key]["f_star"]
        a.plot([p, p], [0, top], color=JOINT, lw=.6, ls=(0, (1, 1.5)))
        errbar(a, c["f_star"], top * .985, top * .985, top * .985, JOINT, "v", ms=4)
        a.plot(c["f_star_ci"], [top * .985] * 2, color=JOINT, lw=1.2)
        a.set_title(LABEL[key], fontsize=8.5)
        lo_y = min(cl["d_mean"] for cl in cells(key, n_show)) * .8
        a.set_ylim(lo_y, top * 1.02)
        frac_axis(a)
    ax[0].set_ylabel(r"joint dimension per neuron $D/N$")
    ax[0].legend(loc="lower center", fontsize=6.8)
    letters(ax)
    save(fig, "main_dimension_curves",
         f"Joint dimension along the composition path at three target-rate pairs, N = {n_show:,}. Solid line: parameter-only "
         "conditional theory; dotted vertical line: its maximum f* (0.851, 0.787, 0.895). Dots: single networks; markers: mean of "
         "per-network D/N with one standard error (16 networks per cell). Dashed line and gray band: the theory's four-coefficient "
         "form with T_abc and v_a fitted to the measured block energies and rate variances (pointwise 95% bootstrap band); the "
         "triangle and bar at the top give its maximum with a 95% bootstrap interval. " + COHORT)


def main_optimum():
    """Fitted optimum against N for each rate pair, and the shifts."""
    fig, ax = panels(1, 2, width=6.0, height=2.6)
    a = ax[0]
    for key in RATES:
        p = PRED["rate_pairs"][key]["f_star"]
        a.axhline(p, color=SHADE[key], lw=1.1, zorder=1)
        for k, n in enumerate(SIZES):
            c = cohort(key, n)
            if c is None:
                continue
            x = n * (1 + .06 * (RATES.index(key) - 1))
            errbar(a, x, c["f_star"], *c["f_star_ci"], SHADE[key], MARK[key], label=LABEL[key] if k == 0 else None)
    a.axhline(.887, color=E, lw=.8, ls=(0, (3, 2)), zorder=1)
    a.text(SIZES[-1] * 1.3, .8855, r"$D_E/N$ optimum (ref.)", color=E, fontsize=6, va="top", ha="right")
    size_axis(a)
    a.set_ylabel(r"fitted optimum $\hat f^\star$")
    a.set_ylim(.765, .955)
    h, l = a.get_legend_handles_labels()
    a.legend(h, [x.replace("(m_E, m_I) = ", "") for x in l], loc="upper left", fontsize=6, ncol=3, title=r"$(m_E, m_I)$",
             title_fontsize=6, handletextpad=.3, columnspacing=.8)
    b = ax[1]
    for side, mk in (("lo", "s"), ("hi", "^")):
        pv = PRED["contrasts"][f"shift_{side}"]
        b.axhline(pv, color=SHADE[side], lw=1.1)
        for k, n in enumerate([s for s in SIZES if f"shift_{side}_N{s}" in RES["shifts"]]):
            s = RES["shifts"][f"shift_{side}_N{n}"]
            errbar(b, n, s["estimate"], *s["ci"], SHADE[side], mk,
                   label=(r"$\hat f^\star_{lo}-\hat f^\star_{ref}$" if side == "lo" else r"$\hat f^\star_{hi}-\hat f^\star_{ref}$") if k == 0 else None)
    b.axhline(0, color=GRAY, lw=.6, ls=(0, (4, 2)))
    size_axis(b)
    b.set_ylabel("shift of the optimum")
    b.legend(loc="center right", fontsize=6.5)
    letters(ax)
    save(fig, "main_optimum_vs_size",
         "(a) Maximum of the joint dimension computed from the fitted structure (six T_abc from block energies, v_E and v_I from "
         "rate variances) at each network size and target-rate pair, with 95% stratified-bootstrap intervals; horizontal lines are "
         "the parameter-only optima 0.851, 0.787 and 0.895, and the dashed line is the optimum 0.887 of total E dimension at the "
         "reference rates. (b) Shift of the fitted optimum from the reference rate pair, with 95% intervals from independent "
         "bootstraps of the two cohorts; lines are the predicted shifts -0.064 and +0.044. " + COHORT)


def main_structure():
    """T-hat / T and v-hat / v against N, one panel per rate pair."""
    fig, ax = panels(1, 3, width=7.0, height=2.5, sharey=True)
    for a, key in zip(ax, RATES):
        a.axhline(1, color=GRAY, lw=.6, ls=(0, (4, 2)))
        a.axhspan(.9, 1.1, color="#F1F3F6", lw=0, zorder=0)
        qs = [("T_" + t, TCOLOR[t], "o" if t[2] == "E" else "s", t[2] == "E", rf"$T_{{{t}}}$") for t in TNAMES]
        qs += [("v_E", E, "D", True, r"$v_E$"), ("v_I", I, "D", False, r"$v_I$")]
        for j, (q, col, mk, fill, lab) in enumerate(qs):
            rows = sorted([r for r in STRUCT if r["rates"] == key and r["quantity"] == q], key=lambda r: r["n"])
            xs = np.array([r["n"] for r in rows]) * (1 + .035 * (j - 3.5))
            for x, r in zip(xs, rows):
                errbar(a, x, r["ratio"], r["ratio_lo"], r["ratio_hi"], col, mk, fill=fill, ms=3.2)
            a.plot(xs, [r["ratio"] for r in rows], color=col, lw=.5, alpha=.6, ls="-" if fill else ":", label=lab if key == "ref" else None)
        size_axis(a)
        a.set_title(LABEL[key], fontsize=8.5)
    ax[0].set_ylabel("fitted / predicted")
    ax[0].legend(ncol=2, fontsize=6, loc="upper left", handlelength=1.2)
    letters(ax)
    save(fig, "main_structure_vs_size",
         "Fitted coefficients of the composition structure divided by their parameter-only values, against network size. T_abc "
         "(block ab from source population c; filled circles: E sources, open squares: I sources; color gives the block: EE blue, "
         "EI teal, II red) are fitted by weighted least squares to the cell means of the block energies over ten fractions; v_E and "
         "v_I (diamonds) are the means of the per-population rate variances. Bars: 95% stratified-bootstrap intervals; shaded band: "
         "+/-10%. Small T_abc (T_EEI, T_EII, T_III) are poorly determined at the high fractions used. " + COHORT)


def paper_figure():
    """Compact four-panel summary for the manuscript."""
    fig, ax = panels(1, 4, width=7.2, height=2.15)
    a = ax[0]
    n = 8000
    for key in RATES:
        f, d = curve(key)
        scale = PRED["rate_pairs"][key]["d_star"]
        theory_line(a, f, d / scale, color=SHADE[key])
        cl = cells(key, n)
        a.errorbar([c["fraction_e"] for c in cl], [c["d_ratio"] / scale for c in cl], yerr=[c["d_se"] / scale for c in cl], fmt=MARK[key],
                   color=SHADE[key], ms=3.2, elinewidth=.7, capsize=1.3, ls="none", label=LABEL[key].replace("(m_E, m_I) = ", ""))
    a.set_ylim(.78, 1.08)
    frac_axis(a)
    a.set_ylabel(r"$(D/N)\,/\,\max_f (D/N)_{\rm theory}$")
    pair_handles = a.get_legend_handles_labels()
    b = ax[1]
    for key in RATES:
        b.axhline(PRED["rate_pairs"][key]["f_star"], color=SHADE[key], lw=1)
        for n in SIZES:
            c = cohort(key, n)
            if c:
                errbar(b, n * (1 + .07 * (RATES.index(key) - 1)), c["f_star"], *c["f_star_ci"], SHADE[key], MARK[key], ms=3.2)
    size_axis(b)
    b.set_xticklabels([f"{n // 1000}k" for n in SIZES])
    b.set_ylabel(r"$\hat f^\star$")
    c_ = ax[2]
    pr, me, lo, hi, mk = [], [], [], [], []
    for side in ("lo", "hi"):
        s = RES["shifts"][f"shift_{side}_N8000"]
        errbar(c_, s["predicted"], s["estimate"], *s["ci"], SHADE[side], MARK[side], ms=3.6)
        t = RES["shifts"][f"tilt_{side}_minus_ref_N8000"]
    lim = (-.09, .07)
    c_.plot(lim, lim, color=GRAY, lw=.6, ls=(0, (4, 2)))
    c_.axhline(0, color=LIGHT, lw=.5)
    c_.axvline(0, color=LIGHT, lw=.5)
    c_.set_xlim(*lim)
    c_.set_ylim(*lim)
    c_.legend(*pair_handles, fontsize=5.8, loc="lower right", handlelength=1, title=r"$(m_E, m_I)$", title_fontsize=5.8)
    c_.set_xlabel(r"predicted shift of $f^\star$")
    c_.set_ylabel(r"measured shift ($N = 8{,}000$)")
    d_ = ax[3]
    d_.axhline(1, color=GRAY, lw=.6, ls=(0, (4, 2)))
    for q, col, mk, fill in (("T_EEE", E, "o", True), ("T_EIE", CROSS, "o", True), ("T_IIE", I, "o", True), ("T_EEI", E, "s", False),
                             ("v_E", E, "D", True), ("v_I", I, "D", False)):
        rows = sorted([r for r in STRUCT if r["rates"] == "ref" and r["quantity"] == q], key=lambda r: r["n"])
        d_.plot([r["n"] for r in rows], [r["ratio"] for r in rows], marker=mk, color=col, mfc=col if fill else "white", ms=3, lw=.7,
                label={"T_EEE": r"$T_{EEE}$", "T_EIE": r"$T_{EIE}$", "T_IIE": r"$T_{IIE}$", "T_EEI": r"$T_{EEI}$", "v_E": r"$v_E$", "v_I": r"$v_I$"}[q])
    size_axis(d_)
    d_.set_xticklabels([f"{n // 1000}k" for n in SIZES])
    d_.set_ylabel("fitted / predicted (ref.)")
    lo_, hi_ = d_.get_ylim()
    d_.set_ylim(lo_, hi_ + .45 * (hi_ - lo_))
    d_.legend(fontsize=5.6, ncol=3, loc="upper center", handlelength=1, columnspacing=.6)
    letters(ax)
    save(fig, "paper_composition_optimum",
         "Testing the joint-dimension optimum along the composition path. (a) D/N (ratio of cell means, one standard error) "
         "against the excitatory fraction at N = 8,000 for target rates (0.3, 0.3), (0.2, 0.4) and (0.4, 0.2), each divided by "
         "the maximum of its parameter-only curve (lines). (b) Optimum computed from the fitted structure (T_abc from block "
         "energies, v_a from rate variances) against N, 95% bootstrap intervals; lines: predicted 0.851, 0.787, 0.895. (c) Shift "
         "of the fitted optimum from the reference rates, measured at N = 8,000 against predicted; dashed: identity. (d) Fitted "
         "coefficients over predicted at the reference rates against N (point estimates; intervals in main_structure_vs_size); T_EEI is the pole residue beta and is poorly determined. " + COHORT)


# ---------------------------------------------------------------- diagnostics
def all_curves():
    fig, ax = panels(len(SIZES), 3, width=7.0, height=1.9 * len(SIZES) + .3)
    ax = np.atleast_2d(ax)
    for i, n in enumerate(SIZES):
        for j, key in enumerate(RATES):
            a = ax[i, j]
            c = cohort(key, n)
            if c is None:
                a.axis("off")
                continue
            f, d = curve(key)
            theory_line(a, f, d)
            a.plot(c["fitted_curve"]["f"], c["fitted_curve"]["d"], color=JOINT, lw=.8, ls=(0, (4, 2)))
            a.fill_between(c["fitted_curve"]["f"], c["curve_band"]["lo"], c["curve_band"]["hi"], color=LIGHT, lw=0, zorder=1)
            for cl in cells(key, n):
                v = [r["d"] for r in nets(key, n, cl["fraction_e"])]
                a.plot(np.full(len(v), cl["fraction_e"]), v, ".", ms=1.8, color=JOINT, alpha=.15, mew=0, rasterized=True)
                a.errorbar(cl["fraction_e"], cl["d_mean"], yerr=cl["d_se"], fmt="o", color=JOINT, ms=2.8, elinewidth=.7, capsize=1.2)
                a.plot(cl["fraction_e"], cl["d_ratio"], "o", mfc="white", mec=GRAY, ms=2.8, zorder=5)
            a.set_ylim(min(cl["d_mean"] for cl in cells(key, n)) * .75, max(d.max(), max(cl["d_mean"] for cl in cells(key, n))) * 1.12)
            a.text(.03, .95, f"N = {n:,}", transform=a.transAxes, fontsize=6.8, va="top")
            frac_axis(a)
            if i == 0:
                a.set_title(LABEL[key], fontsize=8)
            if j == 0:
                a.set_ylabel(r"$D/N$")
    save(fig, "diag_dimension_all_cohorts",
         "D/N against f for every cohort. Solid: parameter-only theory; dashed with gray band: structural fit and pointwise 95% "
         "bootstrap band; filled markers: mean of per-network D/N with one standard error; open markers: ratio of cell means "
         "(mean nu)^2 / mean E, the quantity the structural fit uses; faint dots: networks. " + COHORT)


def block_ratios():
    fig, ax = panels(3, 3, width=7.0, height=6.2, sharex=True)
    for i, (blk, lab, col) in enumerate(BLOCKS):
        for j, key in enumerate(RATES):
            a = ax[i, j]
            a.axhline(1, color=GRAY, lw=.6, ls=(0, (4, 2)))
            for n in SIZES:
                cl = cells(key, n)
                if not cl:
                    continue
                x = np.array([c["fraction_e"] for c in cl]) + NOFF[n]
                y = np.array([c[blk] / c["th_" + blk] for c in cl])
                e = np.array([c[blk + "_se"] / c["th_" + blk] for c in cl])
                a.errorbar(x, y, yerr=e, fmt=NMARK[n], color=col, mfc=col if n >= 8000 else "white", ms=3, elinewidth=.7, capsize=1.2,
                           ls="none", label=f"N = {n:,}")
            frac_axis(a)
            if i == 0:
                a.set_title(LABEL[key], fontsize=8)
            if j == 0:
                a.set_ylabel(f"measured / predicted {lab}")
            if i < 2:
                a.set_xlabel("")
    top_legend(fig, ax[0, 0])
    save(fig, "diag_block_energy_ratios",
         "Each block covariance energy (E-E blue, E-I teal, I-I red; normalized by total N, same-neuron terms included) divided "
         "by its parameter-only value f_a f_b sum_c T_abc / f_c, per cell, with one standard error. Marker shape gives N (open: "
         "2,000 and 4,000; filled: 8,000 and 16,000). " + COHORT)


def block_absolute():
    fig, ax = panels(1, 3, width=7.0, height=2.5)
    n = 8000
    for a, key in zip(ax, RATES):
        T = np.array(PRED["rate_pairs"][key]["tensor_array"])
        c = cohort(key, n)
        Tf = np.zeros((2, 2, 2))
        for name, (p, q, s) in dict(EEE=(0, 0, 0), EIE=(0, 1, 0), IIE=(1, 1, 0), EEI=(0, 0, 1), EII=(0, 1, 1), III=(1, 1, 1)).items():
            Tf[p, q, s] = Tf[q, p, s] = c["T"][name]
        ff = np.linspace(.57, .96, 100)
        for k, (blk, lab, col) in enumerate(BLOCKS):
            p_, q_ = [(0, 0), (0, 1), (1, 1)][k]
            fr = np.array([ff, 1 - ff])
            th = fr[p_] * fr[q_] * (T[p_, q_, 0] / ff + T[p_, q_, 1] / (1 - ff))
            fi = fr[p_] * fr[q_] * (Tf[p_, q_, 0] / ff + Tf[p_, q_, 1] / (1 - ff))
            theory_line(a, ff, th, color=col, label=lab)
            a.plot(ff, fi, color=col, lw=.8, ls=(0, (4, 2)))
            cl = cells(key, n)
            a.errorbar([x["fraction_e"] for x in cl], [x[blk] for x in cl], yerr=[x[blk + "_se"] for x in cl], fmt="o", color=col, ms=3,
                       elinewidth=.7, capsize=1.2, ls="none")
        a.set_yscale("log")
        a.set_title(LABEL[key], fontsize=8)
        frac_axis(a)
    ax[0].set_ylabel("block energy (per network neuron)")
    ax[0].legend(fontsize=6.5, loc="lower left")
    save(fig, "diag_block_energy_fit_N8000",
         "Block energies at N = 8,000 against f (log scale): parameter-only theory (solid), the two-term linear forms with fitted "
         "T_abc (dashed) and cell means with one standard error (16 networks per cell). " + COHORT)


def block_residuals():
    fig, ax = panels(3, 3, width=7.0, height=6.0, sharex=True)
    for j, key in enumerate(RATES):
        for n in SIZES:
            c = cohort(key, n)
            if c is None:
                continue
            fs = sorted(float(k) for k in c["per_cell"])
            for i, (blk, lab, col) in enumerate(BLOCKS):
                a = ax[i, j]
                a.plot(np.array(fs) + NOFF[n], c["block_residual_rel"][i], NMARK[n], color=col, mfc=col if n >= 8000 else "white", ms=3, ls="-",
                       lw=.4, label=f"N = {n:,}, $\\chi^2$ = {c['block_chi2'][i]:.0f}/{c['dof_blocks']}")
        for i, (blk, lab, col) in enumerate(BLOCKS):
            a = ax[i, j]
            a.axhline(0, color=GRAY, lw=.6, ls=(0, (4, 2)))
            a.legend(fontsize=5.2, loc="upper left")
            frac_axis(a)
            if i < 2:
                a.set_xlabel("")
            if i == 0:
                a.set_title(LABEL[key], fontsize=8)
            if j == 0:
                a.set_ylabel(f"relative residual of {lab}")
    save(fig, "diag_block_fit_residuals",
         "Residuals of the two-term linear fit of each block energy, (cell mean - fit) / fit, per cohort, with the chi-square of "
         "the cell means about the fit (empirical cell standard errors, 8 degrees of freedom). Systematic curvature here is "
         "f-dependence that the f-independent tensor cannot absorb. " + COHORT)


def variances():
    fig, ax = panels(1, 3, width=7.0, height=2.5)
    for a, key in zip(ax, RATES):
        for s, col, th in (("nu_E", E, PRED["rate_pairs"][key]["v_E"]), ("nu_I", I, PRED["rate_pairs"][key]["v_I"])):
            a.axhline(th, color=col, lw=1.1)
            for n in SIZES:
                cl = cells(key, n)
                if not cl:
                    continue
                a.errorbar(np.array([c["fraction_e"] for c in cl]) + NOFF[n], [c[s] for c in cl], yerr=[c[s + "_se"] for c in cl], fmt=NMARK[n],
                           color=col, mfc=col if n >= 8000 else "white", ms=3, elinewidth=.7, capsize=1.2, ls="none",
                           label=f"N = {n:,}" if s == "nu_E" else None)
        a.set_title(LABEL[key], fontsize=8)
        frac_axis(a)
    ax[0].set_ylabel(r"mean temporal rate variance $\nu_a$")
    top_legend(fig, ax[0])
    save(fig, "diag_rate_variances_vs_f",
         "Per-population mean temporal rate variance (E blue, I red) against f, cell means with one standard error; horizontal "
         "lines are the parameter-only v_E and v_I, which the theory says do not depend on f along this path. " + COHORT)


def rates():
    fig, ax = panels(1, 3, width=7.0, height=2.5)
    for a, key in zip(ax, RATES):
        tE, tI = PRED["rate_pairs"][key]["rates"]
        a.axhline(tE, color=E, lw=1.1)
        a.axhline(tI, color=I, lw=1.1, ls="-" if tI != tE else (0, (2, 2)))
        for n in SIZES:
            cl = cells(key, n)
            if not cl:
                continue
            for s, col in (("rate_E", E), ("rate_I", I)):
                a.errorbar(np.array([c["fraction_e"] for c in cl]) + NOFF[n], [c[s] for c in cl], yerr=[c[s + "_sd"] for c in cl], fmt=NMARK[n],
                           color=col, mfc=col if n >= 8000 else "white", ms=3, elinewidth=.6, capsize=1.0, ls="none",
                           label=f"N = {n:,}" if s == "rate_E" else None)
        a.set_title(LABEL[key], fontsize=8)
        frac_axis(a)
    ax[0].set_ylabel("population mean rate")
    top_legend(fig, ax[0])
    save(fig, "diag_rates_vs_targets",
         "Measured population mean rates (E blue, I red) against their targets (lines), per cell; bars are the across-network "
         "standard deviation, not the standard error. The unequal target rates enter only through the balance drive "
         "-sqrt(N) J0 (f * m) + h. " + COHORT)


def bootstrap_hist():
    rows = table("bootstrap_fstar.csv")
    fig, ax = panels(1, 3, width=7.0, height=2.3, sharey=False)
    for a, key in zip(ax, RATES):
        for n in SIZES:
            v = np.array([r["f_star"] for r in rows if r["rates"] == key and int(r["n"]) == n])
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            a.hist(v, bins=np.linspace(.70, .98, 113), histtype="step", color=JOINT, lw=.6 + .3 * SIZES.index(n),
                   alpha=.4 + .2 * SIZES.index(n), label=f"N = {n:,}", density=True)
        a.axvline(PRED["rate_pairs"][key]["f_star"], color=JOINT, lw=1.1)
        a.set_title(LABEL[key], fontsize=8)
        a.set_xlabel(r"bootstrap $\hat f^\star$")
        a.set_xlim(.70, .98)
        a.set_yticks([])
    ax[0].legend(fontsize=6, loc="upper left")
    save(fig, "diag_bootstrap_fstar",
         f"Bootstrap distributions of the fitted optimum ({RES['draws']} stratified resamples of networks within cells) per "
         "cohort; thicker lines are larger N. Vertical line: parameter-only f*. " + COHORT)


def tilt():
    fig, ax = panels(1, 2, width=6.0, height=2.5)
    a = ax[0]
    for key in RATES:
        a.axhline(PRED["rate_pairs"][key]["tilt_log_d_94_minus_70"], color=SHADE[key], lw=1.1)
        for k, n in enumerate(SIZES):
            c = cohort(key, n)
            if c and np.isfinite(c["tilt"]):
                errbar(a, n * (1 + .06 * (RATES.index(key) - 1)), c["tilt"], *c["tilt_ci"], SHADE[key], MARK[key], label=LABEL[key] if k == 0 else None)
    size_axis(a)
    a.set_ylabel(r"tilt $\tau = \log\,[d(0.94)/d(0.70)]$")
    top_legend(fig, a)
    b = ax[1]
    for side in ("lo", "hi"):
        b.axhline(PRED["contrasts"][f"tilt_{side}_minus_ref"], color=SHADE[side], lw=1.1)
        for n in SIZES:
            k = f"tilt_{side}_minus_ref_N{n}"
            if k in RES["shifts"]:
                s = RES["shifts"][k]
                errbar(b, n, s["estimate"], *s["ci"], SHADE[side], MARK[side])
    b.axhline(0, color=GRAY, lw=.6, ls=(0, (4, 2)))
    size_axis(b)
    b.set_ylabel(r"$\tau - \tau_{\rm ref}$")
    letters(ax)
    save(fig, "diag_tilt_model_free",
         "Model-free tilt of the dimension curve, tau = log d(0.94) - log d(0.70), with d the ratio of cell means, at each N and "
         "rate pair (a), and its difference from the reference rate pair (b); 95% bootstrap intervals; lines are parameter-only "
         "values. " + COHORT)


def windows_fig():
    fig, ax = panels(1, 1, width=4.2, height=2.6)
    a = ax[0]
    xs, labels = [], []
    k = 0
    for key in RATES:
        for n in SIZES:
            c = cohort(key, n)
            if c is None:
                continue
            errbar(a, k - .2, c["f_star"], *c["f_star_ci"], JOINT, MARK[key], ms=3.2, label="T = 8,000, gap 0 (primary)" if k == 0 else None)
            for dx, tag, col, lab in ((0, "T4000", GRAY, "T = 4,000, gap 0"), (.2, "gap1", LIGHT, "T = 8,000, gap 1")):
                w = RES["windows"].get(f"{key}_N{n}_{tag}")
                if w:
                    errbar(a, k + dx, w["f_star"], *w["f_star_ci"], col if col != LIGHT else "#9AA3B2", MARK[key], ms=3.2, fill=False, label=lab if k == 0 else None)
            a.plot([k - .4, k + .4], [PRED["rate_pairs"][key]["f_star"]] * 2, color=JOINT, lw=1)
            labels.append(f"{key}\n{n // 1000}k")
            xs.append(k)
            k += 1
    a.set_xticks(xs)
    a.set_xticklabels(labels, fontsize=6)
    a.set_ylabel(r"fitted optimum $\hat f^\star$")
    a.legend(fontsize=6, loc="upper left")
    save(fig, "diag_estimator_windows",
         "Fitted optimum per cohort from three estimator windows: the primary (8,000 leak times, no gap), the first 4,000 leak "
         "times, and 8,000 with one empty block between windows; 95% bootstrap intervals; short lines: parameter-only f*. " + COHORT)


def decomposition():
    fig, ax = panels(1, 1, width=4.2, height=2.6)
    a = ax[0]
    k, xs, labels = 0, [], []
    for key in RATES:
        for n in SIZES:
            c = cohort(key, n)
            if c is None:
                continue
            errbar(a, k - .2, c["f_star"], *c["f_star_ci"], JOINT, MARK[key], ms=3.2, label="fitted T and v" if k == 0 else None)
            errbar(a, k, c["f_star_T_only"], *c["f_star_T_only_ci"], CROSS, MARK[key], ms=3.2, fill=False, label="fitted T, predicted v" if k == 0 else None)
            errbar(a, k + .2, c["f_star_v_only"], *c["f_star_v_only_ci"], GRAY, MARK[key], ms=3.2, fill=False, label="predicted T, fitted v" if k == 0 else None)
            a.plot([k - .4, k + .4], [PRED["rate_pairs"][key]["f_star"]] * 2, color=JOINT, lw=1)
            labels.append(f"{key}\n{n // 1000}k")
            xs.append(k)
            k += 1
    a.set_xticks(xs)
    a.set_xticklabels(labels, fontsize=6)
    a.set_ylabel(r"optimum $f^\star$")
    a.legend(fontsize=6, loc="upper left")
    save(fig, "diag_decomposition",
         "Which ingredient moves the optimum at finite N: f* from fitted T and v (filled), from fitted T with parameter-only v "
         "(teal open) and from parameter-only T with fitted v (gray open), with 95% bootstrap intervals; short lines: "
         "parameter-only f*. " + COHORT)


def d_ratio():
    fig, ax = panels(1, 3, width=7.0, height=2.4, sharey=True)
    for a, key in zip(ax, RATES):
        a.axhline(1, color=GRAY, lw=.6, ls=(0, (4, 2)))
        for n in SIZES:
            cl = cells(key, n)
            if not cl:
                continue
            a.errorbar(np.array([c["fraction_e"] for c in cl]) + NOFF[n], [c["d_mean"] / c["th_d"] for c in cl],
                       yerr=[c["d_se"] / c["th_d"] for c in cl], fmt=NMARK[n], color=JOINT, mfc=JOINT if n >= 8000 else "white", ms=3,
                       elinewidth=.7, capsize=1.2, ls="none", label=f"N = {n:,}")
        a.set_title(LABEL[key], fontsize=8)
        frac_axis(a)
    ax[0].set_ylabel("measured / predicted $D/N$")
    top_legend(fig, ax[0])
    save(fig, "diag_dimension_ratio_cells",
         "Mean of per-network D/N divided by the parameter-only value, per cell and size, with one standard error. An "
         "f-dependent offset that shrinks with N is what limits a direct argmax at small N. " + COHORT)


def spread():
    fig, ax = panels(1, 3, width=7.0, height=2.4, sharey=True)
    for a, key in zip(ax, RATES):
        for q, col, lab in (("E_EE", E, r"$\mathcal{E}_{EE}$"), ("E_II", I, r"$\mathcal{E}_{II}$"), ("nu_E", E, r"$\nu_E$"), ("nu_I", I, r"$\nu_I$")):
            ys = []
            for n in SIZES:
                cl = cells(key, n)
                ys.append(np.median([c[q + "_rel_sd"] for c in cl]) if cl else np.nan)
            a.plot(SIZES, ys, "o-" if q.startswith("E") else "D:", color=col, mfc=col if q.startswith("E") else "white", ms=3, lw=.8, label=lab)
        ns = np.array(SIZES, float)
        a.plot(ns, .14 * (ns / 2000)**-.5, color=GRAY, lw=.6, ls=(0, (4, 2)), label=r"$\propto N^{-1/2}$")
        size_axis(a)
        a.set_yscale("log")
        a.set_title(LABEL[key], fontsize=8)
    ax[0].set_ylabel("relative SD across networks")
    ax[0].legend(fontsize=6, loc="lower left")
    save(fig, "diag_spread_vs_size",
         "Across-network relative standard deviation of block energies and rate variances against N (median over the ten fractions); the "
         "dashed line is N^(-1/2) through 0.14 at N = 2,000. This sets how many networks each size needs. " + COHORT)


def campaign():
    rc = RES["campaign_replication"]
    new = cohort("ref", 2000)
    fig, ax = panels(1, 1, width=4.2, height=2.6)
    a = ax[0]
    a.axhline(1, color=GRAY, lw=.6, ls=(0, (4, 2)))
    names = list(TNAMES) + ["v_E", "v_I"]
    for k, nm in enumerate(names):
        if nm.startswith("v"):
            i = 0 if nm == "v_E" else 1
            r1, c1 = rc["v_ratio"][i], rc["v_ratio_ci"][i]
            r2, c2 = new["v_ratio"][i], new["v_ratio_ci"][i]
            col = E if nm == "v_E" else I
        else:
            r1, c1 = rc["T_ratio"][nm], rc["T_ratio_ci"][nm]
            r2, c2 = new["T_ratio"][nm], new["T_ratio_ci"][nm]
            col = TCOLOR[nm]
        errbar(a, k - .15, r1, *c1, col, "o", fill=False, ms=3.2, label="campaign, 13 x 48" if k == 0 else None)
        errbar(a, k + .15, r2, *c2, col, "o", ms=3.2, label="this cohort, 10 x 64" if k == 0 else None)
    a.set_xticks(range(len(names)))
    a.set_xticklabels([rf"$T_{{{n}}}$" if not n.startswith("v") else rf"$v_{n[-1]}$" for n in names], fontsize=7)
    a.set_ylabel("fitted / predicted, N = 2,000, ref.")
    a.set_ylim(.5, 1.6)
    a.legend(fontsize=6, loc="upper left")
    a.set_title(rf"$\hat f^\star$: campaign {rc['f_star']:.3f} [{rc['f_star_ci'][0]:.3f}, {rc['f_star_ci'][1]:.3f}]; "
                rf"this {new['f_star']:.3f} [{new['f_star_ci'][0]:.3f}, {new['f_star_ci'][1]:.3f}]", fontsize=7)
    save(fig, "diag_campaign_replication",
         "The structural fit at N = 2,000 and reference rates on two independent cohorts: the paper campaign's 13 fractions x 48 "
         "networks (open, 0.50 to 0.94) and this experiment's 10 fractions x 64 (filled, 0.60 to 0.94); 95% bootstrap intervals. "
         "The campaign fit was seen before this experiment was designed.")


def ratio_bias():
    fig, ax = panels(1, 3, width=7.0, height=2.4, sharey=True)
    for a, key in zip(ax, RATES):
        a.axhline(0, color=GRAY, lw=.6, ls=(0, (4, 2)))
        for n in SIZES:
            cl = cells(key, n)
            if not cl:
                continue
            a.plot(np.array([c["fraction_e"] for c in cl]) + NOFF[n], [100 * (c["d_mean"] / c["d_ratio"] - 1) for c in cl], NMARK[n], color=JOINT,
                   mfc=JOINT if n >= 8000 else "white", ms=3, ls="-", lw=.4, label=f"N = {n:,}")
        a.set_title(LABEL[key], fontsize=8)
        frac_axis(a)
    ax[0].set_ylabel("mean of ratios / ratio of means $-$ 1 (%)")
    top_legend(fig, ax[0])
    save(fig, "diag_ratio_bias",
         "Difference between the mean of per-network D/N and the ratio of cell means (mean nu)^2 / mean E, in percent, per cell. "
         "The structural fit uses ratios of means; this finite-sample difference vanishes as networks self-average. " + COHORT)


def main():
    import shutil
    common.FIGURES.mkdir(exist_ok=True)
    for name in ("optimum.csv", "structure.csv", "cells.csv", "results.json"):   # the tables behind the figures
        shutil.copy(HERE / name, common.FIGURES / name)
    style()
    for fn in (main_dimension, main_optimum, main_structure, paper_figure, all_curves, block_ratios, block_absolute, block_residuals,
               variances, rates, bootstrap_hist, tilt, windows_fig, decomposition, d_ratio, spread, campaign, ratio_bias):
        try:
            fn()
        except Exception as exc:     # a missing cohort should not stop the other figures
            print("FAILED", fn.__name__, repr(exc))


if __name__ == "__main__":
    main()
