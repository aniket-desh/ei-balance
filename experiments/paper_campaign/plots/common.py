"""Shared loading, statistics and drawing conventions for the campaign figures.

Visual conventions follow David G. Clark's theory figures (Clark, Abbott and
Litwin-Kumar, PRL 2023, Fig. 2; Clark et al., PRX 2025, Fig. 4): the
parameter-only prediction is a solid line, each simulated network is a faint
dot, and the across-network mean is a marker with a one-standard-error bar.
Axes are thin and open, math is Computer Modern, and color only encodes
population (E blue, I red, E-I teal, whole network near-black).

figures4papers precedents consulted (github.com/ChenLiu-1996/figures4papers,
scientific-figure-making/references/{demos,design-theory,common-patterns}.md):
figure_ophthal_review (trend plots) and figure_RNAGenScape (heat maps). Adapted
from them: top/right spines off, frameless legends, Arial/Helvetica stack, at most
about four curves per axis, no grid on trend plots, y-limits tightened to the data,
editable SVG text. Departures, following the Clark figures and the manuscript's
print size: thinner lines (1.3 pt rather than 2-3), 8.5 pt text for a 6.6 in wide
figure, markers with standard-error bars instead of filled bands, and a
population-coded palette instead of the method-coded blue/green/red one. The
repository has no precedent for paired observed-versus-shuffle dot plots; those
follow Clark et al., PRX 2025, Fig. 4 (dots for realizations, bars for summaries).
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
FIGURES = CAMPAIGN / "figures"

E = "#2F6DB5"
I = "#C8414F"
CROSS = "#1E8A8A"
JOINT = "#1F2430"
GRAY = "#8A93A3"
LIGHT = "#D5DAE3"
GOLD = "#C98A1B"
POP = {"E": E, "I": I, "EI": CROSS, "EE": E, "II": I, "joint": JOINT}
SEQ = LinearSegmentedColormap.from_list("campaign_seq", ["#C9D7EA", "#6C9BD1", E, "#17325C", JOINT])
SIGNED = LinearSegmentedColormap.from_list("campaign_signed", [E, "#F4F5F7", I])
NUMERIC = ("n", "seed", "seed_index", "chi", "fraction_e", "scale", "rate", "dt", "duration")


def style():
    rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "cm", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
        "axes.linewidth": .7, "axes.edgecolor": JOINT, "axes.labelcolor": JOINT, "text.color": JOINT,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlepad": 5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "xtick.color": JOINT, "ytick.color": JOINT,
        "xtick.major.width": .7, "ytick.major.width": .7, "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.6, "ytick.minor.size": 1.6, "xtick.minor.width": .5, "ytick.minor.width": .5,
        "lines.linewidth": 1.3, "legend.frameon": False, "legend.fontsize": 7.5, "legend.handlelength": 1.6,
        "legend.borderaxespad": .2, "legend.labelspacing": .3,
        "figure.facecolor": "white", "savefig.facecolor": "white", "figure.dpi": 150,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def number(value):
    if value in ("", "None", None):
        return np.nan
    try:
        return float(value)
    except ValueError:
        return value


def load(name="networks.csv"):
    """Rows as dicts with numeric fields converted; missing values are nan."""
    with open(CAMPAIGN / name) as handle:
        return [{k: number(v) if k not in ("name", "family", "host", "state") else v for k, v in row.items()}
                for row in csv.DictReader(handle)]


def fluctuating(rows):
    """Drop networks that ended the recording at a fixed point (no stationary PR); they are counted separately."""
    return [r for r in rows if r.get("fluctuating", 1) == 1]


def select(rows, **where):
    out = []
    for r in rows:
        if all((r[k] in v) if isinstance(v, (list, tuple, set)) else np.isclose(r[k], v) if isinstance(v, float)
               else r[k] == v for k, v in where.items()):
            out.append(r)
    return out


def groups(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return dict(sorted(out.items()))


def column(rows, name):
    return np.array([r[name] for r in rows], dtype=float)


def mean_se(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, 0
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan), len(v)


def t95(n):
    from scipy.stats import t
    return t.ppf(.975, n - 1) if n > 1 else np.nan


def median_se(values, draws=2000):
    """Median across networks and its bootstrap standard error (networks resampled with replacement)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.median(v) if len(v) else np.nan), np.nan, len(v)
    rng = np.random.default_rng(20261124)
    boot = np.median(v[rng.integers(0, len(v), size=(draws, len(v)))], axis=1)
    return np.median(v), boot.std(ddof=1), len(v)


def sweep(ax, rows, x, y, color=JOINT, marker="o", label=None, dots=True, transform=None, jitter=0.,
          xmap=None, ms=3.6, zorder=4, fill="full", stat="mean"):
    """Faint dot per network; marker = across-network mean with one SE (or median with bootstrap SE)."""
    xs, ms_, ses = [], [], []
    for value, cell in groups(rows, x).items():
        v = column(cell, y)
        if transform is not None:
            v = transform(v, cell)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        xv = xmap(value) if xmap else value
        if dots:
            dx = np.linspace(-jitter, jitter, len(v)) if jitter and len(v) > 1 else 0.
            ax.plot(np.full(len(v), xv) + dx, v, ".", ms=2.2, color=color, alpha=.22, mew=0, zorder=zorder - 1,
                    rasterized=len(v) > 40)
        m, se, _ = mean_se(v) if stat == "mean" else median_se(v)
        xs.append(xv), ms_.append(m), ses.append(se)
    ax.errorbar(xs, ms_, yerr=ses, fmt=marker, color=color, ms=ms, mew=.6, mec=color,
                mfc=color if fill == "full" else "white", elinewidth=.8, capsize=1.6, capthick=.6,
                zorder=zorder, label=label, ls="none")
    return np.array(xs), np.array(ms_), np.array(ses)


def theory_line(ax, x, y, color=JOINT, label=None, **kw):
    return ax.plot(x, y, "-", color=color, lw=kw.pop("lw", 1.3), zorder=kw.pop("zorder", 2), label=label, **kw)


def panels(nrows, ncols, width=6.6, height=None, **kw):
    height = height or 2.05 * nrows + .25
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), constrained_layout=True, **kw)
    return fig, np.atleast_1d(axes)


def letter(ax, text, x=-.02, y=1.04):
    ax.text(x, y, f"({text})", transform=ax.transAxes, ha="right", va="bottom", weight="bold", fontsize=9)


def letters(axes):
    for ax, t in zip(np.ravel(axes), "abcdefghijklmnopqrstuvwxyz"):
        letter(ax, t)


def sqrt_axis(ax, ticks=(0, .1, .5, 1, 2, 3)):
    """Square-root horizontal scale resolves the closely spaced small dispersions."""
    ax.set_xscale("function", functions=(lambda v: np.sqrt(np.maximum(v, 0)), lambda v: v**2))
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlim(-.004, max(ticks) * 1.04)


def save(fig, name, caption=None):
    """Vector PDF plus a PNG preview; optional standalone caption beside them."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=220)
    plt.close(fig)
    if caption:
        (FIGURES / f"{name}.txt").write_text(" ".join(caption.split()) + "\n")
    print("wrote", name)
