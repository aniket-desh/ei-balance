"""Estimators for the composition-optimum validation (structural fit, optimum, bootstrap, contrasts).

A cohort cell is one (rate pair, N, f). Each network contributes the five primary-window
measurements y = (E_EE, E_EI, E_II, nu_E, nu_I): block covariance energies normalized by the
total N (E-I counted once here, twice in the total) and per-population mean temporal rate
variances.

Structural fit. Prediction (i) says the cell means obey, block by block,

    E_EE(f) = f T_EEE + (f^2/g) T_EEI
    E_EI(f) = g T_EIE + f T_EII                    (g = 1 - f)
    E_II(f) = (g^2/f) T_IIE + g T_III
    nu_a(f) = v_a.

Each block is a linear regression with two unknowns over the K fractions. It is fitted to the
cell means by weighted least squares with weights n_k / mu_k^2, where mu_k is the
parameter-only prediction of that block at f_k (a constant relative error across cells, which is
what the pilot data show). The weights therefore do not depend on the data. v_a is the mean of
nu_a over all networks of the rate pair and size (equal cell sizes, so equal weights).
The optimum f* is then the root of the note's quintic in (0,1) with the largest d, computed
from the fitted (T, v) with check.interior.

Uncertainty: stratified bootstrap over networks within cells (networks are independent).
"""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "paper_campaign")]
import check  # noqa: E402

COEFFS = check.symbolic()
BLOCKS = [(0, 0), (0, 1), (1, 1)]
TOTAL = np.array([1., 2., 1.])


def basis(f):
    g = 1 - f
    return [np.array([f, f * f / g]), np.array([g, f]), np.array([g * g / f, g])]


def fit(cells, weights_mu):
    """cells: {f: (n, 5) array}; weights_mu: {f: (3,) predicted block energies}. Returns dict."""
    fs = sorted(cells)
    T = np.zeros((2, 2, 2))
    out = dict(chi2=[], residual_rel=[])
    means = np.array([cells[f].mean(0) for f in fs])
    ns = np.array([len(cells[f]) for f in fs])
    ses = np.array([cells[f].std(0, ddof=1) / np.sqrt(len(cells[f])) for f in fs])
    for k, (a, b) in enumerate(BLOCKS):
        X = np.array([basis(f)[k] for f in fs])
        y = means[:, k]
        w = np.sqrt(ns) / np.array([weights_mu[f][k] for f in fs])
        coef = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        resid = y - X @ coef
        out["chi2"].append(float(np.sum((resid / ses[:, k])**2)))
        out["residual_rel"].append((resid / (X @ coef)).tolist())
        T[a, b, 0] = T[b, a, 0] = coef[0]
        T[a, b, 1] = T[b, a, 1] = coef[1]
    allv = np.concatenate([cells[f][:, 3:5] for f in fs])
    v = allv.mean(0)
    # f-independence of v_a: slope of the cell means against f (weighted by 1/se^2) and chi2 about v
    slope, slope_se, chi2_v = [], [], []
    for i in range(2):
        y, s = means[:, 3 + i], ses[:, 3 + i]
        X = np.c_[np.ones(len(fs)), np.array(fs) - np.mean(fs)]
        W = 1 / s**2
        cov = np.linalg.inv(X.T @ (X * W[:, None]))
        beta = cov @ X.T @ (W * y)
        slope.append(float(beta[1]))
        slope_se.append(float(np.sqrt(cov[1, 1])))
        chi2_v.append(float(np.sum(((y - v[i]) / s)**2)))
    out.update(T=T, v=v, v_slope=slope, v_slope_se=slope_se, v_chi2=chi2_v, dof_blocks=len(fs) - 2, dof_v=len(fs) - 1)
    return out


def optimum(T, v):
    """(f*, d*, number of interior stationary points); f* is nan if Phi has no root in (0,1)."""
    roots, ds = check.interior(COEFFS, *check.reduce(T, v))
    if not roots:
        return np.nan, np.nan, 0
    k = int(np.argmax(ds))
    return roots[k], ds[k], len(roots)


def curve(T, v, f):
    al, be, p, q, vE, vI = check.reduce(T, v)
    f = np.asarray(f, float)
    return (vE * f + vI * (1 - f))**2 / (al / f + be / (1 - f) + p * f + q * (1 - f))


def cell_d(arr, f, kind="ratio"):
    """Cell d: 'ratio' = (mean nu)^2 / mean E (the theory's quantity); 'mean' = mean of per-network nu^2/E."""
    nu = f * arr[:, 3] + (1 - f) * arr[:, 4]
    energy = arr[:, :3] @ TOTAL
    if kind == "ratio":
        return nu.mean()**2 / energy.mean()
    return float(np.mean(nu**2 / energy))


def tilt(cells, lo, hi, kind="ratio"):
    if lo not in cells or hi not in cells:
        return np.nan
    return float(np.log(cell_d(cells[hi], hi, kind) / cell_d(cells[lo], lo, kind)))


def summarize(cells, pred, flanks):
    """Point estimates; pred = dict(mu={f: predicted blocks}, T=predicted tensor, v=predicted variances)."""
    r = fit(cells, pred["mu"])
    fstar, dstar, nroots = optimum(r["T"], r["v"])
    return dict(fit=r, f_star=fstar, d_star=dstar, nroots=nroots, tilt=tilt(cells, *flanks),
                tilt_mean_of_ratios=tilt(cells, *flanks, kind="mean"),
                f_star_T_only=optimum(r["T"], pred["v"])[0],    # measured T, predicted v
                f_star_v_only=optimum(pred["T"], r["v"])[0])    # predicted T, measured v


def bootstrap(cells, pred, flanks, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        bc = {f: c[rng.integers(0, len(c), len(c))] for f, c in cells.items()}
        s = summarize(bc, pred, flanks)
        out.append(dict(f_star=s["f_star"], nroots=s["nroots"], tilt=s["tilt"], tilt_mean_of_ratios=s["tilt_mean_of_ratios"],
                        T=s["fit"]["T"], v=s["fit"]["v"], f_star_T_only=s["f_star_T_only"], f_star_v_only=s["f_star_v_only"],
                        d_star=s["d_star"]))
    return out
