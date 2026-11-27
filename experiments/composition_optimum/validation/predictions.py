"""Parameter-only predictions for the composition-optimum validation; writes predictions.json.

    python experiments/composition_optimum/validation/predictions.py

For each target-rate pair (m_E, m_I) the conditional mean-field theory is solved once at the
reference fractions f0 = (0.8, 0.2), reference couplings G0, J0 and Bernoulli edges (chi = 1).
Along the composition path W = G diag(f) and J0 diag(f) are fixed, so the one-site law, the
per-population rate variances v_a and the tensor T_abc do not depend on f, and

    E_ab(f) = f_a f_b sum_c T_abc / f_c,      d(f) = (f v_E + g v_I)^2 / (E_EE + 2 E_EI + E_II).

No network data enter this file. The tensor uses check.tensor_for, the routine the note uses
for its unequal-rate table, and the network drive uses the same solve's mean inputs.
"""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "paper_campaign")]
import check      # noqa: E402  (the note's own routines; not modified)
import theory     # noqa: E402
from eicavity.multitype import solve_populations  # noqa: E402

RATES = {"ref": (.3, .3), "lo": (.2, .4), "hi": (.4, .2)}
FRACTIONS = (.60, .70, .75, .79, .82, .85, .88, .90, .92, .94)
FLANKS = (.70, .94)          # model-free tilt contrast: log d(0.94) - log d(0.70)
NAMES = dict(EEE=(0, 0, 0), EIE=(0, 1, 0), IIE=(1, 1, 0), EEI=(0, 0, 1), EII=(0, 1, 1), III=(1, 1, 1))


def blocks(T, f):
    """E_EE, E_EI, E_II at fraction f (E-I counted once here; the total counts it twice)."""
    fr = np.array([f, 1 - f])
    E = np.outer(fr, fr) * np.einsum("abc,c->ab", T, 1 / fr)
    return np.array([E[0, 0], E[0, 1], E[1, 1]])


def curve(T, v, f):
    f = np.asarray(f, float)
    e = np.array([blocks(T, x) for x in np.atleast_1d(f)])
    nu = v[0] * f + v[1] * (1 - f)
    return nu**2 / (e[:, 0] + 2 * e[:, 1] + e[:, 2])


def optimum(T, v, coeffs):
    roots, ds = check.interior(coeffs, *check.reduce(T, v))
    k = int(np.argmax(ds))
    return roots[k], ds[k], len(roots)


def one(rates, coeffs):
    T, v = check.tensor_for(theory.G0, rates)
    sol = solve_populations(theory.G0, fractions=theory.F0, mean_rates=tuple(rates), dt=.08, horizon=100.,
                            degree_dispersion=1.)
    al, be, p, q, vE, vI = check.reduce(T, v)
    fstar, dstar, nroots = optimum(T, v, coeffs)
    grid = np.linspace(1e-4, 1 - 1e-4, 200001)
    dg = curve(T, v, grid)
    assert abs(grid[np.argmax(dg)] - fstar) < 2e-5
    plateau = lambda lev: [float(grid[dg >= lev * dstar][0]), float(grid[dg >= lev * dstar][-1])]
    # sensitivity of f* to a +5% change in one tensor entry or one variance (all else fixed)
    sens = {}
    for k, (a, b, c) in NAMES.items():
        Tp = T.copy()
        Tp[a, b, c] *= 1.05
        Tp[b, a, c] = Tp[a, b, c]
        sens[f"T_{k}"] = optimum(Tp, v, coeffs)[0] - fstar
    for i, s in enumerate("EI"):
        vp = v.copy()
        vp[i] *= 1.05
        sens[f"v_{s}"] = optimum(T, vp, coeffs)[0] - fstar
    tot_e = float(np.sqrt(T[0, 0, 0]) / (np.sqrt(T[0, 0, 0]) + np.sqrt(T[0, 0, 1])))
    lo, hi = FLANKS
    return dict(
        rates=list(rates), mean_inputs=sol.mean_inputs.tolist(),
        tensor={k: float(T[i]) for k, i in NAMES.items()}, v_E=float(vE), v_I=float(vI),
        alpha=float(al), beta=float(be), p=float(p), q=float(q),
        f_star=float(fstar), d_star=float(dstar), interior_stationary_points=nroots,
        uniqueness_sufficient_condition=float((vE - vI) * (p * vI - q * vE)),
        plateau_1pct=plateau(.99), plateau_0p1pct=plateau(.999),
        f_star_total_E_dimension=tot_e, d_at_total_E_optimum_relative=float(curve(T, v, tot_e)[0] / dstar - 1),
        sensitivity_of_f_star_to_plus_5pct=sens,
        cells={f"{f:g}": dict(d=float(curve(T, v, f)[0]), E_EE=float(blocks(T, f)[0]), E_EI=float(blocks(T, f)[1]),
                              E_II=float(blocks(T, f)[2]), nu=float(vE * f + vI * (1 - f)),
                              energy=float(blocks(T, f) @ [1, 2, 1]))
               for f in FRACTIONS},
        tilt_log_d_94_minus_70=float(np.log(curve(T, v, hi)[0] / curve(T, v, lo)[0])),
        curve=dict(f=np.round(np.linspace(.5, .97, 95), 6).tolist(), d=curve(T, v, np.linspace(.5, .97, 95)).tolist()),
        tensor_array=T.tolist(), v=v.tolist())


def main():
    coeffs = check.symbolic()
    out = dict(fractions=list(FRACTIONS), flanks=list(FLANKS), rate_pairs={})
    for key, rates in RATES.items():
        out["rate_pairs"][key] = one(rates, coeffs)
    # the reference must agree with the campaign's theory helper and the note
    _, Tref, vref, _ = theory.composition_tensor()
    ref = out["rate_pairs"]["ref"]
    out["reference_check"] = dict(max_tensor_difference_vs_theory_py=float(np.abs(np.array(ref["tensor_array"]) - Tref).max()),
                                  max_v_difference=float(np.abs(np.array(ref["v"]) - vref).max()),
                                  note_f_star=0.8510, f_star=ref["f_star"])
    assert out["reference_check"]["max_tensor_difference_vs_theory_py"] < 1e-8
    assert abs(ref["f_star"] - 0.851048) < 1e-5
    rp = out["rate_pairs"]
    out["contrasts"] = dict(
        shift_lo=rp["lo"]["f_star"] - rp["ref"]["f_star"], shift_hi=rp["hi"]["f_star"] - rp["ref"]["f_star"],
        span_hi_minus_lo=rp["hi"]["f_star"] - rp["lo"]["f_star"],
        tilt_lo_minus_ref=rp["lo"]["tilt_log_d_94_minus_70"] - rp["ref"]["tilt_log_d_94_minus_70"],
        tilt_hi_minus_ref=rp["hi"]["tilt_log_d_94_minus_70"] - rp["ref"]["tilt_log_d_94_minus_70"])
    (HERE / "predictions.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, r in rp.items():
        print(k, r["rates"], "f*=%.4f d*=%.5f roots=%d plateau1%%=%s h=%s" % (r["f_star"], r["d_star"], r["interior_stationary_points"],
              np.round(r["plateau_1pct"], 3), np.round(r["mean_inputs"], 4)), "tilt=%.4f" % r["tilt_log_d_94_minus_70"])
        print("   sens", {k2: round(v2, 4) for k2, v2 in r["sensitivity_of_f_star_to_plus_5pct"].items()})
    print(json.dumps(out["contrasts"], indent=1), json.dumps(out["reference_check"]))


if __name__ == "__main__":
    main()
