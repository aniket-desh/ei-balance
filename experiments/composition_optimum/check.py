"""Symbolic and numerical checks for joint_dimension_optimum.tex; writes values.json.

    python experiments/composition_optimum/check.py

1. Symbolic: the reduction E(f) = alpha/f + beta/(1-f) + p f + q (1-f), the stationarity
   polynomial Phi(f) and its six coefficients, the special cases, the difference-mode form,
   the projective form of 1/d used for the uniqueness condition, and the inverse formula.
2. Numerical at the reference parameters: the complex tensor and its imaginary part, the four
   coefficients, all roots of the quintic, the maximum by direct search, the Newton
   approximations, the plateau, the energy split by source and block, convexity of 1/d.
   (theory.solve caches its solutions in experiments/paper_campaign/theory_cache, which is ignored by git.)
3. Counterexamples to uniqueness, and the Newton formula away from the reference (unequal rates).
4. Generic solvability: Galois group of the quintic for random integer coefficients.
5. Comparison with the 624 composition networks of experiments/paper_campaign and with the
   624 seed triplets of experiments/readout_confirmation; the power calculation.
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "paper_campaign"))
import theory


def symbolic():
    f, vE, vI, al, be, p, q = sp.symbols("f v_E v_I alpha beta p q")
    TEEE, TEIE, TIIE, TEEI, TEII, TIII = sp.symbols("T_EEE T_EIE T_IIE T_EEI T_EII T_III")
    g = 1 - f
    E_full = (f**2 * TEEE + 2 * f * g * TEIE + g**2 * TIIE) / f + (f**2 * TEEI + 2 * f * g * TEII + g**2 * TIII) / g
    sub = {al: TIIE, be: TEEI, p: TEEE - TIIE - 2 * TEEI + 2 * TEII, q: 2 * TEIE - 2 * TIIE - TEEI + TIII}
    E = al / f + be / g + p * f + q * g
    assert sp.simplify(E_full - E.subs(sub)) == 0
    nu = vE * f + vI * g
    D = vE - vI
    r = p - q
    d = nu**2 / E
    Phi = nu * (be * f**2 - al * g**2 + r * f**2 * g**2) - 2 * D * (al * f * g**2 + be * f**2 * g + (p * f + q * g) * f**2 * g**2)
    # d'(f) = -nu * Phi / (f^2 g^2 E^2)
    assert sp.simplify(sp.diff(d, f) + nu * Phi / (f**2 * g**2 * E**2)) == 0
    c = [-al * vI, al * (2 * vI - 3 * D), (be - al + r) * vI + D * (6 * al - 2 * be - 2 * q),
         -2 * r * vI + D * (3 * be - 3 * al - r + 4 * q), r * vI + D * (2 * r - 2 * q), -r * D]
    assert sp.expand(Phi - sum(ck * f**k for k, ck in enumerate(c))) == 0
    assert sp.simplify(Phi.subs(f, 0) + al * vI) == 0 and sp.simplify(Phi.subs(f, 1) - be * vE) == 0
    # special cases
    quartic = sp.expand(Phi.subs(vE, vI) / vI)
    assert sp.expand(quartic - (be * f**2 - al * g**2 + r * f**2 * g**2)) == 0
    quad = sp.expand(Phi.subs({vE: vI, p: q}) / vI)
    a_, b_ = sp.symbols("a b", positive=True)
    root = a_ / (a_ + b_)
    assert sp.simplify(quad.subs({al: a_**2, be: b_**2, f: root})) == 0
    # difference-mode form: p = delta_E + k, q = delta_I + k
    dE_, dI_ = TEEE - 2 * TEIE + TIIE, TEEI - 2 * TEII + TIII
    k_ = 2 * (TEIE - TIIE + TEII - TEEI)
    assert sp.expand(p.subs(sub) - dE_ - k_) == 0 and sp.expand(q.subs(sub) - dI_ - k_) == 0
    # projective variables u = f/nu, w = g/nu: 1/d = alpha (u+w)^3/u + beta (u+w)^3/w + (p u + q w)(u+w)
    u, w = f / nu, g / nu
    assert sp.simplify(E / nu**2 - (al * (u + w)**3 / u + be * (u + w)**3 / w + (p * u + q * w) * (u + w))) == 0
    assert sp.simplify(vE * u + vI * w - 1) == 0 and sp.simplify(sp.diff(u, f) - vI / nu**2) == 0
    # inverse: Delta / v_I = E' / (2E - f E') is equivalent to Phi = 0
    assert sp.simplify(Phi - f**2 * g**2 * (sp.diff(E, f) * (vI + D * f) - 2 * D * E)) == 0
    # Psi = h - kappa and Phi = f^2 g^2 nu Psi
    Psi = be / g**2 - al / f**2 - (2 * D * E / nu - r)
    assert sp.simplify(Phi - f**2 * g**2 * nu * Psi) == 0
    return [sp.lambdify((al, be, p, q, vE, vI), ck) for ck in c]


def interior(coeffs, al, be, p, q, vE, vI):
    """Roots of Phi in (0,1) and d there."""
    c = [ck(al, be, p, q, vE, vI) for ck in coeffs]
    while abs(c[-1]) < 1e-14 * max(abs(x) for x in c):
        c = c[:-1]
    d = lambda f: (vE * f + vI * (1 - f))**2 / (al / f + be / (1 - f) + p * f + q * (1 - f))
    roots = sorted(float(z.real) for z in np.roots(c[::-1]) if abs(z.imag) < 1e-9 and 0 < z.real < 1)
    return roots, [float(d(z)) for z in roots]


def reduce(T, v):
    al, be = T[1, 1, 0], T[0, 0, 1]
    p = T[0, 0, 0] - T[1, 1, 0] - 2 * T[0, 0, 1] + 2 * T[0, 1, 1]
    q = 2 * T[0, 1, 0] - 2 * T[1, 1, 0] - T[0, 0, 1] + T[1, 1, 1]
    return al, be, p, q, v[0], v[1]


def complex_tensor(W, cond, dt, cutoff=12.):
    """T[a,b,c] without taking the real part, to test that the integral is real and symmetric."""
    from eicavity.pair import even_spectrum
    wts = cond["weights"]
    omega, spectrum = even_spectrum(cond["covariance"], dt)
    keep = np.abs(omega) <= cutoff
    omega, spectrum = omega[keep], spectrum[..., keep]
    a2 = cond["alpha"]**2 @ wts
    Q = np.array([(c.T * wts) @ c for c in spectrum])
    h = 1 / (1 + 1j * omega)
    T = np.zeros((2, 2, 2), dtype=complex)
    for lo in range(0, len(omega), 32):
        hi = min(lo + 32, len(omega))
        u = (h[lo:hi, None] * h[None, :])[..., None] * a2
        R = np.linalg.inv(np.eye(2) - u[..., :, None] * W)
        T += np.einsum("ijac,ijbc,ijc->abc", R, R.conj(), np.moveaxis(Q[:, lo:hi], 0, -1), optimize=True)
    return T, float(spectrum.min())


def tensor_for(G, rates):
    """Tensor and mean variances for a variance profile G (at f0) and target rates (E, I), Bernoulli edges."""
    from eicavity.multitype import solve_populations
    one = solve_populations(np.asarray(G, float), fractions=theory.F0, mean_rates=tuple(rates), dt=.08, horizon=100.,
                            degree_dispersion=1.)
    cond = one.conditional(96)
    T, _ = complex_tensor(np.asarray(G, float) * theory.F0, cond, one.time[1] - one.time[0])
    v = cond["covariance"][..., 0] @ cond["weights"]
    return T.real, v


def newton(al, be, p, q, vE, vI):
    D, r = vE - vI, p - q
    s = np.sqrt(al) + np.sqrt(be)
    f0 = np.sqrt(al) / s
    E0, nu0 = s**2 + (p * np.sqrt(al) + q * np.sqrt(be)) / s, (vE * np.sqrt(al) + vI * np.sqrt(be)) / s
    kappa0, hprime0 = 2 * D * E0 / nu0 - r, 2 * s**4 / np.sqrt(al * be)
    kprime0 = 2 * D * (r * nu0 - D * E0) / nu0**2
    return dict(f0=f0, kappa0=kappa0, h_prime0=hprime0, kappa_prime0=kprime0, f1=f0 + kappa0 / (hprime0 - kprime0),
                E0=E0, nu0=nu0, smallness=abs(kappa0) / (hprime0 * min(f0, 1 - f0)))


def generic_galois(trials=6):
    """Galois group of Phi for random integer parameters; S5 means no solution in radicals."""
    from sympy.polys.numberfields.galoisgroups import galois_group
    f = sp.symbols("f")
    rng = np.random.default_rng(0)
    out = []
    for _ in range(trials):
        al, be, p, q, vE, vI = [int(v) for v in rng.integers(1, 30, 6)]
        if vE == vI or p == q:
            continue
        D, r = vE - vI, p - q
        c = [-al * vI, al * (2 * vI - 3 * D), (be - al + r) * vI + D * (6 * al - 2 * be - 2 * q),
             -2 * r * vI + D * (3 * be - 3 * al - r + 4 * q), r * vI + D * (2 * r - 2 * q), -r * D]
        poly = sp.Poly(sum(ck * f**k for k, ck in enumerate(c)), f)
        if not poly.is_irreducible:
            out.append(dict(params=[al, be, p, q, vE, vI], irreducible=False))
            continue
        group, _ = galois_group(poly)
        out.append(dict(params=[al, be, p, q, vE, vI], irreducible=True, group=str(group), order=int(group.order()),
                        solvable=bool(group.is_solvable)))
    return out


def main():
    coeffs = symbolic()
    out = {}
    # ---- reference tensor, with the imaginary part kept
    _, arrays = theory.solve(1., .3, 1.)
    cond = dict(weights=arrays["weights"], covariance=arrays["covariance"].astype(float), alpha=arrays["alpha"])
    Tc, min_mass = complex_tensor(theory.G0 * theory.F0, cond, arrays["time"][1] - arrays["time"][0])
    T = Tc.real
    v = cond["covariance"][..., 0] @ cond["weights"]
    out["tensor_checks"] = dict(max_abs_imaginary_part=float(np.abs(Tc.imag).max()),
                                max_asymmetry_real_part=float(np.abs(T - T.transpose(1, 0, 2)).max()),
                                min_spectral_mass=min_mass,
                                source_matrices_min_eigenvalue=[float(np.linalg.eigvalsh(T[:, :, c]).min()) for c in range(2)])
    assert np.allclose(T, theory.composition_tensor()[1], atol=1e-12)
    al, be, p, q, vE, vI = reduce(T, v)
    D, r = vE - vI, p - q
    E = lambda f: al / f + be / (1 - f) + p * f + q * (1 - f)
    nu = lambda f: vE * f + vI * (1 - f)
    d = lambda f: nu(f)**2 / E(f)
    for f in (.5, .8, .94):   # reduced formula against the campaign's full block sum
        assert abs(d(f) - theory.composition(f)["dimension_fraction"]) < 1e-12
    c = [ck(al, be, p, q, vE, vI) for ck in coeffs]
    roots = np.roots(c[::-1])
    inside, _ = interior(coeffs, al, be, p, q, vE, vI)
    assert len(inside) == 1
    fstar = inside[0]
    grid = np.linspace(1e-4, 1 - 1e-4, 2_000_001)
    assert abs(grid[np.argmax(d(grid))] - fstar) < 1e-6
    n = newton(al, be, p, q, vE, vI)
    Psi = lambda f: be / (1 - f)**2 - al / f**2 - (2 * D * E(f) / nu(f) - r)
    dPsi = lambda f: 2 * be / (1 - f)**3 + 2 * al / f**3 - 2 * D * ((-al / f**2 + be / (1 - f)**2 + r) * nu(f) - D * E(f)) / nu(f)**2
    f_two = n["f1"] - Psi(n["f1"]) / dPsi(n["f1"])
    plateau = lambda level: [float(grid[d(grid) >= level * d(fstar)][0]), float(grid[d(grid) >= level * d(fstar)][-1])]
    a, b = T[0, 0]
    Eprime = lambda f: -al / f**2 + be / (1 - f)**2 + r
    # energy at the optimum by block and source: f_a f_b T_abc / f_c
    fr = np.array([fstar, 1 - fstar])
    split = {f"{'EI'[i]}{'EI'[j]}_from_{'EI'[k]}": float((2 if i != j else 1) * fr[i] * fr[j] * T[i, j, k] / fr[k])
             for i in range(2) for j in range(i, 2) for k in range(2)}
    assert abs(sum(split.values()) - E(fstar)) < 1e-12
    # convexity of 1/d along the segment v_E u + v_I w = 1 (unit tangent (v_I, -v_E)/|v|)
    t = np.linspace(.002, .998, 4991)
    uu, ww = t / nu(t), (1 - t) / nu(t)
    inv = lambda u_, w_: al * (u_ + w_)**3 / u_ + be * (u_ + w_)**3 / w_ + (p * u_ + q * w_) * (u_ + w_)
    eps = 1e-5
    tang = np.array([vI, -vE]) / np.hypot(vI, vE)
    curv = (inv(uu + eps * tang[0], ww + eps * tang[1]) - 2 * inv(uu, ww) + inv(uu - eps * tang[0], ww - eps * tang[1])) / eps**2
    quad_curv = -2 * D * (p * vI - q * vE) / (vI**2 + vE**2)
    out["reference"] = dict(
        tensor=dict(T_EEE=T[0, 0, 0], T_EIE=T[0, 1, 0], T_IIE=T[1, 1, 0], T_EEI=T[0, 0, 1], T_EII=T[0, 1, 1], T_III=T[1, 1, 1]),
        v_E=vE, v_I=vI, alpha=al, beta=be, p=p, q=q, r=r, Delta=D,
        delta_E=T[0, 0, 0] - 2 * T[0, 1, 0] + T[1, 1, 0], delta_I=T[0, 0, 1] - 2 * T[0, 1, 1] + T[1, 1, 1],
        k=2 * (T[0, 1, 0] - T[1, 1, 0] + T[0, 1, 1] - T[0, 0, 1]),
        quintic_coefficients_c0_to_c5=c, all_roots=sorted(float(z.real) for z in roots),
        d_at_all_roots=[float(d(z.real)) for z in sorted(roots, key=lambda z: z.real)],
        f_star=fstar, d_max=d(fstar), E_at_optimum=E(fstar), nu_at_optimum=nu(fstar),
        energy_by_block_and_source_at_optimum=split,
        pole_terms_alpha_over_f_plus_beta_over_g=al / fstar + be / (1 - fstar),
        f0=n["f0"], d_at_f0_relative=d(n["f0"]) / d(fstar) - 1, kappa0=n["kappa0"], h_prime0=n["h_prime0"],
        kappa_prime0=n["kappa_prime0"], f_one_newton_step=n["f1"], newton_error=n["f1"] - fstar,
        d_at_newton_relative=d(n["f1"]) / d(fstar) - 1, f_two_newton_steps=f_two, newton_smallness=n["smallness"],
        f_star_if_Delta_zero=interior(coeffs, al, be, p, q, vI, vI)[0], f_star_if_r_zero=interior(coeffs, al, be, q, q, vE, vI)[0],
        plateau_1_percent=plateau(.99), plateau_0p1_percent=plateau(.999),
        relative_curvature=float((d(fstar + 1e-4) - 2 * d(fstar) + d(fstar - 1e-4)) / 1e-8 / d(fstar)),
        f_star_total_E_dimension=float(np.sqrt(a) / (np.sqrt(a) + np.sqrt(b))),
        d_at_total_E_optimum_relative=d(np.sqrt(a) / (np.sqrt(a) + np.sqrt(b))) / d(fstar) - 1,
        inverse_formula_Delta_over_vI=[float(Eprime(fstar) / (2 * E(fstar) - fstar * Eprime(fstar))), float(D / vI)],
        uniqueness=dict(sufficient_condition_Delta_times_pvI_minus_qvE=float(D * (p * vI - q * vE)),
                        quadratic_curvature_along_segment=float(quad_curv),
                        min_total_curvature_of_inverse_d_along_segment=float(curv.min())),
        d_at_fractions={f"{f:g}": d(f) for f in (.5, .55, .6, .65, .7, .75, .8, .84, .86, .88, .9, .92, .94)},
        sensitivity_of_f_star_to_plus_5_percent={
            "v_E": interior(coeffs, al, be, p, q, 1.05 * vE, vI)[0][0] - fstar,
            "v_I": interior(coeffs, al, be, p, q, vE, 1.05 * vI)[0][0] - fstar,
            "alpha": interior(coeffs, 1.05 * al, be, p - .05 * al, q - .1 * al, vE, vI)[0][0] - fstar,
            "beta": interior(coeffs, al, 1.05 * be, p - .1 * be, q - .05 * be, vE, vI)[0][0] - fstar})
    # sensitivity of f_star to the quadrature (lag step, horizon, Hermite nodes, frequency cutoff)
    quad = {}
    for dt_, hor_, nh_, cut_ in ((.06, 120., 128, 16.), (.1, 80., 64, 10.), (.08, 100., 96, 24.)):
        _, arr_ = theory.solve(1., .3, 1., dt_, hor_, nh_, cut_)
        cond_ = dict(weights=arr_["weights"], covariance=arr_["covariance"].astype(float), alpha=arr_["alpha"])
        Tq, _ = complex_tensor(theory.G0 * theory.F0, cond_, arr_["time"][1] - arr_["time"][0], cut_)
        quad[f"dt={dt_},horizon={hor_},hermite={nh_},cutoff={cut_}"] = \
            interior(coeffs, *reduce(Tq.real, cond_["covariance"][..., 0] @ cond_["weights"]))[0][0] - fstar
    out["quadrature_shift_of_f_star"] = quad
    # range of f_star for which the inverse formula returns a positive variance ratio v_E / v_I
    ratio = (2 * E(grid) + (1 - grid) * Eprime(grid)) / (2 * E(grid) - grid * Eprime(grid))
    ok = grid[ratio > 0]
    assert np.all(np.diff(ok) < 2e-6)          # one interval
    out["inverse_formula_admissible_range"] = [float(ok[0]), float(ok[-1])]
    out["reference"]["argmin_curvature_f"] = float(t[np.argmin(curv)])
    # missing I-to-I edges per row on the simulated path, N = 2000
    Gii, Jii2 = theory.G0[1, 1], theory.J0[1, 1]**2
    out["absent_I_to_I_edges_per_row_N2000"] = {f"{f:g}": float(2000 * (1 - f)**2 * Gii / ((1 - f) * Gii + .2 * Jii2))
                                                for f in (fstar, .9, .94, .97)}

    # ---- uniqueness fails for some admissible parameters
    abstract = (0.001, 0.001, 8.997, 0.997, 3., 1.)
    ra, da = interior(coeffs, *abstract)
    ff = np.linspace(1e-4, 1 - 1e-4, 20001)
    assert np.all(abstract[0] / ff + abstract[1] / (1 - ff) + abstract[2] * ff + abstract[3] * (1 - ff) > 0)
    W = np.array([[21.769, 0.1731], [0.184, 35.9971]])
    Tm, vm = tensor_for(W / theory.F0, (0.2615, 0.0809))
    pm = reduce(Tm, vm)
    rm, dm = interior(coeffs, *pm)
    out["uniqueness_counterexamples"] = dict(
        abstract=dict(alpha_beta_p_q_vE_vI=abstract, interior_roots=ra, d=da,
                      sufficient_condition=float((abstract[4] - abstract[5]) * (abstract[2] * abstract[5] - abstract[3] * abstract[4]))),
        conditional_theory=dict(W=W.tolist(), rates=[0.2615, 0.0809], alpha_beta_p_q_vE_vI=[float(x) for x in pm],
                                interior_roots=rm, d=dm, newton=newton(*pm),
                                sufficient_condition=float((pm[4] - pm[5]) * (pm[2] * pm[5] - pm[3] * pm[4]))))

    # ---- the Newton formula with unequal target rates at the reference couplings
    out["unequal_rates"] = {}
    for rates in ((.3, .3), (.25, .35), (.2, .4), (.1, .5), (.5, .1), (.15, .15)):
        try:
            Tr, vr = tensor_for(theory.G0, rates)
        except Exception as exc:
            out["unequal_rates"][str(rates)] = dict(error=repr(exc)[:80])
            continue
        pr = reduce(Tr, vr)
        rr, dr = interior(coeffs, *pr)
        nr = newton(*pr)
        dfun = lambda f: (pr[4] * f + pr[5] * (1 - f))**2 / (pr[0] / f + pr[1] / (1 - f) + pr[2] * f + pr[3] * (1 - f))
        best = rr[int(np.argmax(dr))]
        out["unequal_rates"][str(rates)] = dict(interior_roots=rr, f_star=best, f0=nr["f0"], f1=nr["f1"], smallness=nr["smallness"],
                                                loss_at_f1=float(dfun(nr["f1"]) / max(dr) - 1) if 0 < nr["f1"] < 1 else None,
                                                p=float(pr[2]), q=float(pr[3]))
    out["generic_galois"] = generic_galois()

    # ---- simulations: the 13 x 48 sweep of the paper campaign
    rows = [x for x in csv.DictReader(open(HERE.parent / "paper_campaign/networks.csv"))
            if x["family"] == "composition" and x["fluctuating"] == "1"]
    cells = {}
    for x in rows:
        cells.setdefault(float(x["fraction_e"]), []).append((float(x["d"]), float(x["nu"]), float(x["energy"])))
    sweep, chi2 = {}, 0.
    for f, vals in sorted(cells.items()):
        arr = np.array(vals)
        m, se = arr[:, 0].mean(), arr[:, 0].std(ddof=1) / np.sqrt(len(arr))
        chi2 += ((m - d(f)) / se)**2
        sweep[f"{f:g}"] = dict(mean=float(m), se=float(se), sd=float(arr[:, 0].std(ddof=1)), networks=len(arr), theory=d(f),
                               z=float((m - d(f)) / se), ratio_of_means=float(arr[:, 1].mean()**2 / arr[:, 2].mean()),
                               nu_over_theory=float(arr[:, 1].mean() / nu(f)), energy_over_theory=float(arr[:, 2].mean() / E(f)))
    out["sweep_13x48"] = dict(cells=sweep, chi_square=float(chi2), degrees_of_freedom=len(sweep))
    # ---- simulations: the 624 seed triplets of the readout confirmation (joint D/N per cell)
    trip = {}
    for x in csv.DictReader(open(HERE.parent / "readout_confirmation/network_estimates.csv")):
        if x["window"] == "T8000_gap0" and x["joint_per_cell"] not in ("", "None"):
            trip.setdefault(int(x["seed"]), {})[float(x["fraction_e"])] = float(x["joint_per_cell"])
    from scipy.stats import t as student
    conf = {}
    for f in (.8, .88, .94):
        vals = np.array([s_[f] for s_ in trip.values() if f in s_])
        conf[f"{f:g}"] = dict(mean=float(vals.mean()), se=float(vals.std(ddof=1) / np.sqrt(len(vals))), sd=float(vals.std(ddof=1)),
                              networks=len(vals), theory=d(f), ratio=float(vals.mean() / d(f)))
    for lo, hi in ((.8, .88), (.88, .94)):
        diff = np.array([s_[hi] - s_[lo] for s_ in trip.values() if lo in s_ and hi in s_])
        half = student.ppf(.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
        conf[f"{lo:g}_to_{hi:g}"] = dict(mean=float(diff.mean()), ci95=[float(diff.mean() - half), float(diff.mean() + half)],
                                         pairs=len(diff), theory=d(hi) - d(lo))
    ratios = np.array([conf[k]["ratio"] for k in ("0.8", "0.88", "0.94")])
    rse = np.array([conf[k]["se"] / conf[k]["theory"] for k in ("0.8", "0.88", "0.94")])
    common = np.sum(ratios / rse**2) / np.sum(1 / rse**2)
    conf["offset"] = dict(ratio_se=rse.tolist(), common_factor=float(common), chi_square_common=float(np.sum(((ratios - common) / rse)**2)),
                          difference_0p88_minus_0p80=[float(ratios[1] - ratios[0]), float(np.hypot(rse[0], rse[1]))])
    out["triplets_624"] = conf
    # the 13-cell sweep after one common factor
    m_ = np.array([c_["mean"] for c_ in sweep.values()]); t_ = np.array([c_["theory"] for c_ in sweep.values()]); e_ = np.array([c_["se"] for c_ in sweep.values()])
    fac = np.sum(m_ * t_ / e_**2) / np.sum(t_**2 / e_**2)
    hi_ = np.array([float(k_) >= .8 for k_ in sweep])
    fac_hi = np.sum((m_ * t_ / e_**2)[hi_]) / np.sum((t_**2 / e_**2)[hi_])
    out["sweep_13x48"]["common_factor"] = dict(factor=float(fac), chi_square=float(np.sum(((m_ - fac * t_) / e_)**2)), dof=len(m_) - 1,
                                               factor_f_ge_0p8=float(fac_hi),
                                               chi_square_f_ge_0p8=float(np.sum((((m_ - fac_hi * t_) / e_)**2)[hi_])), dof_f_ge_0p8=int(hi_.sum() - 1))
    # ---- how many networks a model-free comparison of 0.851 and 0.887 would need (two-sided 5%, power 80%)
    delta = d(fstar) - d(out["reference"]["f_star_total_E_dimension"])
    sd = np.mean([c_["sd"] for k_, c_ in sweep.items() if .8 <= float(k_) <= .9])
    out["power"] = dict(predicted_difference=float(delta), relative=float(delta / d(fstar)), single_network_sd_N2000=float(sd),
                        networks_per_fraction_N2000=float(2 * (1.959964 + 0.841621)**2 * (sd / delta)**2))
    (HERE / "values.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: out[k] for k in out if k not in ("generic_galois",)}, indent=1, default=float))


if __name__ == "__main__":
    main()
