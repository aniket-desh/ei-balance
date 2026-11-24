"""Parameter-only predictions for every campaign condition, cached on disk.

One condition is (coupling_scale s, target rate m, degree dispersion chi) at the
reference fractions f0 = (0.8, 0.2), with variance profile s*G0 and mean
connectivity J0. Composition changes at fixed W = G diag(f) and J0 diag(f) reuse
the chi = 1 solve through the fraction-independent energy tensor
T[a,b,c] (Eq. `composition` in the manuscript), so no extra DMFT solve is needed.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eicavity.composition import dimension_at_fraction, energy_tensor
from eicavity.multitype import population_dimension, solve_populations

G0 = np.array([[16., 9.], [25., 12.]])
J0 = np.array([[1., -6.], [2., -10.]])
F0 = np.array([.8, .2])
CACHE = HERE / "theory_cache"


def solve(scale=1., rate=.3, chi=1., dt=.08, horizon=100., n_hermite=96, cutoff=12.):
    """Return (summary dict, arrays dict) for one condition; arrays hold the conditional law."""
    key = hashlib.sha256(json.dumps([scale, rate, chi, dt, horizon, n_hermite, cutoff]).encode()).hexdigest()[:16]
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "summary"}
            return json.loads(str(z["summary"])), arrays
    G = scale * G0
    one = solve_populations(G, fractions=F0, mean_rates=(rate, rate), dt=dt, horizon=horizon,
                            degree_dispersion=chi)
    pair = population_dimension(one, n_hermite=n_hermite, max_frequency=cutoff)
    means = np.array(pair["mean_variances"])
    seconds = np.array(pair["variance_seconds"])
    per_pair = np.array(pair["pair_covariance_energy"])   # N * E[C_ij^2] for distinct i,j
    block = np.outer(F0, F0) * per_pair + np.diag(F0 * seconds)  # E_ab, sums to total energy
    w = pair["weights"]
    summary = dict(scale=scale, rate=rate, chi=chi, G=G.tolist(), J0=J0.tolist(), fractions=F0.tolist(),
                   mean_inputs=one.mean_inputs.tolist(), static_variances=one.static_variances.tolist(),
                   dynamic_variances=one.dynamic_variances.tolist(),
                   alpha_first=(pair["alpha"] @ w).tolist(), alpha_second=(pair["alpha"]**2 @ w).tolist(),
                   mean_variances=means.tolist(), variance_seconds=seconds.tolist(),
                   pair_energy=per_pair.tolist(), block_energy=block.tolist(),
                   nu=float(F0 @ means), energy=float(pair["covariance_energy"]),
                   d=float(pair["dimension_fraction"]),
                   d_within=pair["within_population_dimension_fractions"],
                   response_radius=pair["response_radius"],
                   residuals=[one.bvp_residual, one.static_residual, one.boundary_residual])
    arrays = dict(time=one.time, kernels=one.kernels, z=pair["z"], weights=w, mu=pair["mu"],
                  means=pair["means"], alpha=pair["alpha"], covariance=pair["covariance"].astype(np.float32))
    np.savez_compressed(path, summary=json.dumps(summary), **arrays)
    return summary, arrays


@functools.lru_cache(maxsize=None)
def composition_tensor(rate=.3):
    """Fraction-independent T[a,b,c] and the one-site variance moments of the chi = 1 law."""
    summary, a = solve(1., rate, 1.)
    dt = a["time"][1] - a["time"][0]
    tensor = energy_tensor(G0 * F0, dict(weights=a["weights"], covariance=a["covariance"].astype(float),
                                         alpha=a["alpha"]), dt)
    v = a["covariance"][..., 0].astype(float)
    return summary, tensor, v @ a["weights"], v**2 @ a["weights"]


def composition(fraction_e, rate=.3):
    """Predictions along the fixed-aggregate-coupling path from the chi = 1 reference law."""
    summary, tensor, means, seconds = composition_tensor(rate)
    f = np.array([fraction_e, 1 - fraction_e])
    out = dimension_at_fraction(tensor, means, seconds, f)
    full = np.einsum("abc,c->ab", tensor, 1 / f)
    out.update(mean_variances=means.tolist(), variance_seconds=seconds.tolist(),
               block_energy=(np.outer(f, f) * full).tolist(), nu=float(f @ means),
               mean_inputs=summary["mean_inputs"])
    return out


def readout_ratio(fraction_e, k, n=2000):
    """Ratio-of-moments PR of a uniform K-cell readout (experiments/readout_theory)."""
    c = composition(fraction_e)
    counts = np.array([round(n * fraction_e), n - round(n * fraction_e)])
    m1, m2 = np.array(c["mean_variances"]), np.array(c["variance_seconds"])
    pair = np.diag(c["pair_covariance_energy"])
    out = []
    for a in range(2):
        m = counts[a]
        if k > m:
            out.append(None)
            continue
        out.append(float((k * m1[a]**2 + (m - k) / (m - 1) * (m2[a] - m1[a]**2)) / (m2[a] + (k - 1) * pair[a] / n)))
    return out


def averaging_controls(scale=1., rate=.3, chi=1., fraction_e=None, cutoff=12.):
    """Predicted d under the two premature-averaging approximations of the proposition.

    'spectra': replace Q_a = E_z[c c] by the product of z-averaged spectra.
    'gains':   replace A_a = E_z[alpha^2] by (E_z alpha)^2.
    Both keep the conditional numerator (mean variances), as in the manuscript's
    Fig. `local`; with fraction_e set, W = G F and the local law are held fixed.
    """
    from eicavity.pair import even_spectrum
    _, a = solve(scale, rate, chi)
    w = a["weights"]
    dt = a["time"][1] - a["time"][0]
    omega, spectrum = even_spectrum(a["covariance"].astype(float), dt)
    keep = np.abs(omega) <= cutoff
    omega, spectrum = omega[keep], spectrum[..., keep]
    f = F0 if fraction_e is None else np.array([fraction_e, 1 - fraction_e])
    W = scale * G0 * F0                      # fixed aggregate variance coupling
    h = 1 / (1 + 1j * omega)
    means = a["covariance"][..., 0].astype(float) @ w
    out = {}
    for name in ("full", "spectra", "gains"):
        a2 = (a["alpha"] @ w)**2 if name == "gains" else a["alpha"]**2 @ w
        if name == "spectra":
            mean_spec = np.einsum("azw,z->aw", spectrum, w)
            Q = np.einsum("ai,aj->aij", mean_spec, mean_spec)
        else:
            Q = np.array([(c.T * w) @ c for c in spectrum])
        total = np.zeros((2, 2), dtype=complex)
        for lo in range(0, len(omega), 32):
            hi = min(len(omega), lo + 32)
            u = (h[lo:hi, None] * h[None, :])[..., None] * a2
            R = np.linalg.inv(np.eye(2) - u[..., :, None] * W)
            q = np.moveaxis(Q[:, lo:hi, :] / f[:, None, None], 0, -1)
            total += np.einsum("...ac,...c,...bc->...ab", R, q, R.conj(), optimize=True).sum(axis=(0, 1))
        full = total.real
        energy = float(f @ full @ f)
        out[name] = dict(d=float((f @ means)**2 / energy), energy=energy,
                         overlap=float(full[0, 1] / np.sqrt(full[0, 0] * full[1, 1])),
                         d_within=(means**2 / (f * np.diag(full))).tolist())
    return out


def lag_map(scale=1., rate=.3, chi=1., fraction_e=None, lags=np.arange(-24, 25.), cutoff=12.):
    """Two-lag distinct-pair moments psi_ab(t1,t2) = N E[C_ij(t1) C_ij(t2)] and sources Q_a(t1,t2).

    With C_ij(t) = <r_i(t+t') r_j(t')> and transform c(w) = int e^{-iwt} c(t) dt, the
    first index propagates with U = H(w1) H(w2) diag(A), H = 1/(1+iw), so
    X(w1,w2) = R diag(Q/f) R^dagger, R = (I - U W)^{-1}, and
    psi_ab(t1,t2) = sum_{w1,w2} [X - diag(Q/f)]_ab e^{i(w1 t1 + w2 t2)} over spectral masses.
    Returns dict with psi (2,2,L,L) and source (2,L,L).
    """
    from eicavity.pair import even_spectrum
    _, a = solve(scale, rate, chi)
    w = a["weights"]
    cov = a["covariance"].astype(float)
    dt = a["time"][1] - a["time"][0]
    omega, spectrum = even_spectrum(cov, dt)
    keep = np.abs(omega) <= cutoff
    omega, spectrum = omega[keep], spectrum[..., keep]
    f = F0 if fraction_e is None else np.array([fraction_e, 1 - fraction_e])
    W = scale * G0 * F0
    a2 = a["alpha"]**2 @ w
    h = 1 / (1 + 1j * omega)
    Q = np.array([(c.T * w) @ c for c in spectrum])                    # (2, nw, nw)
    u = (h[:, None] * h[None, :])[..., None] * a2                      # (nw, nw, 2)
    R = np.linalg.inv(np.eye(2) - u[..., :, None] * W)                 # (nw, nw, 2, 2)
    q = np.moveaxis(Q / f[:, None, None], 0, -1)                       # (nw, nw, 2)
    X = np.einsum("...ac,...c,...bc->...ab", R, q, R.conj(), optimize=True)
    X[..., 0, 0] -= q[..., 0]
    X[..., 1, 1] -= q[..., 1]
    phase = np.exp(1j * np.outer(lags, omega))                         # (L, nw)
    psi = np.einsum("tk,klab,sl->abts", phase, X, phase, optimize=True)
    assert np.max(np.abs(psi.imag)) < 1e-6 * np.max(np.abs(psi.real))
    # same-neuron source directly in the time domain: E_z[c(z,t1) c(z,t2)], c even in lag
    idx = np.rint(np.abs(lags) / dt).astype(int)
    c_l = cov[..., idx]                                                # (2, nz, L)
    source = np.einsum("azt,z,azs->ats", c_l, w, c_l)
    return dict(lags=lags, psi=psi.real, source=source, mean_kernel=np.einsum("azt,z->at", c_l, w))


if __name__ == "__main__":
    m = lag_map()
    s0, _ = solve(1., .3, 1.)
    print("psi(0,0)", m["psi"][:, :, 24, 24], "pair energy", s0["pair_energy"])
    print("EE symmetric under joint lag reversal:", np.max(np.abs(m["psi"][0, 0] - m["psi"][0, 0][::-1, ::-1])))
    print(averaging_controls())
    for args in [(1., .3, 0.), (1., .3, 1.), (1., .3, 2.)]:
        s, _ = solve(*args)
        print(args, round(s["d"], 6), np.round(s["mean_inputs"], 6))
    print(composition(.8)["dimension_fraction"], readout_ratio(.8, 100), readout_ratio(.88, 100))
