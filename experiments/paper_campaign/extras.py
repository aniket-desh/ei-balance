"""Measurements taken on the retained trajectory, beside the archived engine's estimators.

Everything here reads the float32 input trajectory x(t) saved by engine_v4 and
uses rates r = 2 Phi(x) in float64. None of it changes the dynamics.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "readout_pilot"))
from measure import readout_moments  # four-window mean-difference estimator for a principal submatrix

SUBSET_SEED = 20261119   # same priority rule as experiments/readout_confirmation
LAG_UNITS = 24           # lag maps cover -24..24 leak times in steps of one
KERNEL_UNITS = 40.       # dense autocovariances out to 40 leak times
SAMPLED = 192            # neurons per population for lag maps and kernels


def sampled_neurons(labels, count=SAMPLED):
    """Deterministic, outcome-independent sample; neuron order within a class is exchangeable."""
    out = []
    for a in range(2):
        g = np.flatnonzero(labels == a)
        out.append(g[np.linspace(0, len(g) - 1, min(count, len(g)), dtype=int)])
    return out


def readouts(traj, labels, config, sizes, draws=8, block=500.):
    """PR of uniform K-cell readouts; nested in K and matched across fractions by shared priorities.

    For class a and draw d, cells are ranked by rng.random(n)[:count] with the
    generator seeded by (SUBSET_SEED, network seed, a, d); the K smallest form the
    readout. Returns, per class and size, a list over draws of the estimator windows.
    """
    n, seed = config["n"], config["seed"]
    counts = np.bincount(labels, minlength=2)
    stride = round(block / config["sample_dt"])
    nb = len(traj) // stride
    result = {}
    for a, offset in enumerate((0, counts[0])):
        usable = [k for k in sizes if k <= counts[a]]
        if not usable:
            continue
        kmax = max(usable)
        order = np.stack([np.argsort(np.random.default_rng(np.random.SeedSequence([SUBSET_SEED, seed, a, d]))
                                     .random(n)[:counts[a]], kind="stable")[:kmax] + offset for d in range(draws)])
        unique, inverse = np.unique(order, return_inverse=True)
        loc = inverse.reshape(order.shape)
        means = np.empty((draws, nb, kmax))
        cov = np.empty((draws, nb, kmax, kmax))
        for b in range(nb):
            rates = 2 * ndtr(np.asarray(traj[b * stride:(b + 1) * stride][:, unique], dtype=float))
            for d in range(draws):
                z = rates[:, loc[d]]
                means[d, b] = z.mean(0)
                cov[d, b] = z.T @ z / stride - np.outer(means[d, b], means[d, b])
        for k in usable:
            rows = []
            for d in range(draws):
                windows = readout_moments(means[d][:, :k], cov[d][:, :k, :k], n, block)
                rows.append({w: {q: v[q] for q in ("PR", "trace", "frobenius_energy", "diagonal_energy")}
                             for w, v in windows.items()})
            result[f"{'EI'[a]}{k}"] = rows
    return result


def lagged(rates, lags):
    """C(L)[i,j] = mean_t r_i(t+L) r_j(t) for centered columns and nonnegative sample lags L."""
    t = len(rates)
    return np.stack([rates[L:].T @ rates[:t - L] / (t - L) for L in lags])


def lag_maps(traj, labels, config):
    """Cross-fitted two-lag pair energies from two halves of the recording.

    psi_ab(t1,t2) = N * mean over distinct (i in a, j in b) of C1_ij(t1) C2_ij(t2),
    symmetrized over which half supplies which factor. C1 and C2 come from disjoint
    halves, so the estimation noise of one factor does not multiply itself; each
    half is centered by its own mean, which leaves a relative bias of order
    (correlation time)/(half length). The same-neuron product, Q_a(t1,t2) =
    mean_i C1_ii(t1) C2_ii(t2), is the conditional source of the theory.
    """
    n = config["n"]
    step = round(1 / config["sample_dt"])
    groups = sampled_neurons(labels)
    ids = np.concatenate(groups)
    ne = len(groups[0])
    half = len(traj) // 2
    positive = np.arange(LAG_UNITS + 1) * step
    cov = []
    for h in range(2):
        r = 2 * ndtr(np.asarray(traj[h * half:(h + 1) * half][:, ids], dtype=float))
        r -= r.mean(0)
        c = lagged(r, positive)                                   # (25, n, n), lags 0..24
        cov.append(np.concatenate([c[:0:-1].transpose(0, 2, 1), c]))  # C(-L) = C(L)^T, lags -24..24
    sl = [slice(0, ne), slice(ne, len(ids))]
    psi, source = {}, {}
    for a, b in ((0, 0), (0, 1), (1, 1)):
        blocks = [c[:, sl[a], sl[b]].copy() for c in cov]
        na, nb = blocks[0].shape[1:]
        pairs = na * nb
        if a == b:
            diag = [np.einsum("lii->li", x).copy() for x in blocks]
            source["EI"[a]] = (.5 * (diag[0] @ diag[1].T + diag[1] @ diag[0].T) / na).astype(np.float32)
            for x in blocks:
                x[:, np.arange(na), np.arange(na)] = 0
            pairs = na * (na - 1)
        flat = [x.reshape(len(x), -1) for x in blocks]
        psi["EI"[a] + "EI"[b]] = (n * .5 * (flat[0] @ flat[1].T + flat[1] @ flat[0].T) / pairs).astype(np.float32)
    return dict({f"psi_{k}": v for k, v in psi.items()}, **{f"source_{k}": v for k, v in source.items()},
                lag_axis=np.arange(-LAG_UNITS, LAG_UNITS + 1, dtype=np.float32))


def kernels(traj, labels, config):
    """Mean rate and input autocovariances of the sampled neurons, whole-record centering."""
    nt = len(traj)
    lags = np.arange(round(KERNEL_UNITS / config["sample_dt"]) + 1)
    nf = next_fast_len(2 * nt)
    out = {"kernel_lags": (lags * config["sample_dt"]).astype(np.float32)}
    for a, g in enumerate(sampled_neurons(labels)):
        acx = np.zeros(len(lags))
        acr = np.zeros(len(lags))
        for first in range(0, len(g), 32):
            x = np.asarray(traj[:, g[first:first + 32]], dtype=float)
            r = 2 * ndtr(x)
            x -= x.mean(0)
            r -= r.mean(0)
            for acc, y in ((acx, x), (acr, r)):
                f = rfft(y, n=nf, axis=0)
                acc += (irfft(f * np.conj(f), n=nf, axis=0)[lags] / (nt - lags)[:, None]).sum(1)
        out[f"input_kernel_{'EI'[a]}"] = (acx / len(g)).astype(np.float32)
        out[f"rate_kernel_{'EI'[a]}"] = (acr / len(g)).astype(np.float32)
    return out


def spectrum(traj, labels, block=2000):
    """Eigenvalues of the whole-record-centered rate covariance: joint, E block, I block.

    Ordinary centering and no cross-window repair: these spectra carry the usual
    finite-record noise floor and are for inspecting shape, not for PR estimates.
    """
    nt, n = traj.shape
    total = np.zeros(n)
    c = np.zeros((n, n))
    for first in range(0, nt, block):
        r = 2 * ndtr(np.asarray(traj[first:first + block], dtype=float))
        total += r.sum(0)
        c += r.T @ r
    mean = total / nt
    c = c / nt - np.outer(mean, mean)
    ne = int(np.sum(labels == 0))
    return {"eig_joint": np.linalg.eigvalsh(c)[::-1].astype(np.float32),
            "eig_E": np.linalg.eigvalsh(c[:ne, :ne])[::-1].astype(np.float32),
            "eig_I": np.linalg.eigvalsh(c[ne:, ne:])[::-1].astype(np.float32)}


def tangent(job, action, drive, x, duration=2000.):
    """Largest finite-time Lyapunov exponent of the midpoint map, one tangent vector.

    Same update as experiments/runtime/tangent_checks.py: the tangent of
    dx/dt = -x + J phi(x) + drive is dv/dt = -v + J diag(phi'(x)) v.
    """
    c = job["config"]
    dt = c["dt"]
    v = np.random.default_rng(c["seed"] + 2000003).normal(size=c["n"])
    v /= np.linalg.norm(v)
    nb, ns = round(c["burn"] / dt), round(duration / dt)
    lyap = np.empty(ns)
    for step in range(nb + ns):
        rx = 2 * ndtr(x)
        gx = np.sqrt(2 / np.pi) * np.exp(-.5 * x * x)
        mid = x + .5 * dt * (-x + action(rx) + drive)
        vmid = v + .5 * dt * (-v + action(gx * v))
        rm = 2 * ndtr(mid)
        gm = np.sqrt(2 / np.pi) * np.exp(-.5 * mid * mid)
        vn = v + dt * (-vmid + action(gm * vmid))
        x = x + dt * (-mid + action(rm) + drive)
        norm = np.linalg.norm(vn)
        v = vn / norm
        if step >= nb:
            lyap[step - nb] = np.log(norm) / dt
    return {"largest_finite_time_exponent": float(lyap.mean()), "duration": duration,
            "eight_duration_blocks": [float(b.mean()) for b in np.array_split(lyap, 8)]}
