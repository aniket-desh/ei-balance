"""Verify each root and compare measured modal variance with within-type permutations."""

from pathlib import Path
import hashlib
import json
import resource
import sys
import tarfile

from simulate import HERE, set_memory_limit
import numpy as np
from scipy.linalg import eig, solve
from scipy.optimize import root
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from eicavity.network import NetworkConfig, make_network
from eicavity.single_site import rate, rate_prime
import mode_variance


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def continue_to_full_coupling(J, drive, anchor):
    """Follow F=-x+anchor+lambda*(J phi(x)+drive-anchor)=0 through folds."""
    n = len(anchor)
    scale = np.sqrt(n)
    state = np.r_[anchor, 0.]  # Last coordinate is sqrt(N)*lambda.
    direction = np.r_[(J @ rate(anchor) + drive - anchor)/scale, 1.]
    direction /= np.linalg.norm(direction)
    step = 0.5
    eye = np.eye(n)
    low = high = 0.
    for count in range(1, 351):
        predicted = state + step * direction

        def residual(z):
            response = J @ rate(z[:-1]) + drive - anchor
            F = -z[:-1] + anchor + (z[-1]/scale)*response
            return np.r_[F, (z-predicted) @ direction]

        def derivative(z):
            mat = np.empty((n+1, n+1))
            mat[:n,:n] = -eye + (z[-1]/scale)*J*rate_prime(z[:-1])[None,:]
            mat[:n,-1] = (J @ rate(z[:-1]) + drive - anchor)/scale
            mat[-1] = direction
            return mat

        fit = root(residual, predicted, jac=derivative, options={"xtol": 1e-10, "maxfev": 35})
        if np.max(np.abs(residual(fit.x))) > 1e-9:
            step /= 2
            if step < 0.001:
                break
            continue
        next_state = fit.x
        strength = next_state[-1]/scale
        low, high = min(low, strength), max(high, strength)
        if state[-1] < scale <= next_state[-1]:
            weight = (scale-state[-1])/(next_state[-1]-state[-1])
            guess = state[:-1]*(1-weight) + next_state[:-1]*weight
            final = root(lambda x: -x + J @ rate(x) + drive, guess,
                         jac=lambda x: -eye + J*rate_prime(x)[None,:], options={"xtol": 1e-10})
            if np.max(np.abs(-final.x + J @ rate(final.x) + drive)) <= 1e-8:
                return final.x, {"steps": count, "min_lambda": low, "max_lambda": high}
            step /= 2
            if step < 0.001:
                break
            continue
        if strength < -0.35:
            break
        direction = solve(derivative(next_state), np.r_[np.zeros(n), 1.])
        direction /= np.linalg.norm(direction)
        state = next_state
        step = min(1.5, step * 1.2)
    return None, {"steps": count, "min_lambda": low, "max_lambda": high}


def fixed_points(J, drive, mean_input, trajectory, labels):
    def drift(x):
        return -x + J @ rate(x) + drive

    def jac(x):
        A = J * rate_prime(x)[None, :]
        A.flat[::len(x)+1] -= 1
        return A

    initial = {
        "record_mean": trajectory.mean(axis=0, dtype=float),
        "first_sample": np.asarray(trajectory[0], dtype=float),
        "population_offset": np.asarray(mean_input)[labels],
    }
    roots, attempts = [], []
    for key, guess in initial.items():
        fit = root(drift, guess, jac=jac, options={"xtol": 1e-10})
        residual = float(np.max(np.abs(drift(fit.x))))
        item = {"initial": key, "residual_inf": residual, "solver_status": fit.message, "nfev": fit.nfev}
        if np.isfinite(residual) and residual <= 1e-8:
            same = next((i for i, z in enumerate(roots) if np.max(np.abs(fit.x-z)) < 1e-5), None)
            if same is None:
                same = len(roots)
                roots.append(fit.x.copy())
            item["root_index"] = same
        attempts.append(item)
    if not roots:
        # Recovery on the same realizations, retaining full-strength verified
        # roots only. Try both anchors even if the first one succeeds.
        for key in ("record_mean", "population_offset"):
            candidate, path = continue_to_full_coupling(J, drive, initial[key])
            item = {"initial": "continuation_" + key, **path}
            if candidate is not None:
                item["residual_inf"] = float(np.max(np.abs(drift(candidate))))
                same = next((i for i, z in enumerate(roots) if np.max(np.abs(candidate-z)) < 1e-5), None)
                if same is None:
                    same = len(roots)
                    roots.append(candidate)
                item["root_index"] = same
            attempts.append(item)
    return roots, attempts


def three_covariances(trajectory):
    # This is an independent spatial calculation of the streamed time-series projections.
    x = np.array(trajectory, dtype=float)
    C = []
    for t in (x, x[:len(x)//2], x[len(x)//2:]):
        xc = t - t.mean(0)
        C.append((xc.T @ xc) / len(xc))
    return C


def projected(C, U):
    return np.sum(U.conj() * (C @ U), axis=0).real


def correlation(x, y):
    return float(spearmanr(x, y).statistic)


def analyze_case(record_path):
    record = json.loads(record_path.read_text())
    name = record["name"]
    job = record["job"]
    cfg = NetworkConfig(**job["config"])
    traj_path = HERE / "tmp" / (name + "_trajectory.npy")
    traj = np.load(traj_path, mmap_mode="r", allow_pickle=False)
    J, drive, _, labels, _ = make_network(cfg, np.asarray(job["mean_input"]))
    assert record["status"] == "complete"
    assert traj.shape == (round(cfg.duration/cfg.sample_dt), cfg.n)
    assert hashlib.sha256(J).hexdigest() == record["connectivity_sha256"]
    mean = traj.mean(0, dtype=float)
    for stat in record["estimators"][0]["population_local_stats"]:
        np.testing.assert_allclose(mean[labels == stat["population"]].mean(), stat["h"], rtol=1e-11, atol=1e-12)
    roots, attempts = fixed_points(J, drive, job["mean_input"], traj, labels)
    cov = three_covariances(traj)
    mean_variances = [float(C.trace()/cfg.n) for C in cov]
    fluctuating = min(mean_variances[1:]) > 1e-10
    summary = {"name": name, "N": cfg.n, "seed": cfg.seed, "trajectory_sha256": file_hash(traj_path),
               "connectivity_sha256": record["connectivity_sha256"], "record_sha256": file_hash(record_path),
               "mean_input_variance_full_half1_half2": mean_variances, "fluctuating": fluctuating,
               "final_drift_rms": record["final_drift_rms"], "attempts": attempts, "roots": []}
    for i, fixed in enumerate(roots):
        base = HERE / "modes" / f"{name}_root{i}"
        fp = base.with_suffix(".npy")
        output = base.with_suffix(".npz")
        np.save(fp, fixed)
        mode_variance.main([str(record_path), "--trajectory", str(traj_path),
                            "--fixed-point", str(fp.relative_to(Path.cwd()) if fp.is_relative_to(Path.cwd()) else fp),
                            "--variable", "input", "--output", str(output)])
        with np.load(output, allow_pickle=False) as data:
            variance = data["variance"]
            residual = float(data["fixed_point_residual_inf"])
            A = J * rate_prime(fixed)[None, :]
            eigenvalues, U = eig(A)
            U /= np.linalg.norm(U, axis=0)
            eig_residual = float(np.max(np.linalg.norm(A @ U-U*eigenvalues, axis=0)))
            np.testing.assert_allclose(data["eigenvalues"], eigenvalues, rtol=1e-9, atol=1e-10)
            reference = np.array([projected(C, U) for C in cov])
            err = float(np.max(np.abs(reference-variance)))
            np.testing.assert_allclose(reference, variance, rtol=5e-10, atol=1e-13)
        # Keep only one member of each conjugate pair for the rank statistics.
        keep = eigenvalues.imag >= 0
        item = {"index": i, "residual_inf": residual, "max_eigenpair_residual": eig_residual,
                "spatial_vs_temporal_variance_max_abs_error": err, "variance_file": str(output.relative_to(HERE)),
                "unstable_eigenvalues_with_conjugates": int(np.sum(eigenvalues.real > 1)),
                "representative_modes": int(keep.sum())}
        if fluctuating:
            rng = np.random.default_rng(901000 + cfg.seed + cfg.n)
            null = []
            for _ in range(32):
                permutation = np.arange(cfg.n)
                for a in range(2):
                    ids = np.flatnonzero(labels == a)
                    permutation[ids] = rng.permutation(ids)
                shuffled = projected(cov[0], U[permutation][:, keep])
                null.append(correlation(eigenvalues.real[keep], shuffled))
            item.update(real_part_variance_spearman=[correlation(eigenvalues.real[keep], v[keep]) for v in variance],
                        half_variance_spearman=correlation(variance[1, keep], variance[2, keep]),
                        half_relative_l2=float(np.linalg.norm(variance[1,keep]-variance[2,keep]) /
                                               np.linalg.norm(variance[0,keep])),
                        within_type_shuffle_full_spearman=null)
        summary["roots"].append(item)
    print(name, "fluctuating", fluctuating, "root attempts", attempts, flush=True)
    return summary


def main():
    set_memory_limit()
    (HERE / "modes").mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        cases = [analyze_case(p) for p in sorted((HERE / "runs").glob("spectral_N*_seed*.json"))]
    assert len(cases) == 4
    source = HERE.parent.parent / "recovered/bundle/ei_full/ei_balance_handoff/src/eicavity"
    files = [HERE / "simulate.py", Path(__file__), HERE.parent / "runtime/engine_v4.py",
             Path(mode_variance.__file__), source / "network.py", source / "single_site.py"]
    with tarfile.open(HERE / "source.tar.gz", "w:gz") as archive:
        for p in files:
            archive.add(p, arcname=p.resolve().relative_to(HERE.parent.parent))
    summary = {"cases": cases, "source_sha256": {str(p.resolve().relative_to(HERE.parent.parent)): file_hash(p) for p in files},
               "analysis_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
