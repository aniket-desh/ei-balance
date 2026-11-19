"""Evaluate the prespecified independent networks, search rules and subspace controls."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spectral_pilot"))
import analyze as pilot
from simulate import set_memory_limit
from eicavity.network import NetworkConfig, make_network
from eicavity.single_site import rate, rate_prime
import mode_variance
import numpy as np
from scipy.linalg import eig, schur, svd
from threadpoolctl import threadpool_limits

def rate_basis(values, directions, gains):
    # Complete a conjugate pair at the cutoff so conjugating the basis does not change the span.
    n = len(values)
    cut = np.sort(values.real)[-int(np.ceil(.10*n))]
    selected = (values.real >= cut) & (values.imag >= 0)
    u = directions[:, selected]
    columns = np.column_stack((u.real, u[:, values[selected].imag > 0].imag))
    left, singular, _ = svd(gains[:, None]*columns, full_matrices=False)
    rank = np.sum(singular > np.finfo(float).eps * max(columns.shape) * singular[0])
    Q = left[:, :rank]
    np.testing.assert_allclose(Q.T @ Q, np.eye(rank), atol=1e-12)
    return Q, float(cut)

def projection_metrics(C, Q, trace, energy):
    # For P=QQ^T, ||(I-P)C(I-P)||_F² = ||C||_F² - 2||CQ||_F² + ||Q^TCQ||_F².
    CQ = C @ Q
    block = Q.T @ CQ
    projected_trace = np.trace(block)
    projected_energy = np.vdot(block, block).real
    residual_energy = energy - 2*np.vdot(CQ, CQ).real + projected_energy
    residual_pr = (trace - projected_trace)**2 / residual_energy
    return np.array([projected_trace / trace, projected_energy / energy, residual_pr / (trace**2 / energy)])

def permute(groups, rng):
    index = np.arange(sum(map(len, groups)))
    for group in groups:
        index[group] = rng.permutation(group)
    return index

def measure(record_path):
    set_memory_limit()
    with threadpool_limits(limits=1):
        return measure_one(record_path)

def measure_one(record_path):
    record = json.loads(record_path.read_text())
    job = record["job"]
    cfg = NetworkConfig(**job["config"])
    name = record["name"]
    trace_path = HERE/"tmp"/(name+"_trajectory.npy")
    out = HERE/"measurements"/(name+".json")
    science_sources = [Path(__file__), Path(pilot.__file__), Path(mode_variance.__file__), HERE/"PROTOCOL.md"]
    source_hashes = {p.name:pilot.file_hash(p) for p in science_sources}
    record_hash = pilot.file_hash(record_path)
    trace_hash = pilot.file_hash(trace_path)
    if out.exists():
        existing = json.loads(out.read_text())
        if (existing["source_hashes"] == source_hashes and existing["run_sha256"] == record_hash
                and existing["trajectory_sha256"] == trace_hash
                and all((HERE/r["mode_file"]).is_file() and pilot.file_hash(HERE/r["mode_file"])==r["mode_sha256"]
                        for r in existing["roots"])):
            return name
    trajectory = np.load(trace_path, mmap_mode="r", allow_pickle=False)
    J, drive, _, labels, _ = make_network(cfg, np.asarray(job["mean_input"]))
    assert record["status"] == "complete" and pilot.hashlib.sha256(J).hexdigest() == record["connectivity_sha256"]
    assert trajectory.shape == (round(cfg.duration/cfg.sample_dt),cfg.n)
    roots, attempts = pilot.fixed_points(J, drive, job["mean_input"], trajectory, labels)
    input_cov = pilot.three_covariances(trajectory)
    rates = rate(np.asarray(trajectory,dtype=float))
    rate_cov = pilot.three_covariances(rates)
    mean_var = [float(np.trace(C)/cfg.n) for C in input_cov]
    fluctuating = min(mean_var[1:]) > 1e-10
    moments = [(float(np.trace(C)),float(np.vdot(C,C).real)) for C in rate_cov]
    case = {"name":name, "N":cfg.n, "seed":cfg.seed, "source_hashes":source_hashes,
            "run_sha256":record_hash, "trajectory_sha256":trace_hash, "connectivity_sha256":record["connectivity_sha256"],
            "state_fluctuating":fluctuating, "input_mean_temporal_variance":mean_var,
            "rate_direct_pr":[tr**2/en if en>0 and fluctuating else None for tr,en in moments],
            "final_drift_rms":record["final_drift_rms"],"root_attempts":attempts, "roots":[]}
    groups = [np.flatnonzero(labels==a) for a in (0,1)]
    for i,xstar in enumerate(roots):
        gains = rate_prime(xstar)
        A = J*gains[None,:]
        # A nearly silent cell can strongly unbalance J diag(g). Orthogonal Schur
        # reduction protects physical-coordinate residuals before recovering its right eigenvectors.
        triangular, rotation = schur(A, output="real")
        values,schur_vectors = eig(triangular)
        U = rotation @ schur_vectors
        U /= np.linalg.norm(U,axis=0)
        keep = values.imag>=0
        eigerr = np.linalg.norm(A@U-U*values,axis=0).max()
        if eigerr > 1e-10:
            raise ArithmeticError(f"Inaccurate physical-coordinate eigenpairs in {name}: {eigerr}")
        v = np.stack([pilot.projected(C,U) for C in input_cov])
        # Independently center the streamed trajectory projections for eight probes.
        probes = np.linspace(0,cfg.n-1,8,dtype=int)
        direct = mode_variance.mode_variances(trajectory,U[:,probes])
        np.testing.assert_allclose(direct,v[:,probes],rtol=1e-9,atol=1e-11)
        Q, cutoff = rate_basis(values,U,gains)
        mode_file = HERE/"modes"/f"{name}_root{i}.npz"
        np.savez_compressed(mode_file,root=xstar,eigenvalues=values,input_variance=v,
                            real_eigenvalue_cutoff=cutoff,rate_basis=Q)
        item = {"index":i,"root_residual":float(np.max(np.abs(-xstar+J@rate(xstar)+drive))),
                "max_eigenpair_residual":float(eigerr),
                "spatial_vs_time_input_error":float(np.max(abs(direct-v[:,probes]))),
                "mode_file":str(mode_file.relative_to(HERE)),"mode_sha256":pilot.file_hash(mode_file),
                "max_real_coupling_eigenvalue":float(values.real.max()),
                "unstable_mode_count":int(np.sum(values.real>1)),
                "rate_rank":Q.shape[1],"real_eigenvalue_cutoff":cutoff}
        if fluctuating:
            item["rho_input"] = [pilot.correlation(values[keep].real,z[keep]) for z in v]
            resolved = keep & (np.abs(values.real) > 1e-8)
            item["resolved_real_parts_count"] = int(resolved.sum())
            item["resolved_real_parts_rho"] = pilot.correlation(values.real[resolved],v[0,resolved])
            item["half_rank_agreement"] = pilot.correlation(v[1,keep],v[2,keep])
            item["rate_projection"] = [projection_metrics(C,Q,tr,en).tolist() for C,(tr,en) in zip(rate_cov,moments)]
            # Check the compressed energy algebra against both an explicit residual covariance
            # and the projected rate time series.
            Pperp = np.eye(cfg.n)-Q@Q.T
            residual = Pperp@rate_cov[0]@Pperp
            tr,en = moments[0]
            pr_ratio = np.trace(residual)**2/np.sum(residual**2)/(tr*tr/en)
            scores = rates@Q
            scores -= scores.mean(axis=0)
            block = scores.T@scores/len(scores)
            reference = np.array([np.trace(block)/tr,np.sum(block**2)/en,pr_ratio])
            item["projection_validation_error"] = float(np.max(abs(reference-item["rate_projection"][0])))
            np.testing.assert_allclose(reference,item["rate_projection"][0],rtol=1e-9,atol=1e-11)
            strata = [g[indices] for g in groups for indices in np.array_split(np.argsort(gains[g],kind="stable"),10)]
            null_input, null_rate, null_gain = [],[],[]
            streams = np.random.SeedSequence(cfg.seed+813000).spawn(2)
            rng_pop,rng_gain = map(np.random.default_rng,streams)
            for _ in range(64):
                pop = permute(groups,rng_pop)
                gmatch = permute(strata,rng_gain)
                null_input.append(pilot.correlation(values.real[keep],pilot.projected(input_cov[0],U[pop][:,keep])))
                null_rate.append(projection_metrics(rate_cov[0],Q[pop],tr,en).tolist())
                null_gain.append(projection_metrics(rate_cov[0],Q[gmatch],tr,en).tolist())
            item["population_null_rho"] = null_input
            item["population_null_rate"] = null_rate
            item["gain_decile_null_rate"] = null_gain
        case["roots"].append(item)
    temp = out.with_suffix(".writing")
    temp.write_text(json.dumps(case,indent=2,allow_nan=False)+"\n")
    temp.replace(out)
    return name

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,choices=(1,2,3),default=2)
    args = parser.parse_args()
    for folder in ("measurements","modes"):
        (HERE/folder).mkdir(exist_ok=True)
    paths = sorted((HERE/"runs").glob("spectral_N*_seed*.json"))
    expected = {f"spectral_N{n}_seed{s}" for n,start in ((250,73101),(500,73201)) for s in range(start,start+12)}
    if {p.stem for p in paths}!=expected:
        raise ValueError("Run simulation to completion before measuring.")
    with ProcessPoolExecutor(args.workers) as pool:
        for name in pool.map(measure,paths,chunksize=1):
            print(name,flush=True)
