"""Fixed-point modes and rate covariance on a larger cohort, with the replication's frozen method.

    python experiments/paper_campaign/spectral.py --workers 30

Every measurement is made by experiments/spectral_replication/measure.py (root search
order, Schur eigenvectors, top-tenth rate subspace, 64 within-population and 64
gain-decile permutations); only the cohort changes:

    Bernoulli, 80:20          N = 250, 500, 1000 (32 networks each), N = 2000 (16)
    degree dispersion 0, 2    N = 1000 (16 networks each)

Recordings are 2,000 leak times after 500 of burn-in, as in the replication. Seeds
46001 onward are new. A case that exceeds the time limit is recorded as timed out.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"

HERE = Path(__file__).resolve().parent
SPEC = HERE / "spectral"
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "src"), str(HERE.parent / "runtime"), str(HERE.parent / "spectral_pilot"),
                str(HERE.parent / "spectral_replication")]


def jobs():
    from prepare import composition_config, degree_config
    out = []
    for n, count in ((250, 32), (500, 32), (1000, 32), (2000, 16)):
        for i in range(count):
            config, h = composition_config(n, 46001 + i, .8)
            out.append(dict(name=f"spectral_bernoulli_N{n}_seed{46001 + i}", cell="bernoulli", config=config, mean_input=h))
    for chi in (0, 2):
        for i in range(16):
            config, h = degree_config(1000, 46101 + i, chi)
            out.append(dict(name=f"spectral_chi{chi}_N1000_seed{46101 + i}", cell=f"chi{chi}", config=config, mean_input=h))
    for job in out:
        job["config"] = dict(job["config"], duration=2000.)
        job.update(experiment="paper_campaign_spectral", blocks=[250.], prefixes=[2000.], retain_temporary_trajectory=True)
    return out


def run_case(job, library):
    import numpy as np
    import engine_v4 as engine
    import measure as replication
    engine.HERE = replication.HERE = SPEC
    name = job["name"]
    engine.run_one(dict(job, native_library=library))
    replication.measure_one(SPEC / "runs" / f"{name}.json")
    case = json.loads((SPEC / "measurements" / f"{name}.json").read_text())
    if case["roots"]:   # compact copy of the primary root's spectrum for plotting
        with np.load(SPEC / case["roots"][0]["mode_file"]) as modes:
            np.savez_compressed(SPEC / "spectra" / f"{name}.npz", eigenvalues=modes["eigenvalues"].astype(np.complex64),
                                input_variance=modes["input_variance"].astype(np.float32))
    for path in list((SPEC / "modes").glob(f"{name}_root*.npz")) + [SPEC / "tmp" / f"{name}_trajectory.npy"] \
            + list((SPEC / "runs").glob(f"{name}_B*_neuron_statistics.npz")):
        path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=3600., help="seconds per network")
    parser.add_argument("--one", help=argparse.SUPPRESS)
    parser.add_argument("--library", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for folder in ("tmp", "runs", "measurements", "modes", "spectra"):
        (SPEC / folder).mkdir(parents=True, exist_ok=True)
    cases = jobs()
    if args.one:
        run_case(next(j for j in cases if j["name"] == args.one), args.library)
        return
    from native_action import build
    library, _ = build(SPEC / "tmp" / f"kernel_{os.getpid()}")
    status = {}

    def execute(job):
        name = job["name"]
        if (SPEC / "measurements" / f"{name}.json").exists():
            status[name] = "complete"
            return
        tic = time.monotonic()
        try:
            done = subprocess.run([sys.executable, __file__, "--one", name, "--library", str(library)],
                                  timeout=args.timeout, capture_output=True, text=True)
            status[name] = "complete" if done.returncode == 0 else "failed: " + done.stderr.strip().splitlines()[-1][:200]
        except subprocess.TimeoutExpired:
            status[name] = f"timed out after {args.timeout:.0f} s"
        print(name, status[name], f"{time.monotonic() - tic:.0f} s", flush=True)
        (SPEC / "status.json").write_text(json.dumps(status, indent=1) + "\n")

    # largest networks first so the long cases start immediately
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(execute, sorted(cases, key=lambda j: -j["config"]["n"])))
    (SPEC / "status.json").write_text(json.dumps(status, indent=1) + "\n")
    print("spectral cohort finished", flush=True)


if __name__ == "__main__":
    main()
