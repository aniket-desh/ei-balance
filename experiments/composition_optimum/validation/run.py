"""Run one stripe of the composition-optimum validation, one single-threaded process per network.

    python experiments/composition_optimum/validation/run.py --part 3/8 --workers 30 --memory-gib 56 --scratch-gib 100

Copied from experiments/paper_campaign/run.py (SHA-256 recorded in README.md). The only
changes are paths: outputs go beside this file, the archived engine and extras.py are
imported from experiments/runtime and experiments/paper_campaign, the hashed source list
names those files at their real locations, and the default manifest is jobs.json here.

Each network is simulated and measured by the archived engine (runtime/engine_v4.py,
native constant-weight action), then the retained trajectory supplies the extra
measurements in extras.py. Outputs per network:

    records/<name>.json.gz   engine record (all estimator windows) plus scalar extras
    arrays/<name>.npz        per-neuron moments, in-degrees, kernels, lag maps, spectra

A network is complete when both exist. Stripes are a deterministic longest-job-first
partition of jobs.json by estimated cost, so different pods never share a network.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
RUNTIME = EXPERIMENTS / "runtime"
CAMPAIGN = EXPERIMENTS / "paper_campaign"
sys.path[:0] = [str(HERE), str(CAMPAIGN), str(RUNTIME)]
GIB = 1024**3


def done(name):
    return (HERE / "records" / f"{name}.json.gz").is_file() and (HERE / "arrays" / f"{name}.npz").is_file()


def ram_gib(job):
    """Resident estimate: dense J and its copy at generation, then covariance work arrays."""
    n = job["config"]["n"]
    return .45 + 6e-8 * n * n + (.9 if job["config"]["duration"] > 8000 else 0)


def disk_gib(job):
    c = job["config"]
    samples = round(c["duration"] / c["sample_dt"])
    windows = round(c["duration"] / min(job["blocks"]))
    return (samples * c["n"] * 4 + windows * c["n"]**2 * 8) / GIB + .1


def stripes(jobs, count):
    """Longest-processing-time partition; ties resolved by manifest order."""
    load = [0.] * count
    parts = [[] for _ in range(count)]
    for job in sorted(jobs, key=lambda j: -j["cost"]):
        k = load.index(min(load))
        parts[k].append(job)
        load[k] += job["cost"]
    return parts


def run_case(job, library):
    import numpy as np
    import engine_v4 as engine
    import extras
    from native_action import ConstantAction

    name, c = job["name"], job["config"]
    with TemporaryDirectory(dir=HERE / "tmp", prefix=name + "-") as tmp:
        scratch = Path(tmp)
        (scratch / "tmp").mkdir()
        (scratch / "runs").mkdir()
        engine.HERE = scratch
        original = engine.make_network
        stash = {}

        def recording_make_network(config, mean_input):
            # Same generator; additionally keep each neuron's realized E and I in-degree.
            J, drive, x, labels, info = original(config, mean_input)
            ne = int(np.sum(labels == 0))
            edges = J != 0
            stash["indegree"] = np.stack([edges[:, :ne].sum(1), edges[:, ne:].sum(1)], 1).astype(np.int32)
            return J, drive, x, labels, info

        engine.make_network = recording_make_network
        engine.run_one(dict(job, native_library=str(library), retain_temporary_trajectory=True))
        engine.make_network = original
        record = json.loads((scratch / "runs" / f"{name}.json").read_text())
        assert record["status"] == "complete"
        traj = np.load(scratch / "tmp" / f"{name}_trajectory.npy", mmap_mode="r")
        with np.load(scratch / "runs" / f"{name}_B{int(job['blocks'][0])}_neuron_statistics.npz") as cap:
            labels = cap["labels"]
            arrays = {k: cap[k].astype(np.float32) for k in ("mean_x", "q_x", "mean_rate", "variance_rate", "mean_gain")}
        arrays["indegree"] = stash["indegree"]
        arrays["n_e"] = np.int32(np.sum(labels == 0))
        extra = {}
        tic = time.monotonic()
        want = job["extras"]
        if want["readout_sizes"]:
            extra["readouts"] = extras.readouts(traj, labels, c, want["readout_sizes"])
        arrays.update(extras.kernels(traj, labels, c))
        if want["lag"]:
            arrays.update(extras.lag_maps(traj, labels, c))
        if want["spectrum"]:
            arrays.update(extras.spectrum(traj, labels))
        del traj
        if want["tangent"]:
            J, drive, x, labels2, info = original(engine.NetworkConfig(**c), np.array(job["mean_input"]))
            assert hashlib.sha256(J.tobytes()).hexdigest() == record["connectivity_sha256"]
            S, B = engine.sparse_complement(J, labels2, info)
            del J
            extra["tangent"] = extras.tangent(job, ConstantAction(S, B, labels2, info, Path(library)), drive, x)
        extra["extras_seconds"] = time.monotonic() - tic
        record["extras"] = extra
        record["runner_sha256"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in (Path(__file__), CAMPAIGN / "extras.py", EXPERIMENTS / "readout_pilot/measure.py")}
        record["host"] = dict(node=socket.gethostname(), platform=platform.platform(), python=sys.version.split()[0],
                              numpy=np.__version__)
        for est in record["estimators"]:
            est.pop("neuron_statistics_file", None)
        target = HERE / "arrays" / f"{name}.npz"
        np.savez_compressed(scratch / "arrays.npz", **arrays)
        (scratch / "arrays.npz").replace(target)
        partial = scratch / "record.json.gz"
        with gzip.open(partial, "wt") as handle:
            json.dump(record, handle, separators=(",", ":"), allow_nan=False)
        partial.replace(HERE / "records" / f"{name}.json.gz")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", default="1/1", help="1-based stripe, e.g. 3/8")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--memory-gib", type=float, default=8)
    parser.add_argument("--scratch-gib", type=float, default=40)
    parser.add_argument("--only", help="comma-separated substrings a job name must contain")
    parser.add_argument("--max-n", type=int, help="skip networks larger than this")
    parser.add_argument("--manifest", default="jobs.json", help="manifest beside this file")
    parser.add_argument("--one", help=argparse.SUPPRESS)
    parser.add_argument("--library", help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifest = HERE / args.manifest
    jobs = json.loads(manifest.read_text())
    for folder in ("tmp", "records", "arrays", "audit"):
        (HERE / folder).mkdir(exist_ok=True)
    if args.one:
        run_case(next(j for j in jobs if j["name"] == args.one), args.library)
        return

    import psutil
    from native_action import build
    stripe, count = map(int, args.part.split("/"))
    if not 1 <= stripe <= count:
        parser.error("invalid stripe")
    selected = stripes(jobs, count)[stripe - 1]
    if args.only:
        selected = [j for j in selected if any(s in j["name"] for s in args.only.split(","))]
    if args.max_n:
        selected = [j for j in selected if j["config"]["n"] <= args.max_n]
    pending = [j for j in selected if not done(j["name"])]
    identity = f"{socket.gethostname()}_{stripe}of{count}_{os.getpid()}"
    library, command = build(HERE / "tmp" / f"kernel_{identity}")
    sources = [Path(__file__), CAMPAIGN / "extras.py", manifest, RUNTIME / "engine_v4.py", RUNTIME / "native_action.py",
               RUNTIME / "constant_csr.cpp", EXPERIMENTS / "readout_pilot/measure.py",
               EXPERIMENTS.parent / "src/eicavity/network.py", EXPERIMENTS.parent / "src/eicavity/degree.py",
               EXPERIMENTS.parent / "recovered/bundle/ei_full/ei_balance_handoff/src/eicavity/network.py"]
    audit = dict(part=args.part, manifest=args.manifest, selected=len(selected), pending_at_start=len(pending), workers=args.workers,
                 memory_gib=args.memory_gib, scratch_gib=args.scratch_gib, platform=platform.platform(),
                 python=sys.version, cpu_count=os.cpu_count(), compile_command=command,
                 library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                 source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
                 start_unix=time.time(), failures=[], max_rss_gib=0.)
    audit_path = HERE / "audit" / f"{identity}.json"
    write = lambda: audit_path.write_text(json.dumps(audit, indent=1) + "\n")
    write()
    print(f"{len(pending)} of {len(selected)} pending, estimated {sum(j['cost'] for j in pending)/3600:.1f} core-hours",
          flush=True)
    active = {}
    checkpoint = time.monotonic()
    try:
        while pending or active:
            for proc, job in list(active.items()):
                if proc.poll() is not None:
                    del active[proc]
                    ok = proc.returncode == 0 and done(job["name"])
                    if not ok:
                        audit["failures"].append(job["name"])
                    print(f"{'done' if ok else 'FAILED'} {job['name']} ({len(pending)} pending, {len(active)} active)", flush=True)
            ram = sum(ram_gib(j) for j in active.values())
            disk = sum(disk_gib(j) for j in active.values())
            for job in list(pending):
                if len(active) >= args.workers:
                    break
                if active and (ram + ram_gib(job) > args.memory_gib or disk + disk_gib(job) > args.scratch_gib):
                    continue
                pending.remove(job)
                log = (HERE / "audit" / f"{job['name']}.log").open("w")
                proc = subprocess.Popen([sys.executable, __file__, "--one", job["name"], "--library", str(library),
                                         "--manifest", args.manifest],
                                        stdout=log, stderr=subprocess.STDOUT)
                active[proc] = job
                ram += ram_gib(job)
                disk += disk_gib(job)
            if time.monotonic() - checkpoint > 20:
                checkpoint = time.monotonic()
                rss = 0
                for proc in active:
                    try:
                        rss += psutil.Process(proc.pid).memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                audit.update(max_rss_gib=max(audit["max_rss_gib"], rss / GIB), last_unix=time.time(),
                             active=[j["name"] for j in active.values()], pending=len(pending),
                             free_scratch_gib=shutil.disk_usage(HERE / "tmp").free / GIB,
                             available_ram_gib=psutil.virtual_memory().available / GIB)
                write()
            time.sleep(1)
    finally:
        for proc in active:
            proc.terminate()
        audit.update(finished_unix=time.time(), complete=sum(done(j["name"]) for j in selected))
        write()
    for job in selected:  # logs of successful networks are not needed
        if done(job["name"]):
            (HERE / "audit" / f"{job['name']}.log").unlink(missing_ok=True)
    if audit["failures"]:
        raise SystemExit("Failed networks: " + ", ".join(audit["failures"]))
    print("stripe complete", flush=True)


if __name__ == "__main__":
    main()
