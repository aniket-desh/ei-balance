"""Run fresh, matched readouts and full-network measurements on the same recordings."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE.parent / "runtime"), str(HERE.parent / "readout_pilot")]
import engine_v4 as engine
import numpy as np
from measure import measure_case
from summarize import edge_counts
from native_action import build
from scipy.special import ndtr


def cases(protocol):
    base = json.loads((HERE.parent / "dense_sweeps/jobs.json").read_text())
    for seed in range(protocol["seed_first"], protocol["seed_last"] + 1):
        for f in protocol["fractions"]:
            j = next(x for x in base if x["experiment"] == "composition" and
                     x["config"]["n"] == protocol["network_size"] and x["config"]["fraction_e"] == f
                     and x["config"]["seed"] == protocol["reference_config_seed"])
            yield dict(name=f"confirm_f{f:g}_N2000_seed{seed}", config=dict(j["config"],seed=seed),
                       mean_input=j["mean_input"])


def subset_ids(job, p):
    c = job["config"]
    n, seed = c["n"], c["seed"]
    ne = round(n*c["fraction_e"])
    subsets = []
    for a,(offset,count) in enumerate(((0,ne),(ne,n-ne))):
        group = []
        for draw in range(p["draws_per_class"]):
            rng = np.random.default_rng(np.random.SeedSequence([p["subset_seed"],seed,a,draw]))
            priority = rng.random(n)[:count]
            group.append(np.sort(np.argsort(priority,kind="stable")[:p["readout_size"]] + offset))
        subsets.append(group)
    return np.asarray(subsets)


def changing_cells(trajectory, start, stop):
    """A rate varies iff its extrema differ after applying the monotone transfer."""
    low = np.array(trajectory[start], dtype=float)
    high = low.copy()
    for offset in range(start, stop, 2000):
        block = trajectory[offset:min(offset+2000,stop)]
        np.minimum(low, block.min(0), out=low)
        np.maximum(high, block.max(0), out=high)
    return ndtr(low) != ndtr(high)


def measure(job, protocol, library):
    import measure as pilot
    name, c = job["name"], job["config"]
    with TemporaryDirectory(dir=HERE/"tmp",prefix=name+"-") as tmp:
        scratch = Path(tmp)
        for sub in ("tmp", "runs", "measurements"):
            (scratch/sub).mkdir()
        (scratch/"protocol.json").write_bytes((HERE/"protocol.json").read_bytes())
        engine.HERE = pilot.HERE = scratch
        engine.run_one(dict(job, native_library=str(library), blocks=[protocol["block_duration"]],
                            prefixes=protocol["prefixes"], retain_temporary_trajectory=True))
        full = json.loads((scratch/"runs"/(name+".json")).read_text())
        ids = subset_ids(job,protocol)
        path = scratch/"tmp"/(name+"_trajectory.npy")
        trajectory = np.load(path, mmap_mode="r")
        half = len(trajectory)//2
        change = {duration:changing_cells(trajectory,0,round(duration/c["sample_dt"])) for duration in protocol["prefixes"]}
        late = changing_cells(trajectory,half,len(trajectory))
        del trajectory
        measure_case(dict(job,subsets=ids), path)
        draw = json.loads((scratch/"measurements"/(name+".json")).read_text())
        assert full["connectivity_sha256"] == draw["connectivity_sha256"] and full["status"] == "complete"
        est, = full["estimators"]
        with np.load(scratch/"runs"/Path(est["neuron_statistics_file"]).name) as all_cells, np.load(scratch/"measurements"/draw["block_statistics"]) as subset:
            k = ids.shape[-1]
            diag = np.cumsum(np.arange(1,k+1))-1
            muerr = np.max(abs(all_cells["block_mean_rate"][:,ids].transpose(1,2,0,3) - subset["block_mean_rate"]))
            varerr = np.max(abs(all_cells["block_variance_rate"][:,ids].transpose(1,2,0,3) - subset["block_cov_lower"][...,diag]))
            np.testing.assert_allclose([muerr,varerr],0,atol=1e-10)
        n=c["n"]
        counts=np.array([round(n*c["fraction_e"]),n-round(n*c["fraction_e"])])
        windows={}
        for prefix in est["prefixes"]:
            for item in prefix["gaps"]:
                key = f"T{int(prefix['duration'])}_gap{item['empty_blocks']}"
                energy = np.diag(item["population_energy"])
                pair = n**2*(energy-np.array(item["diagonal_population_energy"]))/(counts*(counts-1))
                trace_over_n = counts/n*np.array(item["nu_by_population"])
                varying = change[prefix["duration"]]
                activity = [bool(mask.any()) for mask in np.split(varying,[counts[0]])]
                group_pr = [float(trace_over_n[a]**2/energy[a]) if activity[a] and energy[a]>0 else None for a in range(2)]
                readouts = []
                for a,group in enumerate(draw["readouts"]):
                    values = []
                    for cells,d in zip(ids[a],group):
                        v = {x:d[key][x] for x in ("PR","trace","frobenius_energy","diagonal_energy")}
                        if not varying[cells].any() or v["frobenius_energy"]<=0:
                            v["PR"] = None
                        values.append(v)
                    readouts.append(values)
                windows[key] = dict(full_D_over_N=group_pr,full_joint_D_over_N=item["d"] if any(activity) and item["energy"]>0 else None,
                    varying_rate_counts=[int(g.sum()) for g in np.split(varying,[counts[0]])],
                    full_pair_energy=pair.tolist(), full_energy=item["population_energy"],
                    full_diag=item["diagonal_population_energy"],full_nu=item["nu_by_population"],
                    subsets=readouts)
        check = {"block_mean_max_error":float(muerr),"block_variance_max_error":float(varerr),"dense_tile_error":est["dense_formula_error"]}
        output = dict(job=job, windows=windows,checks=check,counts=edge_counts(job, full["connectivity_sha256"]),
                      connectivity_sha256=full["connectivity_sha256"], engine_sha256=full["engine_source_sha256"],
                      binary_sha256=full["native_library_sha256"], protocol_sha256=draw["protocol_sha256"],
                      indices_sha256=hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
                      measurement_source_sha256=draw["measurement_source_sha256"],
                      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      state=full["state_description"],last_half_varying_rate_counts=[int(g.sum()) for g in np.split(late,[counts[0]])],
                      drift=full["final_drift_rms"],started_unix=full["started_unix"],
                      simulation_seconds=full["simulation_seconds"],estimated_peak_rss_bytes=full["max_rss_bytes"])
        encoded=json.dumps(output,separators=(",",":"),allow_nan=False)
        target=HERE/"records"/(name+".json")
        temp=target.with_suffix(".partial")
        temp.write_text(encoded+"\n");temp.replace(target)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part",default="1/1",help="1-based stripe of complete matched seed triplets")
    ap.add_argument("--workers",type=int,default=1)
    ap.add_argument("--one")
    ap.add_argument("--library")
    args=ap.parse_args()
    p=json.loads((HERE/"protocol.json").read_text())
    jobs=list(cases(p))
    for folder in ("tmp","records"):
        (HERE/folder).mkdir(exist_ok=True)
    if args.one:
        measure(next(j for j in jobs if j["name"]==args.one),p,args.library)
        return
    stripe,count=map(int,args.part.split("/"))
    if not 1 <= stripe <= count or args.workers < 1: ap.error("invalid stripe or worker count")
    disk_per_case = 2*1024**3
    if shutil.disk_usage(HERE).free < args.workers*disk_per_case: raise RuntimeError("Insufficient scratch disk")
    import psutil
    if psutil.virtual_memory().available < args.workers*1.8*1024**3: raise RuntimeError("Insufficient available RAM")
    library,_=build(HERE/"tmp"/f"kernel_{stripe}")
    selected=[j for j in jobs if (j["config"]["seed"]-p["seed_first"])%count==stripe-1]
    fingerprint=hashlib.sha256((HERE/"protocol.json").read_bytes()).hexdigest()
    def execute(j):
        path=HERE/"records"/(j["name"]+".json")
        if path.exists():
            saved=json.loads(path.read_text())
            if saved["job"]!=j or saved["protocol_sha256"]!=fingerprint: raise ValueError(path)
        else:
            subprocess.run([sys.executable,__file__,"--one",j["name"],"--library",str(library)],check=True)
        print(j["name"],flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        list(workers.map(execute,selected))

if __name__=="__main__":
    main()
