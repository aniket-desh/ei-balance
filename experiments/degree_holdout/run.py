"""Nine independent N=2000 degree networks for evaluating frozen predictions."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "runtime"))


def jobs():
    earlier = json.loads((HERE.parent / "dense_sweeps/jobs.json").read_text())
    for seed in (15101, 15102, 15103):
        for chi in (0, 1, 2):
            base = next(j for j in earlier if j["experiment"] == "degree"
                        and j["config"]["seed"] == 5101 and j["config"]["n"] == 2000
                        and j["config"]["degree_dispersion"] == chi)
            yield dict(base, config=dict(base["config"], seed=seed), name=f"degree_holdout_{chi}_N2000_seed{seed}",
                       blocks=[500.], shard="local")


if __name__ == "__main__":
    from native_action import build
    import engine_v4 as engine
    engine.HERE = HERE
    for folder in ("tmp", "runs"):
        (HERE / folder).mkdir(exist_ok=True)
    if len(sys.argv) == 3:
        j = next(j for j in jobs() if j["name"] == sys.argv[1])
        print(engine.run_one(dict(j, native_library=sys.argv[2])), flush=True)
    else:
        library, flags = build(HERE / "tmp")
        manifest = list(jobs())
        engine.atomic(HERE/"manifest.json", dict(jobs=manifest, workers=2, build_flags=flags,
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        def run(j):
            record = HERE / "runs" / (j["name"] + ".json")
            capsule = HERE / "runs" / (j["name"] + "_B500_neuron_statistics.npz")
            if not (capsule.exists() and record.exists() and json.loads(record.read_text())["status"] == "complete"):
                subprocess.run([sys.executable, __file__, j["name"], str(library)], check=True)
        with ThreadPoolExecutor(max_workers=2) as workers:
            list(workers.map(run, manifest))
