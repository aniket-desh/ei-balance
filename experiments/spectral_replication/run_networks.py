"""Simulate the 24 independent draws in PROTOCOL.md with the archived integrator."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spectral_pilot"))
import simulate as pilot

engine = pilot.engine


def jobs():
    reference = HERE.parent / "runtime/runs/composition_f80_N2000_seed3002.json"
    template = json.loads(reference.read_text())["job"]
    result = []
    for n, first in [(250, 73101), (500, 73201)]:
        for seed in range(first, first + 12):
            cfg = {**template["config"], "n": n, "seed": seed, "duration": 2000.0}
            result.append(
                {
                    "name": f"spectral_N{n}_seed{seed}",
                    "experiment": "spectral_replication",
                    "config": cfg,
                    "mean_input": template["mean_input"],
                    "blocks": [250.0],
                    "prefixes": [2000.0],
                    "retain_temporary_trajectory": True,
                }
            )
    return result


def simulate(job):
    pilot.set_memory_limit()
    engine.HERE = HERE
    path = HERE / "runs" / (job["name"] + ".json")
    if path.is_file():
        record = json.loads(path.read_text())
        if record["job"] != job:
            raise ValueError(f"Saved job differs from protocol: {path}")
        if (
            record["status"] == "complete"
            and (HERE / "tmp" / (job["name"] + "_trajectory.npy")).is_file()
        ):
            return
    with pilot.threadpool_limits(limits=1):
        print(engine.run_one(job), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    (HERE / "tmp").mkdir(exist_ok=True)
    (HERE / "runs").mkdir(exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(simulate, jobs()))
