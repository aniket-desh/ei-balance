"""Generate the four retained spectral pilot recordings using the campaign integrator."""

import json
import os
from pathlib import Path
import resource
import signal
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "runtime"))
import engine_v4 as engine


def set_memory_limit():
    limit = int(1.5 * 2**30)
    def check_rss(signum, frame):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if rss > limit:
            raise MemoryError("Spectral pilot RSS exceeded 1.5 GiB")
    signal.signal(signal.SIGALRM, check_rss)
    signal.setitimer(signal.ITIMER_REAL, 0.25, 0.25)
    return limit


def main():
    limit = set_memory_limit()
    reference_path = HERE.parent / "runtime/runs/composition_f80_N2000_seed3002.json"
    template = json.loads(reference_path.read_text())["job"]
    engine.HERE = HERE
    (HERE / "tmp").mkdir(exist_ok=True)
    (HERE / "runs").mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        for n in (250, 500):
            for seed in (7101, 7102):
                name = f"spectral_N{n}_seed{seed}"
                cfg = {**template["config"], "n": n, "seed": seed, "duration": 2000.}
                job = {"name": name, "experiment": "spectral_pilot", "config": cfg,
                       "mean_input": template["mean_input"], "blocks": [250.],
                       "prefixes": [2000.], "retain_temporary_trajectory": True}
                record = HERE / "runs" / (name + ".json")
                trace = HERE / "tmp" / (name + "_trajectory.npy")
                if record.exists() and trace.exists():
                    saved = json.loads(record.read_text())
                    if saved["status"] == "complete" and saved["job"] == job:
                        continue
                    if saved["job"] != job:
                        raise ValueError(f"Existing run uses different settings: {name}")
                out = engine.run_one(job)
                print(name, out, flush=True)
                if out["max_rss_bytes"] > limit:
                    raise MemoryError(f"RSS exceeded pilot budget in {name}")


if __name__ == "__main__":
    main()
