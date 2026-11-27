"""Collect completed networks into networks.csv (one row per network) and coverage.json.

    python experiments/composition_optimum/validation/summarize.py [--manifest jobs.json] [--out networks.csv]

Columns per network: identity (rate pair, target rates, N, f, seed), state, measured population
mean rates, and for each estimator window W in {T8000 (primary: 8,000 leak times, window 500,
no gap), T4000 (first 4,000), gap1 (8,000 with one empty block between windows)}:
E_EE, E_EI, E_II (normalized by total N, E-I once), nu_E, nu_I and d = nu^2/(E_EE + 2E_EI + E_II).
Primary-window columns carry no suffix. Parameter-only predictions are added as th_*.

`fluctuating` is 0 when the final drift |dx/dt| is below 1e-10 (the network ended at a fixed
point); `state` separates fixed points for the whole recording from networks that settled
during it, as in the campaign. Those networks are kept in the table and excluded from averages.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WINDOWS = {(8000., 0): "", (4000., 0): "_T4000", (8000., 1): "_gap1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="jobs.json")
    ap.add_argument("--out", default="networks.csv")
    args = ap.parse_args()
    pred = json.loads((HERE / "predictions.json").read_text())
    jobs = json.loads((HERE / args.manifest).read_text())
    rows, coverage = [], {}
    for job in jobs:
        cell, c = job["cell"], job["config"]
        key = f"{cell['rates']}_f{cell['fraction_e']:g}_N{c['n']}"
        cov = coverage.setdefault(key, dict(planned=0, complete=0, fluctuating=0))
        cov["planned"] += 1
        path = HERE / "records" / f"{job['name']}.json.gz"
        if not path.exists() or not (HERE / "arrays" / f"{job['name']}.npz").exists():
            continue
        with gzip.open(path, "rt") as handle:
            rec = json.load(handle)
        if rec["job"]["config"] != c or rec["job"]["mean_input"] != job["mean_input"] or rec["status"] != "complete":
            raise ValueError(f"{job['name']} does not match the manifest")
        cov["complete"] += 1
        n, f = c["n"], cell["fraction_e"]
        counts = np.array([round(n * f), n - round(n * f)])
        assert counts[0] / n == f
        est = next(e for e in rec["estimators"] if e["block_duration"] == 500)
        row = dict(name=job["name"], rates=cell["rates"], m_E=cell["mean_rates"][0], m_I=cell["mean_rates"][1], n=n,
                   fraction_e=f, seed=c["seed"], seed_index=job["seed_index"])
        settled = rec["final_drift_rms"] < 1e-10
        for prefix in est["prefixes"]:
            for item in prefix["gaps"]:
                tag = WINDOWS.get((prefix["duration"], item["empty_blocks"]))
                if tag is None:
                    continue
                e, nu = np.array(item["population_energy"]), np.array(item["nu_by_population"])
                assert abs(e[0, 1] - e[1, 0]) <= 1e-12 * abs(e[0, 1]) + 1e-300
                total = float(e.sum())
                row.update({f"E_EE{tag}": e[0, 0], f"E_EI{tag}": e[0, 1], f"E_II{tag}": e[1, 1], f"nu_E{tag}": nu[0],
                            f"nu_I{tag}": nu[1], f"energy{tag}": total, f"d{tag}": item["d"]})
                assert abs(item["d"] - (counts @ nu / n)**2 / total) <= 1e-12 * abs(item["d"]) + 1e-300
        if "d" in row:
            row["state"] = "fluctuating" if not settled else "fixed_point" if row["nu_E"] + row["nu_I"] < 1e-8 else "settled_during_recording"
        row["fluctuating"] = int(not settled)
        cov["fluctuating"] += int(not settled)
        local = est["population_local_stats"]
        row.update(rate_E=local[0]["mean_rate"], rate_I=local[1]["mean_rate"], h_E=local[0]["h"], h_I=local[1]["h"],
                   drift=rec["final_drift_rms"], sim_seconds=rec["simulation_seconds"], elapsed_seconds=rec["elapsed_seconds"],
                   host=rec["host"]["node"], numpy=rec["host"]["numpy"])
        th = pred["rate_pairs"][cell["rates"]]
        thc = th["cells"].get(f"{f:g}")
        if thc:
            row.update({f"th_{k}": v for k, v in thc.items()})
        row.update(th_nu_E=th["v_E"], th_nu_I=th["v_I"], th_h_E=th["mean_inputs"][0], th_h_I=th["mean_inputs"][1])
        rows.append(row)
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(HERE / args.out, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if args.manifest == "jobs.json":
        (HERE / "coverage.json").write_text(json.dumps(coverage, indent=1) + "\n")
    print(f"{len(rows)} of {len(jobs)} networks summarized; not fluctuating: {sum(1 - r['fluctuating'] for r in rows)}")


if __name__ == "__main__":
    main()
