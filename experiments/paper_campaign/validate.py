"""Checks behind the campaign's measurements; writes validation.json.

    python experiments/paper_campaign/validate.py        (about two minutes, needs a C++ compiler)

1. Replay one archived dense-sweep network through run.run_case and compare every estimator
   window with the archived record. On the machine and backend that produced the archive
   (the Mac, native action) the comparison is exact; elsewhere a chaotic trajectory differs.
2. A readout made of an entire population must return the engine's block trace and energy.
3. The cross-fitted lag-map code against a brute-force loop over pairs and lags.
4. theory.py against the archived predictions and the manuscript's averaging controls.
"""
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "runtime")]
import run        # before extras, which puts readout_pilot (with its own run.py) first on sys.path
import extras
import theory
from native_action import build


def main():
    for folder in ("tmp", "records", "arrays", "audit"):
        (HERE / folder).mkdir(exist_ok=True)
    library, _ = build(HERE / "tmp" / "kernel_validate")
    out = {}

    # 1. replay of an archived network
    old_job = next(j for j in json.loads((HERE.parent / "dense_sweeps/jobs.json").read_text())
                   if j["name"] == "dense_degree_0_N2000_seed5101")
    archived = json.loads((HERE.parent / "dense_sweeps/runs/dense_degree_0_N2000_seed5101.json").read_text())
    job = dict(name="check_replay", family="check", cell={}, seed_index=0, config=old_job["config"],
               mean_input=old_job["mean_input"], blocks=[500.], prefixes=[4000., 8000.],
               extras=dict(readout_sizes=[100], lag=True, spectrum=False, tangent=False))
    run.run_case(job, library)
    with gzip.open(HERE / "records/check_replay.json.gz", "rt") as handle:
        new = json.load(handle)
    diffs = []
    for pn, po in zip(new["estimators"][0]["prefixes"], archived["estimators"][0]["prefixes"]):
        for gn, go in zip(pn["gaps"], po["gaps"]):
            diffs.append(abs(gn["d"] - go["d"]))
            diffs.append(float(np.max(np.abs(np.array(gn["population_energy"]) - np.array(go["population_energy"])))))
    out["replay_archived_network"] = dict(name=old_job["name"], same_connectivity=new["connectivity_sha256"] == archived["connectivity_sha256"],
                                          max_abs_difference_d_and_block_energy=max(diffs), host=new["host"],
                                          archived_backend=archived["action_backend"], new_backend=new["action_backend"])

    # 2. whole-population readout: N = 500 at 80:20 has exactly 100 I cells
    small = dict(job, name="check_small", config=dict(old_job["config"], n=500, duration=2000., degree_dispersion=1, seed=7),
                 prefixes=[2000.], extras=dict(readout_sizes=[100], lag=False, spectrum=False, tangent=False))
    run.run_case(small, library)
    with gzip.open(HERE / "records/check_small.json.gz", "rt") as handle:
        rec = json.load(handle)
    g = rec["estimators"][0]["prefixes"][-1]["gaps"][0]
    energy, nu = np.array(g["population_energy"]), np.array(g["nu_by_population"])
    readout = rec["extras"]["readouts"]["I100"][0]["T2000_gap0"]
    out["whole_population_readout"] = dict(trace_error=abs(readout["trace"] - 100 * nu[1]),
                                           energy_error=abs(readout["frobenius_energy"] - 500 * energy[1, 1]),
                                           trace=readout["trace"], energy=readout["frobenius_energy"])

    # 3. lag estimator against brute force on a small synthetic trajectory
    rng = np.random.default_rng(0)
    T, n = 400, 12
    x = (np.cumsum(rng.normal(size=(T, n)), 0) * .05 + rng.normal(size=(T, n)) * .3).astype(np.float32)
    labels = np.r_[np.zeros(8, int), np.ones(4, int)]
    extras.LAG_UNITS = 3
    maps = extras.lag_maps(x, labels, dict(n=n, sample_dt=.25))
    extras.LAG_UNITS = 24
    rates = 2 * ndtr(x.astype(float))
    half = T // 2
    H = [rates[:half] - rates[:half].mean(0), rates[half:] - rates[half:].mean(0)]

    def cov(h, i, j, tau):
        L = abs(tau) * 4
        a, b = (i, j) if tau >= 0 else (j, i)
        return (H[h][L:, a] * H[h][:half - L, b]).sum() / (half - L)

    worst = 0.
    for name, ia, ib in (("EE", range(8), range(8)), ("EI", range(8), range(8, 12)), ("II", range(8, 12), range(8, 12))):
        for t1 in range(-3, 4):
            for t2 in range(-3, 4):
                values = [.5 * (cov(0, i, j, t1) * cov(1, i, j, t2) + cov(1, i, j, t1) * cov(0, i, j, t2))
                          for i in ia for j in ib if i != j]
                worst = max(worst, abs(n * np.mean(values) - maps["psi_" + name][t1 + 3, t2 + 3]))
    out["lag_estimator_vs_brute_force"] = dict(max_abs_error=float(worst), map_scale=float(np.abs(maps["psi_EE"]).max()),
                                               note="maps are stored in float32")

    # 4. theory against the archive and the manuscript
    curve = {r["chi"]: r for r in json.loads((HERE.parent / "degree/theory_curve.json").read_text())["rows"]}
    errs = {}
    for chi in (0., .5, 1., 2.):
        s, _ = theory.solve(1., .3, chi)
        errs[f"chi={chi:g}"] = dict(d=abs(s["d"] - curve[chi]["prediction"]["dimension_fraction"]),
                                    mean_inputs=float(np.max(np.abs(np.array(s["mean_inputs"]) - np.array(curve[chi]["mean_inputs"])))))
    controls = theory.averaging_controls()
    lag = theory.lag_map()
    s1, _ = theory.solve(1., .3, 1.)
    out["theory"] = dict(degree_archive_abs_errors=errs,
                         readout_ratio_0p80_and_0p88=[theory.readout_ratio(.8, 100), theory.readout_ratio(.88, 100)],
                         readout_theory_table=[[28.438, 30.243], [28.900, 29.076]],
                         averaging_controls=dict(d_spectra=controls["spectra"]["d"], overlap_spectra=controls["spectra"]["overlap"],
                                                 overlap_gains=controls["gains"]["overlap"],
                                                 manuscript=dict(d_spectra=.07645, overlap_spectra=.78766, overlap_gains=.35138)),
                         lag_map_zero_lag_minus_pair_energy=float(np.max(np.abs(lag["psi"][:, :, 24, 24] - np.array(s1["pair_energy"])))))

    for name in ("check_replay", "check_small"):
        (HERE / "records" / f"{name}.json.gz").unlink()
        (HERE / "arrays" / f"{name}.npz").unlink()
    (HERE / "validation.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
