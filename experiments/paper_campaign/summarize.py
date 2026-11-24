"""Collect every completed network into flat tables for plotting.

    python experiments/paper_campaign/summarize.py

Writes
  networks.csv   one row per network at the primary window (T = 8000, B = 500, gap 0),
                 with population moments, energies, dimensions, local statistics,
                 sensitivities, readout PRs, Lyapunov exponent and the matching prediction
  windows.csv    one row per network x block length x duration x gap (all estimator windows)
  theory.csv     one row per condition (parameter-only predictions)
  coverage.json  planned versus completed networks per family and cell

`fluctuating` is 0 for a network whose final drift |dx/dt| is below 1e-10, that is, one
that ended the recording at a stable fixed point. `state` separates networks at a fixed
point for the whole recording (`fixed_point`, rate variance below 1e-8) from those that
settled partway through (`settled_during_recording`). Neither has a stationary PR; the
figures and report count them separately instead of averaging them in.

Normalizations follow the engine: block energies E_ab = ||C_ab||_F^2 / N, total
E = E_EE + 2 E_EI + E_II, nu_a = tr C_aa / N_a, d = (sum_a f_a nu_a)^2 / E.
P_ab is N times the mean squared covariance over distinct pairs (i in a, j in b).
"""
import csv
import gzip
import json
from pathlib import Path

import numpy as np

import theory

HERE = Path(__file__).resolve().parent


def window_row(item, n, counts):
    """Derived quantities from one estimator window of the engine record."""
    e = np.array(item["population_energy"])
    diag = np.array(item["diagonal_population_energy"])
    nu = np.array(item["nu_by_population"])
    f = counts / n
    total = float(e.sum())
    row = dict(nu_E=nu[0], nu_I=nu[1], nu=item["nu"], E_EE=e[0, 0], E_EI=e[0, 1], E_II=e[1, 1], energy=total,
               d=item["d"], diag_E=diag[0], diag_I=diag[1],
               dE=f[0] * nu[0]**2 / e[0, 0], dI=f[1] * nu[1]**2 / e[1, 1],   # D_a / N_a
               P_EE=n**2 * (e[0, 0] - diag[0]) / (counts[0] * (counts[0] - 1)),
               P_II=n**2 * (e[1, 1] - diag[1]) / (counts[1] * (counts[1] - 1)),
               P_EI=n**2 * e[0, 1] / (counts[0] * counts[1]),
               overlap=e[0, 1] / np.sqrt(e[0, 0] * e[1, 1]) if e[0, 0] > 0 and e[1, 1] > 0 else np.nan,
               classic_energy=item["classic_energy"], classic_d=item["nu"]**2 / item["classic_energy"],
               coherent_fraction=item["coherent_projection_fraction"])
    row["dE_total"] = f[0] * row["dE"]       # D_E / N
    row["het_factor"] = item["nu"]**2 / (diag[0] + diag[1])          # nu^2 / E_diag
    row["pair_factor"] = (diag[0] + diag[1]) / total                 # E_diag / E
    for a, ing in enumerate(item["ingredient_cross_block"]):
        s = "EI"[a]
        row[f"Delta_{s}"] = ing["Delta_cross_block_inputmeans"]
        row[f"A_{s}"] = ing["A_cross_block_timeaveraged_gain"]
        row[f"q_{s}"] = ing["q_mean_free_input"]
    return row


def theory_row(cell):
    """Prediction for one condition, in the same normalization as window_row."""
    f = np.array([cell["fraction_e"], 1 - cell["fraction_e"]])
    if cell.get("bernoulli"):
        c = theory.composition(cell["fraction_e"])
        base, _ = theory.solve(1., cell["rate"], 1.)
        block, pair = np.array(c["block_energy"]), np.array(c["pair_covariance_energy"])
        means, seconds = np.array(c["mean_variances"]), np.array(c["variance_seconds"])
        d, dw = c["dimension_fraction"], c["within_population_dimension_fractions"]
    else:
        base, _ = theory.solve(cell["scale"], cell["rate"], float(cell["chi"]))
        block, pair = np.array(base["block_energy"]), np.array(base["pair_energy"])
        means, seconds = np.array(base["mean_variances"]), np.array(base["variance_seconds"])
        d, dw = base["d"], base["d_within"]
    diag = f * seconds
    return dict(nu_E=means[0], nu_I=means[1], nu=float(f @ means), E_EE=block[0, 0], E_EI=block[0, 1],
                E_II=block[1, 1], energy=float(block.sum()), d=d, dE=dw[0], dI=dw[1], dE_total=f[0] * dw[0],
                diag_E=diag[0], diag_I=diag[1], P_EE=pair[0, 0], P_II=pair[1, 1], P_EI=pair[0, 1],
                overlap=block[0, 1] / np.sqrt(block[0, 0] * block[1, 1]),
                het_factor=float(f @ means)**2 / diag.sum(), pair_factor=diag.sum() / block.sum(),
                Delta_E=base["static_variances"][0], Delta_I=base["static_variances"][1],
                q_E=base["dynamic_variances"][0], q_I=base["dynamic_variances"][1],
                A_E=base["alpha_second"][0], A_I=base["alpha_second"][1],
                h_E=base["mean_inputs"][0], h_I=base["mean_inputs"][1], response_radius=base["response_radius"])


def cell_key(cell):
    return (cell.get("chi"), cell["fraction_e"], cell["scale"], cell["rate"], bool(cell.get("bernoulli")))


def main():
    jobs = json.loads((HERE / "jobs.json").read_text())
    if (HERE / "jobs_extension.json").exists():
        jobs += json.loads((HERE / "jobs_extension.json").read_text())
    networks, windows, theories, coverage = [], [], {}, {}
    for job in jobs:
        cell, c = job["cell"], job["config"]
        cov = coverage.setdefault(job["family"], {}).setdefault(job["name"].rsplit("_seed", 1)[0], [0, 0])
        cov[1] += 1
        path = HERE / "records" / f"{job['name']}.json.gz"
        if not path.exists() or not (HERE / "arrays" / f"{job['name']}.npz").exists():
            continue
        with gzip.open(path, "rt") as handle:
            record = json.load(handle)
        if record["job"]["config"] != c or record["status"] != "complete":
            raise ValueError(f"{job['name']} does not match the manifest")
        cov[0] += 1
        n = c["n"]
        counts = np.array([round(n * c["fraction_e"]), n - round(n * c["fraction_e"])])
        ident = dict(name=job["name"], family=job["family"], n=n, seed=c["seed"], seed_index=job["seed_index"],
                     chi=cell.get("chi"), fraction_e=cell["fraction_e"], scale=cell["scale"], rate=cell["rate"],
                     bernoulli=int(bool(cell.get("bernoulli"))), dt=c["dt"], duration=c["duration"])
        key = cell_key(cell)
        if key not in theories:
            theories[key] = dict(chi=cell.get("chi"), fraction_e=cell["fraction_e"], scale=cell["scale"],
                                 rate=cell["rate"], bernoulli=int(bool(cell.get("bernoulli"))), **theory_row(cell))
        primary = None
        for est in record["estimators"]:
            for prefix in est["prefixes"]:
                for item in prefix["gaps"]:
                    row = window_row(item, n, counts)
                    meta = dict(block=est["block_duration"], window_duration=prefix["duration"], gap=item["empty_blocks"])
                    windows.append({**ident, **meta, **row})
                    if est["block_duration"] == 500:
                        if (prefix["duration"], item["empty_blocks"]) == (8000., 0):
                            primary = row
                        tag = {(4000., 0): "T4000", (8000., 1): "gap1"}.get((prefix["duration"], item["empty_blocks"]))
                        if tag:
                            ident[f"d_{tag}"] = row["d"]
        est = next(e for e in record["estimators"] if e["block_duration"] == 500)
        local = {}
        for a, stats in enumerate(est["population_local_stats"]):
            s = "EI"[a]
            local.update({f"rate_{s}": stats["mean_rate"], f"h_{s}": stats["h"], f"v2_{s}": stats["mean_v_squared"],
                          f"A4_{s}": stats["A4_timeaveraged_gain_fourth"],
                          f"qvar_{s}": stats["variance_temporal_q_across_neurons"]})
        extra = record.get("extras", {})
        # A trajectory at a fixed point has no stationary fluctuations and no defined PR. The final
        # drift |dx/dt| is ~1e-15 there and O(0.1) on fluctuating trajectories. A network that settles
        # partway through the recording still shows variance from before it settled, so the drift, not
        # the recorded variance, decides the state.
        settled = record["final_drift_rms"] < 1e-10
        state = "fluctuating" if not settled else "fixed_point" if primary["nu"] < 1e-8 else "settled_during_recording"
        out = {**ident, "fluctuating": int(not settled), "state": state, **primary, **local, "drift": record["final_drift_rms"],
               "lyapunov": extra.get("tangent", {}).get("largest_finite_time_exponent"),
               "sim_seconds": record["simulation_seconds"], "host": record["host"]["node"]}
        for label, draws in extra.get("readouts", {}).items():
            for w in ("T8000_gap0", "T4000_gap0", "T8000_gap1"):
                values = [x[w]["PR"] for x in draws if w in x and x[w]["PR"] is not None and x[w]["frobenius_energy"] > 0]
                suffix = "" if w == "T8000_gap0" else "_" + w
                out[f"PR_{label}{suffix}"] = float(np.mean(values)) if len(values) == len(draws) else None
            # Ratio of subset-averaged moments, the quantity the readout theory predicts.
            tr = np.array([x["T8000_gap0"]["trace"] for x in draws])
            en = np.array([x["T8000_gap0"]["frobenius_energy"] for x in draws])
            out[f"R_{label}"] = float(np.mean(tr**2) / np.mean(en))
        for k, v in theories[key].items():
            if k not in ("chi", "fraction_e", "scale", "rate", "bernoulli"):
                out[f"th_{k}"] = v
        networks.append(out)

    def write(path, rows):
        if not rows:
            return
        fields = []
        for row in rows:
            fields += [k for k in row if k not in fields]
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write(HERE / "networks.csv", networks)
    write(HERE / "windows.csv", windows)
    write(HERE / "theory.csv", list(theories.values()))
    (HERE / "coverage.json").write_text(json.dumps(coverage, indent=1) + "\n")
    done = sum(v[0] for fam in coverage.values() for v in fam.values())
    print(f"{done} of {len(jobs)} networks summarized")
    for family, cells in coverage.items():
        print(f"  {family:12s} {sum(v[0] for v in cells.values()):4d} / {sum(v[1] for v in cells.values()):4d}")


if __name__ == "__main__":
    main()
