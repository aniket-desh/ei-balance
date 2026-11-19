"""Independently check sampled spectral and projection measurements from the saved arrays."""

from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
from scipy.special import ndtr
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent


def direct_rho(left, right):
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def main():
    result = json.loads((HERE / "summary.json").read_text())
    max_error = {
        "rho": 0.0,
        "trace": 0.0,
        "energy": 0.0,
        "complement_pr_ratio": 0.0,
        "null_rate_first": 0.0,
    }
    quiet = []
    checks = []
    contrasts = []
    for row in result["network_results"]:
        name = row["name"]
        c = json.loads((HERE / "measurements" / (name + ".json")).read_text())
        x = np.load(HERE / "tmp" / (name + "_trajectory.npy"), mmap_mode="r", allow_pickle=False)
        with (HERE / "tmp" / (name + "_trajectory.npy")).open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == c["trajectory_sha256"]
        fluct = all(
            np.mean(np.var(window, axis=0, dtype=float)) > 1e-10 for window in np.array_split(x, 2)
        )
        assert fluct == row["fluctuating"]
        if not row["fluctuating"]:
            quiet_row = {
                "name": name,
                "max_late_mean_input_variance": float(
                    np.var(x[len(x) // 2 :], axis=0, dtype=float).max()
                ),
                "final_drift_rms": c["final_drift_rms"],
            }
            if c["roots"]:
                fixed = np.load(HERE / c["roots"][0]["mode_file"])["root"]
                quiet_row["first_root_distance_from_final_sample_inf"] = float(
                    np.max(abs(x[-1] - fixed))
                )
                quiet_row["first_root_max_real_coupling_eigenvalue"] = row["first_root_max_real"]
            quiet.append(quiet_row)
            continue
        if not row["verified_roots"]:
            continue
        r = c["roots"][0]
        z = np.load(HERE / r["mode_file"])
        e, v, Q = z["eigenvalues"], z["input_variance"], z["rate_basis"]
        indices = e.imag >= 0
        rho = direct_rho(e[indices].real, v[0, indices])
        max_error["rho"] = max(max_error["rho"], abs(rho - row["primary"]["rho"]))
        contrasts.append(rho - np.median(r["population_null_rho"]))
        rates = 2 * ndtr(np.array(x, dtype=float))
        rates -= rates.mean(axis=0)
        C = rates.T @ rates / len(rates)
        tr = np.trace(C)
        energy = np.square(C).sum()

        def measure(projector_basis):
            P = projector_basis @ projector_basis.T
            block = P @ C @ P
            residual = C - P @ C - C @ P + block
            return [
                np.trace(block) / tr,
                np.square(block).sum() / energy,
                np.trace(residual) ** 2 / np.square(residual).sum() / (tr * tr / energy),
            ]

        ref = measure(Q)
        for k, error in zip(
            ("trace", "energy", "complement_pr_ratio"),
            np.abs(np.asarray(ref) - r["rate_projection"][0]),
        ):
            max_error[k] = max(max_error[k], float(error))
        # Rebuild the first gain-matched permutation independently, then project explicitly.
        gain = np.sqrt(2 / np.pi) * np.exp(-0.5 * z["root"] ** 2)
        n = row["N"]
        pop = (np.arange(round(0.8 * n)), np.arange(round(0.8 * n), n))
        rng = np.random.default_rng(np.random.SeedSequence(row["seed"] + 813000).spawn(2)[1])
        permutation = np.arange(n)
        for group in pop:
            for ids in np.array_split(group[np.argsort(gain[group], kind="stable")], 10):
                permutation[ids] = rng.permutation(ids)
        max_error["null_rate_first"] = max(
            max_error["null_rate_first"],
            max(abs(np.asarray(measure(Q[permutation])) - r["gain_decile_null_rate"][0])),
        )
        checks.append(name)
    pos = sum(v > 0 for v in contrasts)
    n = sum(v != 0 for v in contrasts)
    p = sum(math.comb(n, j) for j in range(pos, n + 1)) / 2**n
    assert pos == result["all"]["positive"] and abs(result["all"]["one_sided_sign_p"] - p) < 1e-15
    assert max(max_error.values()) < 1e-9
    science_files = [
        HERE / s
        for s in (
            "PROTOCOL.md",
            "run_networks.py",
            "measure.py",
            "summarize.py",
            "check.py",
            "plot.py",
        )
    ]
    science_files.extend(
        [HERE.parent / "spectral_pilot" / s for s in ("simulate.py", "analyze.py")]
    )
    science_files.extend(
        [
            HERE.parent / "runtime" / s
            for s in (
                "mode_variance.py",
                "engine_v4.py",
                "runs/composition_f80_N2000_seed3002.json",
            )
        ]
    )
    science_files.extend(
        [HERE.parents[1] / "src" / "eicavity" / s for s in ("network.py", "single_site.py")]
    )
    digests = {
        str(p.relative_to(HERE.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in science_files
    }
    saved = {
        "eligible_primary_checked": checks,
        "quiet_draws": quiet,
        "max_abs_errors": max_error,
        "sign_probability_direct": p,
        "source_sha256": digests,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    (HERE / "validation.json").write_text(json.dumps(saved, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
