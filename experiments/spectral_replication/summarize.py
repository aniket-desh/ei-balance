"""Summarize the independent-network contrast, controls and root sensitivity."""

from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent


def root_stats(root):
    trace, energy, pr = root["rate_projection"][0]
    pop_trace, pop_energy, pop_pr = np.median(root["population_null_rate"], axis=0)
    gain_trace, gain_energy, gain_pr = np.median(root["gain_decile_null_rate"], axis=0)
    null_rho = np.median(root["population_null_rho"])
    rho = root["rho_input"][0]
    return {
        "rho": rho,
        "shuffle_rho": null_rho,
        "excess_rho": rho - null_rho,
        "rho_abs_real_above_1e_minus8": root["resolved_real_parts_rho"],
        "rho_first_half": root["rho_input"][1],
        "rho_second_half": root["rho_input"][2],
        "half_projection_rank_agreement": root["half_rank_agreement"],
        "subspace_trace": trace,
        "subspace_energy": energy,
        "residual_pr_ratio": pr,
        "pop_null_trace": pop_trace,
        "pop_null_energy": pop_energy,
        "pop_null_pr_ratio": pop_pr,
        "gain_null_trace": gain_trace,
        "gain_null_energy": gain_energy,
        "gain_null_pr_ratio": gain_pr,
        "excess_trace_gain": trace - gain_trace,
        "excess_energy_gain": energy - gain_energy,
        "excess_pr_ratio_gain": pr - gain_pr,
        "energy_first_half": root["rate_projection"][1][1],
        "energy_second_half": root["rate_projection"][2][1],
    }


def distribution(values):
    v = np.asarray(values)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def describe(records):
    eligible = [r for r in records if "primary" in r]
    info = {
        "networks": len(records),
        "fluctuating": sum(r["fluctuating"] for r in records),
        "quiet": sum(not r["fluctuating"] for r in records),
        "root_found": sum(r["verified_roots"] > 0 for r in records),
        "eligible": len(eligible),
        "multiple_roots": sum(r["verified_roots"] > 1 for r in records),
        "multiple_roots_among_eligible": sum(r["verified_roots"] > 1 for r in eligible),
    }
    if not eligible:
        return info
    effects = np.array([r["primary"]["excess_rho"] for r in eligible])
    positive = int(np.sum(effects > 0))
    ties = int(np.sum(effects == 0))
    info.update(
        positive=positive,
        ties=ties,
        one_sided_sign_p=float(
            binomtest(positive, len(effects) - ties, 0.5, alternative="greater").pvalue
        ),
    )
    interval = binomtest(positive, len(effects) - ties).proportion_ci(
        confidence_level=0.95, method="exact"
    )
    info["positive_fraction_exact95_ci"] = [float(interval.low), float(interval.high)]
    info["primary"] = {
        key: distribution([r["primary"][key] for r in eligible]) for key in eligible[0]["primary"]
    }
    info["root_sensitivity"] = {}
    for key in ("rho", "excess_rho", "subspace_energy", "excess_energy_gain", "residual_pr_ratio"):
        ranges = [
            np.ptp([root[key] for root in r["all_root_stats"]])
            for r in eligible
            if r["verified_roots"] > 1
        ]
        info["root_sensitivity"][key] = {
            "range_within_multiple_root_networks": distribution(ranges) if ranges else None,
            "networks_positive_at_every_found_root": sum(
                all(z[key] > 0 for z in r["all_root_stats"]) for r in eligible
            ),
        }
    return info


def main():
    records = []
    sources = [
        HERE / "measure.py",
        HERE / "PROTOCOL.md",
        HERE.parent / "spectral_pilot/analyze.py",
        HERE.parent / "runtime/mode_variance.py",
    ]
    expected_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for p in sorted((HERE / "measurements").glob("*.json")):
        c = json.loads(p.read_text())
        if c["source_hashes"] != expected_hashes:
            raise ValueError(f"Numerical protocol or source differs from {p.name}")
        row = {
            "N": c["N"],
            "seed": c["seed"],
            "name": c["name"],
            "fluctuating": c["state_fluctuating"],
            "mean_input_variance": c["input_mean_temporal_variance"],
            "verified_roots": len(c["roots"]),
            "rate_pr": c["rate_direct_pr"][0],
            "first_root_max_real": c["roots"][0]["max_real_coupling_eigenvalue"]
            if c["roots"]
            else None,
        }
        if c["state_fluctuating"] and c["roots"]:
            row["all_root_stats"] = [root_stats(z) for z in c["roots"]]
            row.update(
                primary=row["all_root_stats"][0],
                rate_rank=c["roots"][0]["rate_rank"],
                root_mode_file=c["roots"][0]["mode_file"],
            )
        records.append(row)
    assert {(r["N"], r["seed"]) for r in records} == {
        (n, s) for n, first in ((250, 73101), (500, 73201)) for s in range(first, first + 12)
    }
    summary = {
        "measurement_order": "full record, first half, second half",
        "rate_projection_order": [
            "fraction of trace",
            "fraction of squared Frobenius energy inside the projected block",
            "complement PR / full PR",
        ],
        "network_results": records,
        "all": describe(records),
        "by_size": {str(n): describe([r for r in records if r["N"] == n]) for n in (250, 500)},
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
