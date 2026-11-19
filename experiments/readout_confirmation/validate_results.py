"""Verify compact moment identities, the frozen cohort, and independent paired inference."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys

import numpy as np
from scipy.special import stdtrit

from summarize import HERE, WINDOWS, METRICS, CONTRASTS, record_stream, atomic_json, digest


def load(path):
    return json.loads(path.read_text())


THEORY = load(HERE.parent / "readout_theory/predictions.json")


def equal(a, b, context=""):
    if a is None or b is None:
        if a != b:
            raise ValueError(f"Undefined mismatch in {context}: {a}, {b}")
        return
    if (
        not math.isfinite(a)
        or not math.isfinite(b)
        or not math.isclose(a, b, rel_tol=2e-10, abs_tol=2e-10)
    ):
        raise ValueError(f"Numeric mismatch in {context}: {a}, {b}")


def optional_divide(trace, energy, varying):
    return (trace * trace / energy) if varying and energy > 0 else None


def truth(record, protocol, smoke=False):
    """Use the actual compact traces and Frobenius moments as primary numerical inputs."""
    cfg = record["job"]["config"]
    n = cfg["n"]
    f = cfg["fraction_e"]
    seed = cfg["seed"]
    counts = [round(n * f), n - round(n * f)]
    if set(record["windows"]) != set(WINDOWS):
        raise ValueError("Missing or unexpected estimator windows")
    limits = {"block_mean_max_error": 1e-10, "block_variance_max_error": 1e-10, "dense_tile_error": 1e-8}
    if set(record["checks"]) != set(limits):
        raise ValueError("Missing numerical checks")
    for k, value in record["checks"].items():
        if not math.isfinite(value) or not 0 <= value <= limits[k]:
            raise ValueError(f"Spatial numerical check failed: {k} {value}")
    # argpartition implements uniform priorities independently of the production full sort.
    saved_indices = []
    for a, count in enumerate(counts):
        for draw in range(protocol["draws_per_class"]):
            rng = np.random.default_rng(np.random.SeedSequence([protocol["subset_seed"], seed, a, draw]))
            priorities = rng.random(n)[:count]
            k = protocol["readout_size"]
            slots = np.sort(np.argpartition(priorities, k - 1)[:k])
            saved_indices.append(slots + (counts[0] if a == 1 else 0))
    if (
        hashlib.sha256(np.array(saved_indices, dtype="<i8").tobytes()).hexdigest()
        != record["indices_sha256"]
    ):
        raise ValueError(f"Readout identity mismatch for {seed} at {f}")
    block_sizes = {letter: c for letter, c in zip("EI", counts)}
    block_ids = set()
    for c in record["counts"]:
        ab = (c["target"], c["source"])
        block_ids.add(ab)
        M = block_sizes[ab[0]]
        B = block_sizes[ab[1]]
        if c["source_pool_size"] != B or not 0 <= c["total_edges"] <= M * B or c["degree_variance"] < 0:
            raise ValueError(f"Invalid block counts for {seed} at {f}")
        equal(c["mean_degree"], c["total_edges"] / M)
        equal(c["mean_missing_inputs"], B - c["mean_degree"])
    if len(record["counts"]) != 4 or block_ids != {(a, b) for a in "EI" for b in "EI"}:
        raise ValueError("Incomplete E/I degree records")
    outcome = {}
    anchors = {a["fraction_e"]: a for a in THEORY["anchors"]}
    for key, w in record["windows"].items():
        a, b = w["full_energy"]
        for val in [*a, *b, *w["full_diag"], *w["full_nu"], *w["full_pair_energy"]]:
            if not math.isfinite(val):
                raise ValueError(f"Nonfinite full moment in {key}")
        equal(a[1], b[0])
        varying = w.get("varying_rate_counts", [None, None] if smoke else [])
        if len(varying) != 2:
            raise ValueError("Missing activity state")
        row = {}
        T = 0.0
        for ix, pop in enumerate(("E", "I")):
            M = counts[ix]
            weight = M / n
            if not smoke and not 0 <= varying[ix] <= M:
                raise ValueError("Invalid population activity count")
            tr = weight * w["full_nu"][ix]
            T += tr
            en = w["full_energy"][ix][ix]
            full = optional_divide(tr, en, smoke or varying[ix] > 0)
            equal(full, w["full_D_over_N"][ix])
            row[pop + "_per_cell"] = full / weight if full is not None else None
            if ix == 0:
                row["D_E_over_N"] = full
            pair = (en - w["full_diag"][ix]) * (n / M) * (n / (M - 1))
            equal(pair, w["full_pair_energy"][ix])
            row["P_" + pop * 2] = pair
            ss = w["subsets"][ix]
            if len(ss) != protocol["draws_per_class"]:
                raise ValueError("Incorrect number of subsets")
            prs = []
            y = []
            x = []
            for s in ss:
                F = s["frobenius_energy"]
                tr = s["trace"]
                if not all(map(math.isfinite, (tr, F, s["diagonal_energy"]))):
                    raise ValueError("Nonfinite subset moment")
                if s["PR"] is not None:
                    if F <= 0 or (not smoke and varying[ix] == 0):
                        raise ValueError("Defined stationary or nonpositive-energy PR")
                    eq = tr * tr / F
                    equal(eq, s["PR"])
                    prs.append(eq)
                y.append(F)
                x.append(tr * tr)
            valid = len(prs) == len(ss)
            value = mean(prs) if valid else None
            row[pop + "_readout"] = value
            row[pop + "_lo"] = (sum(prs) + len(ss) - len(prs)) / len(ss)
            row[pop + "_hi"] = (sum(prs) + (len(ss) - len(prs)) * protocol["readout_size"]) / len(ss)
            row[pop + "_gap"] = value - mean(x) / mean(y) if valid else None
            row[pop + "_parameter_residual"] = value - anchors[f]["readout_PR"][ix] if valid else None
            k = protocol["readout_size"]
            known_energy = n * (
                k / M * w["full_diag"][ix] + k * (k - 1) / (M * (M - 1)) * (en - w["full_diag"][ix])
            )
            row[pop + "_known_gap"] = (
                mean([z - X / known_energy for z, X in zip(prs, x)])
                if valid and known_energy > 0
                else None
            )
            if valid:
                # The ratio gap also equals minus the covariance of individual ratios and energies divided by mean energy.
                equal(
                    row[pop + "_gap"],
                    -mean((pr - value) * (F - mean(y)) for pr, F in zip(prs, y)) / mean(y),
                )
        joint = optional_divide(T, sum(a) + sum(b), smoke or any(varying))
        equal(joint, w["full_joint_D_over_N"])
        row["joint_per_cell"] = joint
        outcome[key] = row
    if not smoke:
        late = record["last_half_varying_rate_counts"]
        if any(not 0 <= v <= m for v, m in zip(late, counts)) or len(late) != 2:
            raise ValueError("Invalid late activity")
        for a in range(2):
            v = {key: w["varying_rate_counts"][a] for key, w in record["windows"].items()}
            if not (
                v["T4000_gap0"] == v["T4000_gap1"] <= v["T8000_gap1"] == v["T8000_gap0"]
                and late[a] <= v["T8000_gap0"]
            ):
                raise ValueError("Activity mask inconsistent across windows")
    return outcome


def sampling(values, reported, alpha=0.05, ci="ci95"):
    values = [v for v in values if v is not None]
    n = len(values)
    if reported["n"] != n:
        raise ValueError(f"Reported n {reported['n']} differs from {n}")
    m = mean(values) if n else None
    se = stdev(values) / math.sqrt(n) if n > 1 else None
    equal(reported["mean"], m)
    equal(reported["se"], se)
    d = stdtrit(n - 1, 1 - alpha / 2) * se if n > 1 else None
    for actual, ref in zip(reported[ci], [m - d, m + d] if d is not None else [None, None]):
        equal(actual, ref)


def physical(metric, f, p):
    n = p["network_size"]
    if metric == "D_E_over_N":
        return 1 / n, f
    if metric.startswith("P_"):
        return 0, n
    return 1, p["readout_size"]


def interval(row, metric, f, p):
    if row is not None and row[metric] is not None:
        return [row[metric]] * 2
    if row is not None and metric in ("E_readout", "I_readout"):
        return row[metric[0] + "_lo"], row[metric[0] + "_hi"]
    return physical(metric, f, p)


def check_aggregates(summary, rows, p, base):
    csv_path = base / "network_estimates.csv"
    if digest(csv_path.read_bytes()) != summary["network_estimates_csv_sha256"]:
        raise ValueError("CSV digest mismatch")
    found = set()
    with csv_path.open(newline="") as stream:
        for r in csv.DictReader(stream):
            idx = int(r["seed"]), float(r["fraction_e"]), r["window"]
            if idx in found or idx not in rows:
                raise ValueError("Unexpected or repeated CSV row")
            found.add(idx)
            for metric in METRICS:
                equal(rows[idx][metric], float(r[metric]) if r[metric] else None)
            for pop in ("E", "I"):
                a = (
                    float(r[pop + "_quadratic"]) + float(r[pop + "_remainder"])
                    if r[pop + "_quadratic"]
                    else None
                )
                equal(rows[idx][pop + "_gap"], a, "ratio expansion")
    if found != set(rows):
        raise ValueError("Missing CSV rows")
    seeds = range(p["seed_first"], p["seed_last"] + 1)
    summary_windows = summary["windows"]
    predictions = {a["fraction_e"]: a for a in THEORY["anchors"]}
    for w in WINDOWS:
        reported = summary_windows[w]
        for group in reported["groups"]:
            f = group["fraction_e"]
            rr = [rows[s, f, w] for s in seeds if (s, f, w) in rows]
            for m in METRICS:
                sampling([r[m] for r in rr], group["metrics"][m])
            for letter in ("E", "I"):
                for k, ref in [
                    ("ratio_gap", "gap"),
                    ("design_unbiased_ratio_gap", "known_gap"),
                    ("parameter_residual", "parameter_residual"),
                ]:
                    sampling(
                        [r[letter + "_" + ref] for r in rr], group["subset_ratio_diagnostics"][letter][k]
                    )
        for label, (metric, lo, hi) in CONTRASTS.items():
            estimates = []
            ranges = []
            sensitivity = []
            for s in seeds:
                a, b = (rows.get((s, f, w)) for f in (lo, hi))
                al, ah = interval(a, metric, lo, p)
                bl, bh = interval(b, metric, hi, p)
                ranges.append((bl - ah, bh - al))
                if a and b and a[metric] is not None and b[metric] is not None:
                    diff = b[metric] - a[metric]
                    estimates.append(diff)
                    primary_lo = rows[s, lo, p["primary_window"]][metric]
                    primary_hi = rows[s, hi, p["primary_window"]][metric]
                    if primary_lo is not None and primary_hi is not None:
                        sensitivity.append(diff - (primary_hi - primary_lo))
            compare = reported["contrasts"][label]
            sampling(estimates, compare)
            if label != "E_readout_80_88":
                sampling(estimates, compare, 0.01, "ci_family")
            source_key = (
                "readout_PR"
                if metric.endswith("readout")
                else "pair_energy"
                if metric.startswith("P_")
                else "D_E_over_N"
            )
            if source_key == "D_E_over_N":
                theory_delta = predictions[hi][source_key] - predictions[lo][source_key]
            else:
                a = int(metric[0] == "I" or metric == "P_II")
                theory_delta = predictions[hi][source_key][a] - predictions[lo][source_key][a]
            equal(theory_delta, compare["theory"])
            for ref, actual in zip(np.mean(ranges, axis=0), compare["physical_imputation_mean_range"]):
                equal(ref, actual)
            if sum(compare["exclusion_low_high"].values()) != len(seeds) - len(estimates):
                raise ValueError("Missing pair count mismatch")
            if w != p["primary_window"]:
                sampling(sensitivity, summary["window_minus_primary_paired_contrasts"][w][label])


def validate_smoke(path):
    record = load(path)
    p = load(HERE / "protocol.json")
    seed = record["job"]["config"]["seed"]
    f = record["job"]["config"]["fraction_e"]
    pilot = load(HERE.parent / f"readout_pilot/measurements/readout_f{f:g}_N2000_seed{seed}.json")
    if p["seed_first"] <= seed <= p["seed_last"]:
        raise ValueError("Expected prior pilot replay")
    if pilot["connectivity_sha256"] != record["connectivity_sha256"]:
        raise ValueError("Smoke graph differs from pilot")
    independent = truth(record, p, smoke=True)
    from summarize import reduce_window, points

    for w in WINDOWS:
        group = reduce_window(record, w, p, points())
        for m in METRICS:
            equal(independent[w][m], group[m])
        for a, pop in enumerate("EI"):
            for subset, new in zip(pilot["readouts"][a], record["windows"][w]["subsets"][a]):
                for k in ("PR", "trace", "frobenius_energy", "diagonal_energy"):
                    equal(subset[w][k], new[k])
            equal(independent[w][pop + "_gap"], group[pop + "_ratio_gap"])
            equal(independent[w][pop + "_known_gap"], group[pop + "_design_unbiased_ratio_gap"])
    print(
        f"Technical replay of pilot seed {seed}: four windows and both sets of eight subsets agree with independent pilot records. No confirmation output written."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=HERE / "records")
    ap.add_argument("--archives", type=Path, default=HERE / "data")
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument(
        "--smoke", type=Path, help="Verify one technical pilot replay; never create cohort results."
    )
    args = ap.parse_args()
    if args.smoke:
        validate_smoke(args.smoke)
        return
    p = load(HERE / "protocol.json")
    summary = load(args.out / "summary.json")
    pilot = load(HERE.parent / "readout_pilot/protocol.json")
    excluded = set(pilot["network_seeds"])
    rows = {}
    hashes = {}
    graphs = set()
    used_sources = {}
    binary = Counter()
    matched_graphs = 0
    # Independently regenerate one connectivity realization per fraction on each of 12 stripes.
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from eicavity.network import NetworkConfig, make_network

    for name, raw, source in record_stream(args.records, args.archives):
        hash_ = digest(raw)
        if name in hashes:
            if hashes[name] != hash_:
                raise ValueError("Conflicting duplicate records")
            continue
        hashes[name] = hash_
        r = json.loads(raw)
        cfg = r["job"]["config"]
        s = cfg["seed"]
        f = cfg["fraction_e"]
        if s in excluded or not p["seed_first"] <= s <= p["seed_last"] or f not in p["fractions"]:
            raise ValueError("Pilot or alien record in fresh cohort")
        if name != f"confirm_f{f:g}_N2000_seed{s}.json":
            raise ValueError("Filename/parameters mismatch")
        for field in summary["frozen_source_hashes"]:
            used_sources.setdefault(field, set()).add(r[field])
        binary[r["binary_sha256"]] += 1
        if r["connectivity_sha256"] in graphs:
            raise ValueError("Duplicate connectivity across cases")
        graphs.add(r["connectivity_sha256"])
        if s < p["seed_first"] + 12:
            J, _, _, labels, _ = make_network(NetworkConfig(**cfg), np.array(r["job"]["mean_input"]))
            if digest(J.tobytes()) != r["connectivity_sha256"]:
                raise ValueError("Graph reconstruction failed")
            for c in r["counts"]:
                a = int(c["target"] == "I")
                b = int(c["source"] == "I")
                degrees = np.count_nonzero(J[np.ix_(labels == a, labels == b)], axis=1)
                if sum(degrees) != c["total_edges"]:
                    raise ValueError("Edge checksum failed")
                equal(degrees.var(), c["degree_variance"])
            matched_graphs += 1
        for w, vals in truth(r, p).items():
            rows[s, f, w] = vals
    source_files = {
        "protocol_sha256": HERE / "protocol.json",
        "runner_sha256": HERE / "run.py",
        "engine_sha256": HERE.parent / "runtime/engine_v4.py",
        "measurement_source_sha256": HERE.parent / "readout_pilot/measure.py",
    }
    actual = {k: digest(path.read_bytes()) for k, path in source_files.items()}
    if summary["frozen_source_hashes"] != actual or any(
        v != {actual[k]} for k, v in used_sources.items()
    ):
        raise ValueError("Unverified numerical source")
    study = {(s, f) for s in range(p["seed_first"], p["seed_last"] + 1) for f in p["fractions"]}
    missing = sorted(
        f"confirm_f{f:g}_N2000_seed{s}" for s, f in study if (s, f, p["primary_window"]) not in rows
    )
    if sorted(summary["missing_records"]) != missing or summary["cohort_complete"] != bool(not missing):
        raise ValueError("Cohort completeness mismatch")
    if len(hashes) != summary["records"] or binary != summary["native_binary_sha256_counts"]:
        raise ValueError("Record count mismatch")
    manifest = "".join(hash_ + "  " + name + "\n" for name, hash_ in sorted(hashes.items()))
    if digest(manifest.encode()) != summary["record_set_sha256"]:
        raise ValueError("Manifest digest mismatch")
    check_aggregates(summary, rows, p, args.out)
    result = {
        "complete_cohort": not missing,
        "checked_records": len(hashes),
        "checked_csv_rows": len(rows),
        "checked_per_record_windows": list(WINDOWS),
        "pilot_seeds_excluded": sorted(excluded),
        "independently_regenerated_graphs": matched_graphs,
        "record_set_sha256": summary["record_set_sha256"],
        "summary_sha256": digest((args.out / "summary.json").read_bytes()),
        "csv_sha256": summary["network_estimates_csv_sha256"],
        "analysis_sha256": digest((HERE / "summarize.py").read_bytes()),
        "validator_sha256": digest(Path(__file__).read_bytes()),
        "theory_sha256": digest((HERE.parent / "readout_theory/predictions.json").read_bytes()),
        "verified": "All subset indices, finite traces and energies, population and subset PRs, network-level pair contrasts and intervals, moment-ratio expansion, contrast imputation bounds, duration/gap sensitivities and archive/CSV hashes.",
    }
    atomic_json(args.out / "validation.json", result)
    print(
        f"Independently checked {len(hashes)} records and {len(rows)} per-window network rows, including {matched_graphs} regenerated connectivities."
    )


if __name__ == "__main__":
    main()
