"""Aggregate all prespecified windows of the paired readout confirmation."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import tarfile
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
THEORY = HERE.parent / "readout_theory/predictions.json"
PILOT = HERE.parent / "readout_pilot/protocol.json"
BASE = HERE.parent / "dense_sweeps/jobs.json"
WINDOWS = ("T8000_gap0", "T4000_gap0", "T8000_gap1", "T4000_gap1")
METRICS = (
    "E_readout",
    "I_readout",
    "D_E_over_N",
    "E_per_cell",
    "I_per_cell",
    "joint_per_cell",
    "P_EE",
    "P_II",
)
CONTRASTS = {
    "E_readout_80_88": ("E_readout", 0.80, 0.88),
    "I_readout_80_88": ("I_readout", 0.80, 0.88),
    "P_EE_80_88": ("P_EE", 0.80, 0.88),
    "P_II_80_88": ("P_II", 0.80, 0.88),
    "D_E_over_N_80_88": ("D_E_over_N", 0.80, 0.88),
    "D_E_over_N_88_94": ("D_E_over_N", 0.88, 0.94),
}
RATIO_FIELDS = (
    "readout",
    "empirical_ratio",
    "ratio_gap",
    "quadratic",
    "remainder",
    "parameter_residual",
    "trace_squared_mean",
    "energy_mean",
    "population_moment_ratio",
    "population_expected_energy",
    "design_unbiased_ratio_gap",
    "sample_minus_population_moment_ratio",
)
SHA = re.compile(r"[0-9a-f]{64}")


def read(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_hashes():
    return {
        k: digest(p.read_bytes())
        for k, p in (
            ("protocol_sha256", PROTOCOL),
            ("runner_sha256", HERE / "run.py"),
            ("engine_sha256", HERE.parent / "runtime/engine_v4.py"),
            ("measurement_source_sha256", HERE.parent / "readout_pilot/measure.py"),
        )
    }


def atomic_json(path, value):
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    Path(stream.name).replace(path)


def record_stream(record_dir, archive_dir):
    for p in sorted(record_dir.glob("*.json")):
        yield p.name, p.read_bytes(), str(p)
    for archive in sorted(archive_dir.glob("stripe-*.tgz")):
        hashes = {}
        manifest = None
        with tarfile.open(archive, "r|gz") as stream:
            for member in stream:
                name = member.name.removeprefix("./")
                if member.isdir() and name == "records":
                    continue
                if not member.isfile() or member.size > 10 * 1024**2:
                    raise ValueError(f"Unexpected member of {archive.name}: {name}")
                raw = stream.extractfile(member).read()
                if name == "checksums.sha256" and manifest is None:
                    manifest = {}
                    for line in raw.decode().splitlines():
                        h, file = line.split()
                        file = file.removeprefix("*").removeprefix("./")
                        if not SHA.fullmatch(h) or file in manifest:
                            raise ValueError(f"Invalid checksum manifest in {archive}")
                        manifest[file] = h
                elif re.fullmatch(r"records/[A-Za-z0-9_.-]+\.json", name) and name not in hashes:
                    hashes[name] = digest(raw)
                    yield Path(name).name, raw, archive.name
                else:
                    raise ValueError(f"Unexpected or duplicate member: {archive.name}/{name}")
        if not hashes or hashes != manifest:
            raise ValueError(f"Record checksums do not match in {archive}")


def points():
    anchors = read(THEORY)["anchors"]
    return {
        a["fraction_e"]: {
            "E_readout": a["readout_PR"][0],
            "I_readout": a["readout_PR"][1],
            "D_E_over_N": a["D_E_over_N"],
            "E_per_cell": a["within_population_dimension_fraction"][0],
            "I_per_cell": a["within_population_dimension_fraction"][1],
            "joint_per_cell": a["joint_dimension_fraction"],
            "P_EE": a["pair_energy"][0],
            "P_II": a["pair_energy"][1],
        }
        for a in anchors
    }


def estimate(values, alpha=0.05):
    x = np.array([v for v in values if v is not None], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite network estimate")
    n = len(x)
    mean = float(x.mean()) if n else None
    sd = float(x.std(ddof=1)) if n > 1 else None
    se = sd / np.sqrt(n) if n > 1 else None
    half = t.ppf(1 - alpha / 2, n - 1) * se if n > 1 else None
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95": [mean - half, mean + half] if half is not None else [None, None],
    }


def physical_range(metric, f, protocol):
    n = protocol["network_size"]
    if metric.endswith("_readout"):
        return 1.0, float(protocol["readout_size"])
    if metric.startswith("P_"):
        # Each rate lies in [0,2], hence the magnitude of a true covariance is <=1.
        return 0.0, float(n)
    if metric == "D_E_over_N":
        return 1 / n, f
    count = {"E_per_cell": round(n * f), "I_per_cell": n - round(n * f), "joint_per_cell": n}[metric]
    return 1 / count, 1.0


def reduce_window(raw, key, p, pred):
    w = raw["windows"][key]
    cfg = raw["job"]["config"]
    f = cfg["fraction_e"]
    n = cfg["n"]
    sizes = [round(f * n), n - round(f * n)]
    k = p["readout_size"]
    group = w["full_D_over_N"]
    row = {
        "D_E_over_N": group[0],
        "E_per_cell": group[0] / f if group[0] is not None else None,
        "I_per_cell": group[1] * n / sizes[1] if group[1] is not None else None,
        "joint_per_cell": w["full_joint_D_over_N"],
        "P_EE": w["full_pair_energy"][0],
        "P_II": w["full_pair_energy"][1],
    }
    for a, letter in enumerate(("E", "I")):
        cells = w["subsets"][a]
        x = np.asarray([s["trace"] ** 2 for s in cells])
        y = np.asarray([s["frobenius_energy"] for s in cells])
        valid = np.array([s["PR"] is not None for s in cells])
        prs = [s["PR"] for s in cells if s["PR"] is not None]
        row[letter + "_readout"] = float(np.mean(prs)) if valid.all() else None
        m = len(cells)
        row[letter + "_subset_undefined"] = int(m - valid.sum())
        row[letter + "_subset_nonpositive_energy"] = int(np.sum(y <= 0))
        row[letter + "_varying_cells"] = w.get("varying_rate_counts", [None, None])[a]
        row[letter + "_subset_bound_lo"] = (sum(prs) + np.sum(~valid)) / m
        row[letter + "_subset_bound_hi"] = (sum(prs) + k * np.sum(~valid)) / m
        for field in RATIO_FIELDS[1:]:
            row[letter + "_" + field] = None
        if valid.all():
            my = y.mean()
            mx = x.mean()
            ratio = mx / my
            dx = x - mx
            dy = y - my
            quad = (ratio * np.mean(dy**2) - np.mean(dx * dy)) / my**2
            remainder = np.mean((dx - ratio * dy) * dy**2 / (my**2 * y))
            # Uniform subset sampling, evaluated at measured population moments.
            # A squared estimated mean trace still carries finite-record error.
            m2 = w["full_diag"][a] * n / sizes[a]
            mean = w["full_nu"][a]
            expected_x = (k * mean) ** 2 + k * (sizes[a] - k) / (sizes[a] - 1) * (m2 - mean**2)
            expected_y = k * (m2 + (k - 1) * w["full_pair_energy"][a] / n)
            full_ratio = expected_x / expected_y if expected_y > 0 else None
            quantities = {
                "empirical_ratio": ratio,
                "ratio_gap": row[letter + "_readout"] - ratio,
                "quadratic": quad,
                "remainder": remainder,
                "parameter_residual": row[letter + "_readout"] - pred[f][letter + "_readout"],
                "trace_squared_mean": mx,
                "energy_mean": my,
                "population_moment_ratio": full_ratio,
                "population_expected_energy": expected_y,
                "design_unbiased_ratio_gap": row[letter + "_readout"] - mx / expected_y
                if expected_y > 0
                else None,
                "sample_minus_population_moment_ratio": ratio - full_ratio
                if full_ratio is not None
                else None,
            }
            row.update({letter + "_" + k_: v for k_, v in quantities.items()})
    return row


def bounds(row, metric, f, p):
    if row is not None and row[metric] is not None:
        return [row[metric]] * 2
    if row is not None and metric.endswith("_readout"):
        return [row[metric[0] + "_subset_bound_" + side] for side in ("lo", "hi")]
    return physical_range(metric, f, p)


def reason(row, metric):
    if row is None:
        return "missing_record"
    if row[metric] is not None:
        return "defined"
    letter = metric[0]
    if metric == "D_E_over_N":
        letter = "E"
    stationary = (
        row["E_varying_cells"] == row["I_varying_cells"] == 0
        if metric == "joint_per_cell"
        else row[letter + "_varying_cells"] == 0
    )
    if stationary:
        return "stationary_population"
    if metric.endswith("_readout"):
        return (
            "nonpositive_subset_energy_or_stationary_subset"
            if row[letter + "_subset_nonpositive_energy"]
            else "stationary_subset"
        )
    return "nonpositive_population_energy"


def paired(rows, metric, lo, hi, seeds, p, pred, secondary):
    differences = []
    ranges = []
    reasons = Counter()
    for s in seeds:
        a = rows.get((s, lo))
        b = rows.get((s, hi))
        why = (reason(a, metric), reason(b, metric))
        if why == ("defined", "defined"):
            differences.append(b[metric] - a[metric])
        else:
            reasons["/".join(why)] += 1
        a0, a1 = bounds(a, metric, lo, p)
        b0, b1 = bounds(b, metric, hi, p)
        ranges.append((b0 - a1, b1 - a0))
    result = estimate(differences)
    result["ci_family"] = estimate(differences, alpha=0.05 / 5)["ci95"] if secondary else None
    result.update(
        theory=pred[hi][metric] - pred[lo][metric],
        missing_or_undefined_pairs=len(seeds) - len(differences),
        exclusion_low_high=dict(reasons),
        physical_imputation_mean_range=np.mean(ranges, axis=0).tolist(),
    )
    return result


def combine(data, p):
    seeds = range(p["seed_first"], p["seed_last"] + 1)
    predictions = points()
    windows = {}
    for key, rows in data.items():
        groups = []
        for f in p["fractions"]:
            group_rows = [rows[s, f] for s in seeds if (s, f) in rows]
            metrics = {k: estimate(r[k] for r in group_rows) for k in METRICS}
            for k in METRICS:
                metrics[k]["theory"] = predictions[f][k]
                metrics[k]["availability"] = dict(Counter(reason(rows.get((s, f)), k) for s in seeds))
            ratio = {
                letter: {k: estimate(r[letter + "_" + k] for r in group_rows) for k in RATIO_FIELDS}
                for letter in ("E", "I")
            }
            groups.append(
                {
                    "fraction_e": f,
                    "records": len(group_rows),
                    "metrics": metrics,
                    "subset_ratio_diagnostics": ratio,
                }
            )
        contrasts = {
            k: paired(rows, *contract, seeds, p, predictions, k != "E_readout_80_88")
            for k, contract in CONTRASTS.items()
        }
        remainder = {}
        for letter in ("E", "I"):
            remainder[letter] = {}
            for metric in (
                "ratio_gap",
                "design_unbiased_ratio_gap",
                "quadratic",
                "remainder",
                "parameter_residual",
                "sample_minus_population_moment_ratio",
            ):
                values = []
                for s in seeds:
                    a = rows.get((s, 0.8))
                    b = rows.get((s, 0.88))
                    col = letter + "_" + metric
                    if a and b and a[col] is not None and b[col] is not None:
                        values.append(b[col] - a[col])
                remainder[letter][metric] = estimate(values)
        windows[key] = {
            "groups": groups,
            "contrasts": contrasts,
            "paired_subset_ratio_diagnostics_80_88": remainder,
        }
    change = {}
    primary = data[p["primary_window"]]
    for key, rows in data.items():
        if key == p["primary_window"]:
            continue
        change[key] = {}
        for label, (metric, lo, hi) in CONTRASTS.items():
            d = []
            for s in seeds:
                four = [
                    source.get((s, f), {}).get(metric) for source in (rows, primary) for f in (lo, hi)
                ]
                if all(v is not None for v in four):
                    d.append((four[1] - four[0]) - (four[3] - four[2]))
            change[key][label] = estimate(d)
    return windows, change


def collect(record_dir, archive_dir):
    p = read(PROTOCOL)
    pred = points()
    expected_hash = source_hashes()
    template = {
        x["config"]["fraction_e"]: x
        for x in read(BASE)
        if x["experiment"] == "composition"
        and x["config"]["seed"] == p["reference_config_seed"]
        and x["config"]["n"] == p["network_size"]
    }
    seeds = set(range(p["seed_first"], p["seed_last"] + 1))
    if seeds.intersection(read(PILOT)["network_seeds"]):
        raise ValueError("Pilot overlaps confirmation")
    rows = {w: {} for w in WINDOWS}
    hashes = {}
    graph_hashes = set()
    checks = defaultdict(float)
    workers = Counter()
    sources = Counter()
    cell_counts = {}
    last_half = Counter()
    for name, raw, source in record_stream(record_dir, archive_dir):
        h = digest(raw)
        if name in hashes:
            if hashes[name] != h:
                raise ValueError("Conflicting copies of " + name)
            continue
        d = json.loads(raw)
        cfg = d["job"]["config"]
        s = cfg["seed"]
        f = cfg["fraction_e"]
        key = (s, f)
        expected_name = f"confirm_f{f:g}_N2000_seed{s}"
        if (
            s not in seeds
            or f not in p["fractions"]
            or name != expected_name + ".json"
            or d["job"]
            != {
                "name": expected_name,
                "config": {**template[f]["config"], "seed": s},
                "mean_input": template[f]["mean_input"],
            }
        ):
            raise ValueError("Out-of-protocol record: " + name)
        if set(d["windows"]) != set(WINDOWS) or any(d[k] != value for k, value in expected_hash.items()):
            raise ValueError("Unexpected window or numerical source: " + name)
        for k in ("binary_sha256", "indices_sha256", "connectivity_sha256"):
            if not SHA.fullmatch(d[k]):
                raise ValueError(f"Invalid {k}: {name}")
        if d["connectivity_sha256"] in graph_hashes:
            raise ValueError("Repeated graph: " + name)
        graph_hashes.add(d["connectivity_sha256"])
        hashes[name] = h
        workers[d["binary_sha256"]] += 1
        sources[source if not source.endswith(".json") else "records/"] += 1
        for k, v in d["checks"].items():
            checks[k] = max(checks[k], v)
        for w in WINDOWS:
            r = reduce_window(d, w, p, pred)
            r.update(
                seed=s,
                fraction_e=f,
                window=w,
                record_sha256=h,
                connectivity_sha256=d["connectivity_sha256"],
            )
            rows[w][key] = r
        last_half[(f, tuple(v == 0 for v in d["last_half_varying_rate_counts"]))] += 1
        for count in d["counts"]:
            k = (f, count["target"], count["source"])
            cell_counts.setdefault(k, []).append(count)
    if not hashes:
        raise ValueError("No confirmation records supplied.")
    missing = [
        f"confirm_f{f:g}_N2000_seed{s}"
        for s in sorted(seeds)
        for f in p["fractions"]
        if (s, f) not in rows[WINDOWS[0]]
    ]
    counts = [
        {
            "fraction_e": f,
            "target": tgt,
            "source": src,
            "mean_realized_degree": estimate(v["mean_degree"] for v in vals),
            "mean_missing_inputs": estimate(v["mean_missing_inputs"] for v in vals),
            "range_realized_mean_degree": [
                min(v["mean_degree"] for v in vals),
                max(v["mean_degree"] for v in vals),
            ],
        }
        for (f, tgt, src), vals in sorted(cell_counts.items())
    ]
    meta = {
        "cohort_complete": not missing,
        "expected_records": len(seeds) * len(p["fractions"]),
        "records": len(hashes),
        "seed_range": [min(seeds), max(seeds)],
        "fractions": p["fractions"],
        "missing_records": missing,
        "pilot_seeds_excluded": read(PILOT)["network_seeds"],
        "frozen_source_hashes": expected_hash,
        "theory_sha256": digest(THEORY.read_bytes()),
        "native_binary_sha256_counts": dict(workers),
        "record_set_sha256": digest(
            "".join(f"{h}  {name}\n" for name, h in sorted(hashes.items())).encode()
        ),
        "records_by_first_source": dict(sources),
        "maximum_recorded_checks": dict(checks),
        "input_edge_counts": counts,
        "last_half_states": [
            {"fraction_e": f, "E_stationary": flags[0], "I_stationary": flags[1], "records": n}
            for (f, flags), n in sorted(last_half.items())
        ],
    }
    return rows, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=HERE / "records")
    ap.add_argument("--archives", type=Path, default=HERE / "data")
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write an explicitly incomplete diagnostic to a separate output directory.",
    )
    args = ap.parse_args()
    rows, meta = collect(args.records, args.archives)
    if not meta["cohort_complete"] and (not args.allow_incomplete or args.out.resolve() == HERE):
        raise SystemExit(
            f"Missing {len(meta['missing_records'])} networks. Use --allow-incomplete --out <separate directory> for diagnostics."
        )
    args.out.mkdir(parents=True, exist_ok=True)
    windows, sensitivity = combine(rows, read(PROTOCOL))
    summary = {
        **meta,
        "windows": windows,
        "window_minus_primary_paired_contrasts": sensitivity,
        "ratio_scope": "Eight prespecified subset PRs averaged per network. Ratio gap is the finite empirical mean PR minus mean trace squared divided by mean energy. Quadratic and exact remainder sum to that gap. Four-window ratio estimates retain finite-record bias.",
        "known_energy_ratio_gap_scope": "For a fixed recording and uniform K-cell subsets, the exact subset-design mean Frobenius energy is K times the mean diagonal energy plus K(K-1)P_aa/N. Averaging trace²/energy minus trace² divided by that known mean estimates E[PR] - E[trace²]/E[energy] without a sampled denominator. This design-unbiased contrast requires defined readouts and does not correct finite-record temporal bias. The companion population moment ratio plugs a four-window diagonal-energy estimate into the trace-squared moment.",
        "missingness_scope": "If one of eight subset ratios is undefined, the network endpoint is undefined. Physical bounds impute only missing/undefined values, retaining finite observed estimates. Pair-energy bounds use rates in [0,2]. Stationary zero-covariance PR has no defined physical value; imputation is a sensitivity scenario.",
    }
    csv_path = args.out / "network_estimates.csv"
    keys = ["seed", "fraction_e", "window", "record_sha256", "connectivity_sha256", *METRICS]
    keys += sorted(set(next(iter(rows[WINDOWS[0]].values()))) - set(keys))
    with NamedTemporaryFile(mode="w", newline="", dir=args.out, delete=False) as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        for w in WINDOWS:
            writer.writerows(r for _, r in sorted(rows[w].items()))
    Path(fp.name).replace(csv_path)
    summary["network_estimates_csv_sha256"] = digest(csv_path.read_bytes())
    atomic_json(args.out / "summary.json", summary)
    print(
        f"{meta['records']} of {meta['expected_records']} records; {len(WINDOWS)} windows; wrote {args.out}"
    )


if __name__ == "__main__":
    main()
