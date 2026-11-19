"""Read the population measurements used by the degree and composition figures."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np


ROUND = Path(__file__).resolve().parents[1]


def composition_moments():
    """Fraction-independent tensor and local rate moments on the fixed-coupling path."""
    sys.path.insert(0, str(ROUND.parent / "src"))
    from eicavity.composition import energy_tensor

    spec = json.loads((ROUND / "degree/theory_curve.json").read_text())
    W = np.array(spec["G"]) * spec["fractions"]
    with np.load(ROUND / "degree/chi1_one_site.npz") as site:
        dt = site["time"][1] - site["time"][0]
        tensor = energy_tensor(W, site, dt)
        variance = site["covariance"][..., 0]
        means = variance @ site["weights"]
        seconds = variance**2 @ site["weights"]
    return tensor, means, seconds


def figure_record(record):
    """One independently generated network, at the published 8000/500/gap-0 protocol."""
    if record["status"] != "complete":
        raise ValueError(f"{record['name']} has status {record['status']!r}")
    job = record["job"]
    est = next(item for item in record["estimators"] if item["block_duration"] == 500)
    prefix = next(item for item in est["prefixes"] if item["duration"] == 8000)
    gap = {item["empty_blocks"]: item for item in prefix["gaps"]}
    return {"name": record["name"], "config": job["config"], "duration": prefix["duration"],
            "primary": gap[0], "gap1_sensitivity": gap[1], "local": est["population_local_stats"],
            "job": job}


def dense_records(read=None):
    """Load exactly the additional networks from the chosen project tree or ZIP."""
    if read is None:
        read = lambda name: (ROUND.parent / name).read_bytes()
    directory = "experiments/dense_sweeps"
    try:
        jobs = json.loads(read(f"{directory}/jobs.json"))
    except (KeyError, FileNotFoundError):
        return []
    rows = []
    missing = []
    for job in jobs:
        path = f"{directory}/runs/{job['name']}.json"
        try:
            record = json.loads(read(path))
        except (KeyError, FileNotFoundError):
            missing.append(job["name"])
            continue
        if (record["name"] != job["name"] or record["job"]["config"] != job["config"]
                or record["job"]["experiment"] != job["experiment"]):
            raise ValueError(f"Result does not match the dense-sweep manifest: {path}")
        rows.append(figure_record(record))
    if missing:
        raise ValueError(f"{len(missing)} of {len(jobs)} dense-sweep records are missing")
    if len({r["name"] for r in rows}) != len(rows):
        raise ValueError("Duplicate dense-sweep network in the manifest")
    return rows


def grouped_networks(original, dense):
    """Keep size, sweep setting and original/fresh cohort distinct."""
    groups = defaultdict(list)
    for cohort, rows in [("original", original), ("dense", dense)]:
        for row in rows:
            cfg = row["config"]
            key = cfg["degree_dispersion"] if row["job"]["experiment"] == "degree" else cfg["fraction_e"]
            groups[key, cfg["n"], cohort].append(row)
    return groups


def point_offsets(groups, max_width):
    """Dodge cell summaries within the local spacing between real sweep settings."""
    settings = sorted({key[0] for key in groups})
    offsets = {}
    for idx, x in enumerate(settings):
        neighbors = settings[max(0, idx-1):idx] + settings[idx+1:idx+2]
        width = min([max_width] + [0.24*abs(x-y) for y in neighbors])
        keys = sorted(key for key in groups if key[0] == x)
        for key, shift in zip(keys, np.linspace(-width, width, len(keys)) if len(keys)>1 else [0.]):
            offsets[key] = shift
    return offsets


def draw_group(ax, x, values, marker, color, cohort, *, jitter=0., show_networks=True):
    """Mark individual draws, their arithmetic mean and one between-network SE."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(values) == 0:
        raise ValueError(f"Invalid measurements in cell {x:g}")
    if show_networks:
        dx = np.linspace(-jitter, jitter, len(values)) if len(values)>1 else np.zeros(1)
        ax.scatter(x+dx, values, s=6, color=color, alpha=.30, edgecolors="none", zorder=3)
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values)>1 else None
    ax.errorbar(x, values.mean(), yerr=se, fmt=marker, color=color,
                fillstyle="left" if cohort=="original" else "full",
                markerfacecoloralt="white", markeredgewidth=.7, elinewidth=.85,
                markersize=4.1, capsize=1.7, zorder=4)


def cohort_handles(sizes, *, has_dense=True, reference=False):
    from matplotlib.lines import Line2D
    from paper_style import JOINT, SIZE_MARKERS

    def marker(symbol, style, label):
        return Line2D([], [], marker=symbol, color=JOINT, mfcalt="white",
                      fillstyle=style, ls="none", ms=5, label=label)

    handles = [marker(SIZE_MARKERS[n], "full", f"N = {n:,}") for n in sorted(sizes)]
    handles.append(marker("o", "left", "Original cohort"))
    if has_dense:
        handles.append(marker("o", "full", "Dense sweep"))
    if reference:
        handles.append(marker("o", "none", "Bernoulli reference"))
    return handles
