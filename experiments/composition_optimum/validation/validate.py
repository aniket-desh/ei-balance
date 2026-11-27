"""Checks run before and around the cohort; writes validation.json.

    python experiments/composition_optimum/validation/validate.py

1. Generator: make_network with mean_rate [0.3, 0.3] is bit-identical to scalar 0.3; with unequal
   target rates the drive equals -sqrt(N) J0 (f * m) + h computed by hand after a JSON round
   trip, and the connectivity does not depend on the rates. No simulator code was changed.
2. Check networks (checks/check_networks.csv): measured population mean rates against targets.
3. Estimator on synthetic data (estimator_validation.json, from validate_estimator.py).
4. Runner replay (checks/replay): the archived campaign network rerun through this copy of
   run.py on a pod with the archive's numpy/scipy, compared with the archived record.
5. Collection checksums (cloud_execution.json), if present.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "recovered/bundle/ei_full/ei_balance_handoff/src"), str(HERE)]
from eicavity.network import NetworkConfig, make_network  # noqa: E402  (the module the engine imports)
import prepare  # noqa: E402


def generator():
    cfg, h = prepare.config_for("ref", 2000, 50001, .85)
    a = make_network(NetworkConfig(**cfg), np.array(h))
    b = make_network(NetworkConfig(**dict(cfg, mean_rate=[.3, .3])), np.array(h))
    out = dict(scalar_vs_pair_identical=bool(all(np.array_equal(x, y) for x, y in zip(a[:4], b[:4]))))
    for key in ("lo", "hi"):
        c, hh = prepare.config_for(key, 2000, 50001, .85)
        c = json.loads(json.dumps(c))
        J, drive, x, labels, info = make_network(NetworkConfig(**c), np.array(hh))
        fr = np.bincount(labels) / 2000
        hand = -np.sqrt(2000) * (np.array(c["mean_connectivity"]) @ (fr * np.array(c["mean_rate"]))) + np.array(hh)
        out[key] = dict(drive=info["external"].tolist(), max_error_vs_hand=float(np.abs(info["external"] - hand).max()),
                        connectivity_identical_to_reference=bool(np.array_equal(J, a[0])))
    out["passed"] = out["scalar_vs_pair_identical"] and all(out[k]["max_error_vs_hand"] < 1e-12 and out[k]["connectivity_identical_to_reference"]
                                                           for k in ("lo", "hi"))
    return out


def check_networks():
    rows = list(csv.DictReader(open(HERE / "checks/check_networks.csv")))
    out = []
    for r in rows:
        out.append(dict(name=r["name"], targets=[float(r["m_E"]), float(r["m_I"])], measured=[float(r["rate_E"]), float(r["rate_I"])],
                        max_abs_deviation=max(abs(float(r["rate_E"]) - float(r["m_E"])), abs(float(r["rate_I"]) - float(r["m_I"]))),
                        state=r["state"]))
    return dict(networks=out, threshold=0.02, within_threshold=sum(o["max_abs_deviation"] <= .02 for o in out), total=len(out),
                note="Both misses are (0.2, 0.4) at f = 0.94; see EXPERIMENT.md, 'Results of the checks run before freezing'.")


def replay():
    name = prepare.REPLAY
    a = json.load(gzip.open(HERE / f"checks/replay/records/{name}.json.gz", "rt"))
    b = json.load(gzip.open(HERE.parents[1] / f"paper_campaign/records/{name}.json.gz", "rt"))
    worst = 0.
    for ea, eb in zip(a["estimators"], b["estimators"]):
        for pa, pb in zip(ea["prefixes"], eb["prefixes"]):
            for ga, gb in zip(pa["gaps"], pb["gaps"]):
                worst = max(worst, abs(ga["d"] - gb["d"]), float(np.abs(np.array(ga["population_energy"]) - np.array(gb["population_energy"])).max()),
                            float(np.abs(np.array(ga["nu_by_population"]) - np.array(gb["nu_by_population"])).max()))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(network=name, connectivity_identical=a["connectivity_sha256"] == b["connectivity_sha256"], max_abs_difference=worst,
                arrays_byte_identical=sha(HERE / f"checks/replay/arrays/{name}.npz") == sha(HERE.parents[1] / f"paper_campaign/arrays/{name}.npz"),
                replay_host=a["host"], archive_host=b["host"], passed=bool(worst == 0. and a["connectivity_sha256"] == b["connectivity_sha256"]))


def main():
    out = dict(generator=generator(), check_networks=check_networks())
    ev = json.loads((HERE / "estimator_validation.json").read_text())
    out["estimator"] = dict(noiseless={k: v for k, v in ev.items() if k.startswith("noiseless")},
                            designs=[dict(design=d["design"], sd_shift_lo=d["sd_shift_lo"], sd_shift_hi=d["sd_shift_hi"],
                                          by_rate={k: dict(bias=v["mean_f_star"] - v["true_f_star"], sd=v["sd_f_star"], coverage_95=v["coverage_95"])
                                                   for k, v in d["by_rate"].items()}) for d in ev["designs"]])
    out["replay"] = replay()
    if (HERE / "cloud_execution.json").exists():
        out["collection"] = json.loads((HERE / "cloud_execution.json").read_text()).get("checksums")
    (HERE / "validation.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: (v.get("passed") if isinstance(v, dict) else None) for k, v in out.items()}, indent=1))
    print("check networks within threshold:", out["check_networks"]["within_threshold"], "of", out["check_networks"]["total"])


if __name__ == "__main__":
    main()
