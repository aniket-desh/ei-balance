"""Check the structural estimator on synthetic cohorts generated from a known tensor; size the design.

    python experiments/composition_optimum/validation/validate_estimator.py

Synthetic network at fraction f: y = mu(f) * (1 + s (r - 1)), where mu(f) holds the
parameter-only block energies and variances (so the true T, v and f* are known) and r is a
5-vector of multiplicative residuals y / (cell mean) drawn with replacement from the 624
fluctuating composition networks of experiments/paper_campaign (N = 2,000). This keeps their
heavy tails and the correlations between the five quantities. s scales the spread for larger
N; the pilot spreads (relative SD of block energies about 0.14 at N = 2,000, 0.10 to 0.12 at
4,000 and 0.04 at 8,000 from 12 and 6 networks) motivate s = 0.75 at 4,000 and s in
{0.3, 0.45} at 8,000.

Reported per design: bias and SD of each fitted T_abc and of f*, the probability that the
95% bootstrap percentile interval for f* covers the true value, and the SD of the shift
between rate pairs. Writes estimator_validation.json.
"""
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import estimators as est  # noqa: E402

PRED = json.loads((HERE / "predictions.json").read_text())
FR = PRED["fractions"]
KEYS = ("E_EE", "E_EI", "E_II", "nu_E", "nu_I")


def pilot_residuals():
    rows = [x for x in csv.DictReader(open(HERE.parents[1] / "paper_campaign/networks.csv"))
            if x["family"] == "composition" and x["fluctuating"] == "1"]
    cells = {}
    for x in rows:
        cells.setdefault(x["fraction_e"], []).append([float(x[k]) for k in KEYS])
    return np.concatenate([np.array(c) / np.array(c).mean(0) for c in cells.values()])


def truth(key):
    r = PRED["rate_pairs"][key]
    T, v = np.array(r["tensor_array"]), np.array(r["v"])
    mu = {f: np.array([r["cells"][f"{f:g}"][k] for k in ("E_EE", "E_EI", "E_II")]) for f in FR}
    return dict(mu=mu, T=T, v=v, f_star=r["f_star"])


def cohort(rng, res, pred, n, s):
    cells = {}
    for f in FR:
        m = np.r_[pred["mu"][f], pred["v"]]
        cells[f] = m * (1 + s * (res[rng.integers(0, len(res), n)] - 1))
    return cells


def main():
    res = pilot_residuals()
    flanks = tuple(PRED["flanks"])
    preds = {k: truth(k) for k in PRED["rate_pairs"]}
    out = dict(residual_pool=len(res), designs=[])
    rng = np.random.default_rng(20261127)
    tic = time.time()
    # 1. Exactness: with zero noise the fit returns T, v and f* to rounding.
    for k, p in preds.items():
        c = cohort(rng, res, p, 4, 0.)
        s = est.summarize(c, p, flanks)
        out[f"noiseless_{k}"] = dict(max_T_error=float(np.abs(s["fit"]["T"] - p["T"]).max()),
                                     f_star_error=float(s["f_star"] - p["f_star"]),
                                     max_chi2=float(max(s["fit"]["chi2"])) if np.all(np.isfinite(s["fit"]["chi2"])) else None)
        assert out[f"noiseless_{k}"]["max_T_error"] < 1e-12 and abs(out[f"noiseless_{k}"]["f_star_error"]) < 1e-9
    # 2. Bias, spread and coverage for the planned designs.
    for label, n, s, reps, boots in (("N2000_n64", 64, 1., 300, 400), ("N4000_n16", 16, .75, 300, 400),
                                     ("N8000_n10_s0.30", 10, .30, 300, 400), ("N8000_n10_s0.45", 10, .45, 300, 400),
                                     ("N8000_n6_s0.45", 6, .45, 200, 400)):
        row = dict(design=label, n_per_cell=n, spread_scale=s, replicates=reps, bootstrap_draws=boots, by_rate={})
        fstars = {}
        for k, p in preds.items():
            fs, Ts, cover, nroots = [], [], 0, []
            for _ in range(reps):
                c = cohort(rng, res, p, n, s)
                sm = est.summarize(c, p, flanks)
                fs.append(sm["f_star"])
                Ts.append(sm["fit"]["T"])
                nroots.append(sm["nroots"])
                b = np.array([d["f_star"] for d in est.bootstrap(c, p, flanks, draws=boots, seed=int(rng.integers(1 << 31)))])
                lo, hi = np.nanpercentile(b, [2.5, 97.5])
                cover += lo <= p["f_star"] <= hi
            fs, Ts = np.array(fs), np.array(Ts)
            fstars[k] = fs
            rel = Ts / p["T"]
            row["by_rate"][k] = dict(true_f_star=p["f_star"], mean_f_star=float(np.nanmean(fs)), sd_f_star=float(np.nanstd(fs)),
                                     coverage_95=cover / reps, frac_unique=float(np.mean(np.array(nroots) == 1)),
                                     T_rel_bias={n_: float(rel[:, i, j, c].mean() - 1) for n_, (i, j, c) in
                                                 dict(EEE=(0, 0, 0), EIE=(0, 1, 0), IIE=(1, 1, 0), EEI=(0, 0, 1), EII=(0, 1, 1), III=(1, 1, 1)).items()},
                                     T_rel_sd={n_: float(rel[:, i, j, c].std()) for n_, (i, j, c) in
                                               dict(EEE=(0, 0, 0), EIE=(0, 1, 0), IIE=(1, 1, 0), EEI=(0, 0, 1), EII=(0, 1, 1), III=(1, 1, 1)).items()})
        row["sd_shift_lo"] = float(np.nanstd(fstars["lo"] - fstars["ref"]))
        row["sd_shift_hi"] = float(np.nanstd(fstars["hi"] - fstars["ref"]))
        out["designs"].append(row)
        print(label, {k: (round(v["mean_f_star"] - v["true_f_star"], 4), round(v["sd_f_star"], 4), v["coverage_95"]) for k, v in row["by_rate"].items()},
              "sd shifts", round(row["sd_shift_lo"], 4), round(row["sd_shift_hi"], 4), f"{time.time() - tic:.0f}s", flush=True)
    (HERE / "estimator_validation.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
