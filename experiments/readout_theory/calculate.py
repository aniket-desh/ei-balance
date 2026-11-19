"""Finite 100-cell readout theory, pilot ratio diagnostics, and paired-study sizing."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, ncf, t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/figures"))
from eicavity.composition import dimension_at_fraction
from sweep_records import composition_moments

PILOT = HERE.parent / "readout_pilot"
FRACTIONS = (.80, .88, .94)
N, K, NEW_SEEDS = 2000, 100, 624


def uniform_readout(mean, second, pair, population, network_size, k):
    """Exact subset expectations for a fixed covariance supplied via its moments."""
    variance = second - mean**2
    trace = k * mean
    var_trace = k * (population - k) / (population - 1) * variance
    energy = k * (second + (k - 1) * pair / network_size)
    return trace, var_trace, energy


def theory_row(f, tensor, mean, second):
    fractions = np.array([f, 1-f])
    counts = np.rint(N * fractions).astype(int)
    assert np.all(counts >= K) and np.allclose(counts, N*fractions)
    prediction = dimension_at_fraction(tensor, mean, second, fractions)
    pair = np.diag(prediction["pair_covariance_energy"])
    trace, var_trace, energy = uniform_readout(mean, second, pair, counts, N, K)
    within = np.array(prediction["within_population_dimension_fractions"])
    return {"fraction_e": float(f), "population_counts": counts.tolist(),
            "readout_PR": ((trace**2 + var_trace)/energy).tolist(),
            "readout_PR_squared_mean_trace": (trace**2/energy).tolist(),
            "expected_trace": trace.tolist(), "subset_trace_variance": var_trace.tolist(),
            "expected_energy": energy.tolist(), "pair_energy": pair.tolist(),
            "within_population_dimension_fraction": within.tolist(),
            "D_E_over_N": float(f*within[0]), "joint_dimension_fraction": prediction["dimension_fraction"]}


def moment_bias(readouts):
    x = np.array([r["trace"]**2 for r in readouts])
    y = np.array([r["frobenius_energy"] for r in readouts])
    assert np.all(y > 0)
    measured = x/y
    np.testing.assert_allclose(measured, [r["PR"] for r in readouts], atol=1e-12)
    mx, my = x.mean(), y.mean()
    dx, dy = x-mx, y-my
    ratio = mx/my
    bias = measured.mean() - ratio
    cov = np.mean(dx*dy)
    variance = np.mean(dy**2)
    # Expanding 1/y about 1/my leaves this exact third-order remainder.
    quadratic = (ratio * variance - cov) / my**2
    remainder = np.mean((dx-ratio*dy)*dy**2 / (my**2 * y))
    np.testing.assert_allclose(quadratic+remainder, bias, atol=4e-13)
    np.testing.assert_allclose(bias, -np.mean((measured-measured.mean())*dy)/my, atol=4e-13)
    return {"readout_PR": float(measured.mean()), "ratio_of_means": float(ratio),
            "bias": float(bias), "quadratic": float(quadratic), "remainder": float(remainder),
            "squared_mean_trace_over_energy": float(np.mean(np.sqrt(x))**2/my),
            "variance_energy": float(variance), "covariance_squared_trace_energy": float(cov)}


def pilot_rows(protocol, sources):
    rows = []
    for job in protocol["jobs"]:
        measurement = PILOT / "measurements" / (job["name"]+".json")
        dense = job["full_ensemble_record"]
        full = (HERE.parent/"dense_sweeps/runs"/(dense+".json") if dense
                else PILOT/"runs"/(job["name"]+".json"))
        sources.extend((measurement, full))
        data, network = [json.loads(p.read_text()) for p in (measurement, full)]
        cfg = job["config"]
        assert data["connectivity_sha256"] == network["connectivity_sha256"]
        assert data["config"] == network["job"]["config"] == cfg
        assert np.array_equal(data["subset_indices"], job["subsets"])
        est = next(e for e in network["estimators"] if e["block_duration"] == 500)
        prefix = next(z for z in est["prefixes"] if z["duration"] == 8000)
        gap = next(z for z in prefix["gaps"] if z["empty_blocks"] == 0)
        frac = cfg["fraction_e"]
        freq = np.array([frac, 1-frac])
        counts = np.rint(N * freq).astype(int)
        energy = np.diag(gap["population_energy"])
        diagonal = np.array(gap["diagonal_population_energy"])
        pair = N**2 * (energy-diagonal)/(counts*(counts-1))
        assert data["config"]["n"] == N and len(data["readouts"]) == 2
        moments = []
        for readouts in data["readouts"]:
            assert len(readouts) == protocol["draws_per_class"]
            moments.append(moment_bias([r["T8000_gap0"] for r in readouts]))
        full_E = (frac * gap["nu_by_population"][0])**2 / energy[0]
        rows.append({"fraction_e": frac, "seed": cfg["seed"], "subsets": moments,
                     "readout_PR": [v["readout_PR"] for v in moments], "pair_energy": pair.tolist(),
                     "D_E_over_N": float(full_E)})
    return rows


def power(n, effect, sd, alpha):
    """Marginal two-sided paired t power assuming normal differences and fixed SD."""
    cut = t.ppf(1-alpha/2, n-1)
    nc = np.sqrt(n)*abs(effect)/sd
    # The squared noncentral t has a noncentral F distribution with 1, n-1 df.
    return float(ncf.sf(cut**2, 1, n-1, nc**2)) if nc else alpha


def required_n(effect, sd, target, alpha):
    lower, upper = 1, 4
    while power(upper, effect, sd, alpha) < target:
        lower, upper = upper, 2*upper
    while upper - lower > 1:
        mid = (lower+upper)//2
        if power(mid, effect, sd, alpha) >= target:
            upper = mid
        else:
            lower = mid
    return upper


def summarize(x):
    x = np.array(x)
    n = len(x)
    half = t.ppf(.975, n-1) * x.std(ddof=1)/np.sqrt(n)
    return {"values": x.tolist(), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "ci95": [float(x.mean()-half), float(x.mean()+half)]}


def size_contrasts(anchors, rows):
    by_fraction = {f: {r["seed"]: r for r in rows if r["fraction_e"] == f} for f in FRACTIONS}
    seeds = sorted(by_fraction[FRACTIONS[0]])
    assert len(seeds) == 4 and all(sorted(d) == seeds for d in by_fraction.values())
    endpoints = [("readout_PR", 0), ("readout_PR", 1), ("pair_energy", 0),
                 ("pair_energy", 1), ("D_E_over_N", None)]
    def value(row, key, ix):
        return row[key] if ix is None else row[key][ix]
    contrasts = []
    for left, right in zip(anchors[:-1], anchors[1:]):
        lo, hi = left["fraction_e"], right["fraction_e"]
        for key, ix in endpoints:
            effect = value(right,key,ix)-value(left,key,ix)
            differences = summarize([value(by_fraction[hi][s],key,ix) - value(by_fraction[lo][s],key,ix)
                                     for s in seeds])
            sd = differences["sd"]
            df = len(seeds)-1
            sd95 = np.sqrt(df/chi2.ppf([.975,.025], df)) * sd
            primary = lo == .8 and key == "readout_PR" and ix == 0
            secondary = (lo == .8 and not primary) or key == "D_E_over_N"
            alpha = .05/5 if secondary else .05
            result = {"lower": lo, "upper": hi, "metric": key,
                      "population": "E" if ix is None else ("E","I")[ix],
                      "role": "primary" if primary else ("secondary" if secondary else "exploratory"),
                      "two_sided_alpha": alpha,
                      "expected_change": effect, "pilot_seeds": seeds, "pilot_paired": differences,
                      "pilot_sd_ci95_normal": sd95.tolist(),
                      "degrees_of_freedom_pilot": df, "n80": required_n(effect,sd,.8,alpha),
                      "n90": required_n(effect,sd,.9,alpha),
                      "new_seeds": NEW_SEEDS, "planned_95_half_width": t.ppf(.975,NEW_SEEDS-1)*sd/np.sqrt(NEW_SEEDS),
                      "planned_half_width_at_design_alpha": t.ppf(1-alpha/2,NEW_SEEDS-1)*sd/np.sqrt(NEW_SEEDS),
                      "planned_80pct_minimum_detectable_change": brentq(lambda z: power(NEW_SEEDS,z,sd,alpha)-.8, 0, sd),
                      "planned_power_sd_multipliers": {str(scale): power(NEW_SEEDS,effect,sd*scale,alpha)
                                                       for scale in (1,1.5,2)}}
            if key == "readout_PR":
                result["one_dimension_n80"] = required_n(1.,sd,.8,alpha)
                result["squared_mean_trace_predicted_change"] = right["readout_PR_squared_mean_trace"][ix]-left["readout_PR_squared_mean_trace"][ix]
                result["pilot_bias_change"] = summarize([by_fraction[hi][s]["subsets"][ix]["bias"]-
                                                         by_fraction[lo][s]["subsets"][ix]["bias"] for s in seeds])
            contrasts.append(result)
    return contrasts


def main():
    protocol_file = PILOT/"protocol.json"
    protocol = json.loads(protocol_file.read_text())
    assert protocol["readout_size"] == K
    sources = [protocol_file, HERE.parent/"degree/chi1_one_site.npz", HERE.parent/"degree/theory_curve.json"]
    tensor, mean, second = composition_moments()
    # One row per reachable integer population size on the plotted composition interval.
    curve = [theory_row(e/N,tensor,mean,second) for e in range(N//2, N-K+1)]
    anchors = [next(r for r in curve if r["fraction_e"] == f) for f in FRACTIONS]
    coefficient = tensor[[0,1],[0,1],:].copy()
    coefficient[np.diag_indices(2)] -= second
    root = np.sqrt(coefficient)
    a,b = tensor[0,0]
    theory = {"N": N, "K": K, "population_order": ["E","I"],
              "readout_PR_definition": "Conditional ratio E_subset[trace^2]/E_subset[energy] at the two-site moment predictions.",
              "diagonal_mean": mean.tolist(), "diagonal_second": second.tolist(),
              "pair_coefficients": coefficient.tolist(),
              "Paa_minimum_fraction_e": (root[:,0]/root.sum(axis=1)).tolist(),
              "D_E_over_N_peak_fraction": np.sqrt(a)/(np.sqrt(a)+np.sqrt(b)),
              "sources_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources[1:]},
              "anchors": anchors, "curve": curve}
    (HERE/"predictions.json").write_text(json.dumps(theory,separators=(",",":"),allow_nan=False)+"\n")
    rows = pilot_rows(protocol, sources)
    reference = {"new_seeds": list(range(20001,20001+NEW_SEEDS)), "target_n": NEW_SEEDS,
                 "primary": "E 100-cell PR, f_E=.88 minus .80, paired two-sided .05.",
                 "pilot_inputs_sha256": {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                 "numpy_version": np.__version__, "pilot": rows, "contrasts": size_contrasts(anchors,rows)}
    (HERE/"pilot_and_sizing.json").write_text(json.dumps(reference,indent=2,allow_nan=False)+"\n")
    print(f"{len(curve)} theory settings, {len(rows)} pilot networks, {len(reference['contrasts'])} paired decisions")


if __name__ == "__main__":
    main()
