"""Exhaust finite subsets and independently reconstruct one pilot covariance-estimator result."""
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import chi2, t

from calculate import HERE, PILOT, uniform_readout


def all_subsets(covariance, k):
    m = len(covariance)
    rows = []
    for subset in combinations(range(m), k):
        c = covariance[np.ix_(subset, subset)]
        trace, energy = np.trace(c), np.square(c).sum()
        rows.append([trace, trace**2, energy, trace**2 / energy])
    return np.mean(rows, axis=0)


def subset_identities():
    # Small Gram matrix with heterogeneous variances, including positive/negative off-diagonals.
    rng = np.random.default_rng(828)
    a = rng.normal(size=(8, 5))
    c = a@a.T
    diagonal = np.diag(c)
    pair = (np.square(c).sum()-np.square(diagonal).sum())/(len(c)-1)
    errors = []
    for k in range(1, len(c)+1):
        trace, square, energy, _ = all_subsets(c,k)
        mt, vt, me = uniform_readout(diagonal.mean(), (diagonal**2).mean(), pair, len(c), len(c), k)
        errors.extend([abs(mt-trace), abs(mt**2+vt-square), abs(me-energy)])
        np.testing.assert_allclose([mt, mt**2 + vt, me], [trace,square,energy], rtol=1e-12, atol=1e-12)

    # Two positive-definite 4-cell examples with the same one- and two-site moments.
    uniform = np.full((4,4), .2)
    np.fill_diagonal(uniform, 1.)
    concentrated = np.eye(4)
    concentrated[0,1] = concentrated[1,0] = np.sqrt(6)*.2
    results = []
    for matrix in (uniform, concentrated):
        assert np.linalg.eigvalsh(matrix).min() > 0
        trace, square, energy, mean_ratio = all_subsets(matrix, 2)
        results.append({"mean_trace":trace, "mean_squared_trace":square, "mean_energy":energy,
                        "ratio_of_means": square/energy, "mean_ratio": mean_ratio})
    np.testing.assert_allclose(results[0]["ratio_of_means"],results[1]["ratio_of_means"])
    assert results[1]["mean_ratio"]-results[0]["mean_ratio"] > .01
    return {"all_subsets_of_eight_max_absolute_error": max(errors),
            "four_cell_counterexample": results}


def pilot_block_check():
    # Reconstruct the primary estimator directly from packed lower triangles,
    # avoiding the pilot code's dense matrix multiplication and trace products.
    p = PILOT/"measurements/readout_f0.94_N2000_seed6104.json"
    d = json.loads(p.read_text())
    blocks = p.with_name(d["block_statistics"])
    if not blocks.exists():
        return {"status": "omitted raw blocks unavailable", "expected_file": blocks.name}
    population, draw = 1, 4
    with np.load(blocks) as z:
        means = z["block_mean_rate"][population, draw]
        cov = z["block_cov_lower"][population, draw]
        ids = z["indices"][population, draw]
    assert np.array_equal(ids, d["subset_indices"][population][draw])
    nblocks,k = means.shape
    i, j = np.tril_indices(k)
    weight = np.where(i == j, 1., 2.)
    cov_pairs = {}
    for a,b in combinations(range(nblocks),2):
        dm = means[a] - means[b]
        cov_pairs[a,b] = (cov[a]+cov[b]+dm[i]*dm[j])/2
    trace = np.mean([v[i==j].sum() for v in cov_pairs.values()])
    cross = []
    for a,b,c,h in combinations(range(nblocks),4):
        for p1,p2 in (((a,b),(c,h)),((a,c),(b,h)),((a,h),(b,c))):
            cross.append(np.dot(weight*cov_pairs[p1],cov_pairs[p2]))
    energy = np.mean(cross)
    original = d["readouts"][population][draw]["T8000_gap0"]
    measured = {"trace":trace, "frobenius_energy":energy, "PR": trace**2/energy}
    errors = {key: abs(x-original[key]) for key,x in measured.items()}
    assert max(errors.values()) < 1e-10
    return {"status":"verified", "name":d["name"], "population":"I", "subset_label_zero_based":draw,
            "number_of_four_window_pairings":len(cross), "source_sha256":sha256(blocks.read_bytes()).hexdigest(),
            "values":measured, "absolute_errors":errors}


def power_check():
    """Independently integrate conditional normal tails over the paired sample variance."""
    sizing = json.loads((HERE/"pilot_and_sizing.json").read_text())
    results = {}
    for r in sizing["contrasts"]:
        if r["role"] == "exploratory":
            continue
        n = r["new_seeds"]
        tau = np.sqrt(n)*abs(r["expected_change"])/r["pilot_paired"]["sd"]
        cut = t.ppf(1-r["two_sided_alpha"]/2,n-1)
        def integrand(u):
            threshold = cut*np.sqrt(chi2.ppf(u,n-1)/(n-1))
            return ndtr(tau-threshold)
        correct, err = quad(integrand,0,1,epsabs=1e-12)
        wrong, _ = quad(lambda u: ndtr(-tau-cut*np.sqrt(chi2.ppf(u,n-1)/(n-1))),0,1,epsabs=1e-14)
        np.testing.assert_allclose(correct+wrong,r["planned_power_sd_multipliers"]["1"],atol=1e-10,rtol=0)
        key = f"{r['metric']}_{r['population']}_{r['lower']}"
        results[key] = {"correct_direction":correct, "opposite_direction":wrong,
                        "estimated_quadrature_error":err}
    bounds = {}
    for label, keys in (("both_total_E_flanks",["D_E_over_N_E_0.8","D_E_over_N_E_0.88"]),
                        ("both_pair_directions",["pair_energy_E_0.8","pair_energy_I_0.8"])):
        bounds[label] = max(0.,sum(results[k]["correct_direction"] for k in keys)-1)
    return {"normal_mixture_integrals":results, "joint_correct_direction_union_lower_bounds":bounds}


def main():
    result = {"finite_subset_algebra":subset_identities(), "pilot_packed_blocks":pilot_block_check(),
              "paired_design":power_check()}
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
