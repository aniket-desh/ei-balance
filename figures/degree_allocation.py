"""Screen the conditional theory at fixed total degree-variance input.

The two incoming blocks of population a share a count dispersion chi_a.
Lambda gives the fraction of the direct quenched variance source allocated
to I targets. The full quenched variance also includes recurrent feedback.
Run this script to regenerate degree_allocation.json. No networks are sampled.
"""

import hashlib
import json
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "src"
sys.path.insert(0, str(SOURCE))
from eicavity.multitype import population_dimension, solve_populations


def main():
    reference_path = ROOT / "experiments/degree/predictions.json"
    reference = json.loads(reference_path.read_text())
    G, J0 = np.array(reference["G"]), np.array(reference["J0"])
    f, m = np.array(reference["fractions"]), np.array(reference["mean_rates"])
    W = G * f
    baseline_sources = f * (W @ (m**2))
    budget = baseline_sources.sum()
    bernoulli = baseline_sources[1] / budget
    points = np.unique(np.r_[np.linspace(0., 1., 9), bernoulli])
    dt, horizon, n_hermite, cutoff = 0.08, 100., 96, 12.
    rows = []
    for share_i in points:
        chi_target = budget * np.array([1-share_i, share_i]) / baseline_sources
        np.testing.assert_allclose(baseline_sources @ chi_target, budget)
        dmft = solve_populations(G, mean_rates=m, fractions=f, dt=dt, horizon=horizon,
                                 degree_dispersion=np.repeat(chi_target[:, None], 2, axis=1))
        result = population_dimension(dmft, n_hermite=n_hermite, max_frequency=cutoff)
        means = np.asarray(result["mean_variances"])
        diag_cost = np.asarray(result["variance_seconds"]) / means**2
        pair = np.asarray(result["pair_covariance_energy"]).diagonal()
        pair_scaled = pair / means**2    # N * mean(C_ij²) / mean(v_i)², i != j
        within = np.asarray(result["within_population_dimension_fractions"])
        np.testing.assert_allclose(1/within, diag_cost + f*pair_scaled, rtol=1e-11)
        if not 0 <= result["response_radius"] < 1:
            raise ValueError("Covariance response does not converge on this branch.")
        if share_i == bernoulli:
            r = next(r for r in reference["rows"] if r["chi"] == 1)["prediction"]
            np.testing.assert_allclose(within, r["within_population_dimension_fractions"], rtol=1e-6)
            np.testing.assert_allclose(result["dimension_fraction"], r["dimension_fraction"], rtol=1e-6)
        rows.append({
            "fraction_of_direct_source_to_I": float(share_i),
            "chi_by_target": chi_target.tolist(),
            "mean_input_offset": dmft.mean_inputs.tolist(),
            "static_input_variance": dmft.static_variances.tolist(),
            "dynamic_input_variance": dmft.dynamic_variances.tolist(),
            "mean_rate_variance_by_population": means.tolist(),
            "diagonal_inverse_dimension": diag_cost.tolist(),
            "scaled_mean_squared_offdiagonal_covariance": pair_scaled.tolist(),
            "d_E": float(within[0]), "d_I": float(within[1]),
            "d_joint": result["dimension_fraction"], "response_radius": result["response_radius"],
            "max_bvp_residual": dmft.bvp_residual, "static_residual": dmft.static_residual,
        })
    sources = [reference_path, Path(__file__)]
    sources += [SOURCE / "eicavity" / n for n in ("multitype.py", "degree.py", "pair.py", "single_site.py")]
    output = {
        "scope": "Exploratory deterministic conditional theory at fixed population size fractions and fixed edge moments. Networks have not tested this path.",
        "G": G.tolist(), "J0": J0.tolist(), "fractions": f.tolist(), "mean_rates": m.tolist(),
        "direct_source_budget": float(budget), "bernoulli_fraction_of_direct_source_to_I": float(bernoulli),
        "control": "Both inputs to each target population share its chi. Sum_a f_a chi_a sum_b W_ab m_b² remains at its chi=1 value. Use predicted finite drive offset h at each point.",
        "observables": "Scaled squared off-diagonal covariance is N times the within-type pair mean of C_ij², divided by the square of the population mean temporal variance. The diagonal cost plus f_a times that scaled pair mean equals 1/d_a.",
        "numerics": {"dt": dt, "horizon": horizon, "n_hermite": n_hermite, "max_frequency": cutoff},
        "source_sha256": {str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "rows": rows,
    }
    (HERE / "degree_allocation.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    for r in rows:
        print(f"{r['fraction_of_direct_source_to_I']:.4f}  {r['d_E']:.6f}  {r['d_I']:.6f}  {r['d_joint']:.6f}")


if __name__ == "__main__":
    main()
