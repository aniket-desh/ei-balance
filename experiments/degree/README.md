# Nonuniform EI degree test

This is the degree intervention requested in the original EI handoff. The
earlier degree-network simulations used uniform synaptic variance; this one
uses different E and I local laws:
\[
G=\begin{pmatrix}16&9\\25&12\end{pmatrix},\quad
J^0=\begin{pmatrix}1&-6\\2&-10\end{pmatrix},\quad f=(.8,.2).
\]
The prescribed large balance drive is fixed. The order-one external offset
is set to the parameter-only prediction \(h(\chi)\) before outcomes are
observed. Thus the full finite-network external input also varies along
the specified path. No measured covariance amplitude, PR, or mean rate is
used to retune a network.

The leading theory predicts PR/N of 0.17389, 0.05594, 0.03433, and 0.01681 at
degree dispersion 0, 0.5, 1, and 2. Its 10.34-fold endpoint change combines
a broader distribution of single-neuron temporal variances with greater
recurrent correlation amplification. Common covariance rescaling alone
would leave PR unchanged.

The completed24-network comparison reproduces the large degree effect. At
N4000, the measured endpoint means are 0.16228 and 0.01931, an 8.41-fold
change. They differ from the predictions by -6.7% and +14.8%, respectively.
The original N2000 chi2 mean error of +30% remains visible. These results
establish the effect while delimiting the quantitative claim; they do not
show uniform finite-size accuracy. See findings.md and comparison.json.

## Files and reproduction

- degree_control_derivation.tex: microscopic count control, exact finite-pool
  static source, conditional one-site equations, block energy normalization,
  finite rounding, and the regular static susceptibility.
- predict_degree.py, predictions.json, chi*_one_site.npz: parameter-only
  predictions and local conditional kernels.
- check_degree_theory.py, theory_checks.json: exact finite-pool enumeration
  and refined deterministic quadrature.
- rounding_diagnostic.py/json: sensitivity to the finite count dispersion
  while reimposing target means. This changes h and is explicitly a diagnostic.
- static_susceptibility.py/json: small-dispersion derivative and direct checks.
- degree_curve.py, theory_curve.json: independently computed continuous
  parameter curve for figures.
- planned_protocol.json: initial proposal, including burn1000.
- adopted_protocol.json: final common campaign with burn500; the sole-runner
  prospective protocol remains authoritative.
- adopted_protocol_gap1_editorial_draft.json: retained intermediate editorial
  draft. Gap0 is the actual prospective primary; gap1 is a sensitivity check.
- aggregate_degree.py: aggregate every prescribed outcome, report network
  uncertainty, and create degree_validation.pdf/png/svg after all degree
  records are complete.
- comparison.json: current cohort completion status and comparisons.
- findings.md, figure_caption.txt: completed24 findings and exact figure scope.
- initial_18/: complete initial test, including its quantitative failures,
  preserved before the subsequent N4000 endpoint extension.
- degree_third_cumulant_audit.tex, finite_pool_third_cumulant.py/json: exact
  within-row cubic-moment audit; these identities do not close the stationary
  finite-size theory.
- local_variance_feedback.tex and local_variance_feedback_diagnostic.py/json:
  cross-window identities and a measured-statistic explanation of finite-N
  fixed-count heterogeneity, explicitly a calibrated diagnostic.

From the workspace root, run a deterministic calculation with

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/degree/predict_degree.py

Run the other scripts the same way. Network simulations are run only by the
shared resource-limited campaign under experiments/runtime/; launching a
second campaign would duplicate work and memory. The simulation optimization
benchmark and memory audit are saved there.

## Prespecified network comparison

The campaign samples three fresh connectivity draws at each of four
dispersions for N1000 and at both endpoints for N2000. Every outcome is kept.
After the initial18 test exposed quantitative endpoint errors, a staged
N4000 endpoint extension with six further networks was declared before its
outcomes. The extension is specified in the runtime degree-extension protocol;
it is not described retrospectively as part of the original18-network plan.
Primary measurements use T8000, covariance blocks B500, and the mean-free
four-window estimator at gap0. Gap1 and T4000 are checked for every network;
B1000 is checked on the selected endpoint seed4001 examples at each size.
Independent connectivity realizations determine the error bars. Temporal
blocks, neurons, and covariance pairs do not supply additional independent
network replicates.

The exact finite count dispersion is saved for each block. Requested chi0
need not mean exactly fixed counts: stochastic rounding produces residual
dispersion up to 0.0128 in this N1000 profile. Re-solving the conditional
theory with those expected dispersions while reimposing target means changes
PR/N by less than 0.74%. The comparison does not silently use this retargeted
diagnostic as the primary prediction.

Population energies use the normalization
\[
\mathcal B_{ab}=N^{-1}\sum_{i\in a,j\in b}C_{ij}^2,\qquad
\mathcal E=\mathcal B_{EE}+2\mathcal B_{EI}+\mathcal B_{II}.
\]
Their meaning differs from PR of a rectangular cross-population block.
The main figure shows total dimension, local input statistics, and all three
energy contributions. Input mean variance estimated from a finite trajectory
also contains temporal sampling noise; near chi0 this is material and is
reported explicitly. Cross-block input-mean diagnostics, when supplied by
the campaign, separate some of that term under the same mixing assumptions.
The corrected local diagnostics cover14 of24 records, including all six
N2000 and all six N4000 endpoint records plus two selected N1000 examples.
Hollow chi1 symbols in the figure show separately identified composition
cohorts with the exact same Bernoulli law and protocol. Those networks are
not counted or pooled as additional degree replicates.

## What the tests establish

The exact finite-pool enumeration checks edge covariance and input variance
to about machine precision at six count dispersions. Refining time step,
horizon, Hermite quadrature, and spectral cutoff changes the four main
deterministic PR predictions by at most 3.3e-8 in relative value. Small-chi
static susceptibility approaches its predicted derivative linearly in chi.

Those checks validate the numerical conditional calculation and the
connectivity control. Actual network agreement or failure must be judged
from the completed prospective cohort in comparison.json. The pair closure
also needs the low-rank response conditions derived by the independent
finite-pool audit; coherent stability alone does not prove them. No complete
finite-N convergence theorem or broad attractor probability is claimed.
