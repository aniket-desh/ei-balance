# Completing the conditional balanced EI paper

This round follows Nischal's original handoff: finish the conditional
analytics, resolve or delimit the fraction discrepancy, complete the
nonuniform degree comparison, and extract a paper with simulation-versus-theory
figures. The [original scope](../archive/research_round/scope/NEXT_ROUND_SCOPE.md) records the agreed comparisons.

## Completed primary work

The conditional EI algebra is complete in `analytics/final_analytics.tex`.
It includes conditional one-site statistics, same-neuron spectral products,
population covariance energies, E/I and whole-network PR, correct lag
conventions, coherent-feedback qualifications, and population averaging
inequalities. Independent algebra checks pass. These are results within a
stated cavity closure, not a nonlinear convergence theorem.

All 64 staged primary network cases completed: 40 composition cases and 24
nonuniform degree cases. No primary case was excluded. Simulation code,
resource controls, exact protocols, source versions, and all outcomes are
retained in `runtime/`.

| Comparison | Parameter-only theory | Measured network mean | Interpretation |
|---|---:|---:|---|
| 60:40, N2000, 12 networks | d=.0298935 | d=.0294174 | -1.6% mean PR error; mean nu/energy errors -.6%/+1.7% |
| 80:20, N2000, 12 networks | d=.0343262 | d=.0331594 | -3.4% mean PR error; mean nu/energy errors -1.2%/+2.9% |
| 60:40, N4000, 4 networks | d=.0298935 | d=.0299729 | +.3% PR error; raw nu/energy +5.1%/+10.5%, with wider uncertainty |
| 80:20, N4000, 4 networks | d=.0343262 | d=.0334910 | -2.4% PR error |
| Degree chi0, N4000, 3 networks | d=.173891 | d=.162277 | -6.7% endpoint error |
| Degree chi2, N4000, 3 networks | d=.016811 | d=.019306 | +14.8% endpoint error |

At N2000, all six pointwise 95% mean-relative-error intervals for nu, energy,
and d fall within approximately +/-15%. This is descriptive, not a simultaneous
confidence statement, a predeclared 15% equivalence test, or a uniform
theoretical error bound. Ingredient amplitudes do not converge monotonically
across the sampled sizes. Full intervals and per-network data are in
`composition/prospective_summary.json`.

The scoped resolution of the original 60:40 comparison is a supported regime
for the leading theory at the tested larger sizes, with a genuine N1000
finite-network/recording limitation retained. It is not a controlled N1000
finite-size repair. The formal first correction moves PR in the wrong
direction for the historical deficit. Selected atypical networks need
additional numerical/time-sampling care. The numerical follow-up is documented
below and must not be replaced by a unique causal attribution.

The completed degree intervention reproduces an 8.41-fold endpoint change at
N4000, versus 10.34 predicted. Earlier quantitative failures, including a
+30% N2000 overdispersed-endpoint error, remain visible. This is a completed
mechanism test with a measured accuracy boundary, rather than uniform
validation of the entire degree curve.

## What the additional diagnostics establish

The normalized population energies and temporal Gram overlap (linear CKA)
agree more closely than some absolute ingredients. They test population
geometry and separate it from amplitude, while all absolute errors remain
reported. E/I dimensions and the exact relation between population overlap
and whole-network PR are derived and measured. These are not communication
dimensions or tests of computational performance.

Across 28 local calibration cases, evaluating the conditional Gaussian
gain formula with measured neuron-level means and temporal variances predicts
population gain moments within about 1%. The population Gaussian/shared-
variance approximation leaves roughly 5-8% bias at N2000. The intermediate
static-label and temporal-variance substitutions are saved separately. These
are calibrated diagnostics, not parameter-only corrections or proof of the
whole Gaussian trajectory law.

All thirteen original model tests pass. Independent checks cover Price's
formula, the population response equations, lag phase, composition null
limits, averaging inequalities, exact subset contractions and cumulants,
temporal Gram identities, streamed recording estimators, and every primary
record's normalization. The primary normalization residual is below 6e-17.

## Numerical checks and remaining small-network limitation

Five selected time-step comparisons completed. Four change PR by approximately
1% or less, including the N4000 fixed-degree comparison. One atypical N1000
80:20 network changes PR by about 24%, with material ingredient changes.
Three additional same-network native-backend recordings at dt .05/.025/.0125
and T16000 give d=.0190851/.0186359/.0170023, still spanning about 12%.
At the same dt=.05 and T8000, the native recording gives d=.0183607,
versus .015601 for the original SciPy action, although their short operator
actions agree within floating precision. Thus the initial 24% difference
cannot be attributed solely to step size. Trajectory phase, recording
convergence, or state dependence remains material in this atypical network.
These are auxiliary cases, not new independent primary network replicates. Four selected N2000 tangent checks have positive
finite-time exponents in every measured block. Other primary networks are
described as fluctuating rather than automatically classified as chaotic.

All 72 covariance-run records completed: 64 primary cases, five original
step checks, and three targeted outlier follow-ups. Four tangent checks also
completed. The maximum observed aggregate resident memory in the network
phases was 2.10 GB. The runs executed in the separate Linux workspace;
macOS execution was not tested. The portable launcher retains CPU/RAM caps
and a SciPy fallback. No simulation or controller session remains running.

The working paper and figures are ready for scientific review. A broad
attractor phase diagram, all finite-size vertices, and full covariance
spectra remain outside the handoff scope.

## Reading order

1. Read the [paper](../paper/main.pdf) or open its [editable source](../paper/main.tex).
2. `analytics/README.md` and `final_analytics.tex`: the conditional construction,
   population geometry, averaging proposition, and checks.
3. `degree/degree_control_derivation.tex` and `predictions.json`: the
   nonuniform degree intervention and exact finite-pool identities.
4. `composition/composition_diagnosis.tex`: the fraction control, historical
   discrepancy accounting, and prospective statistical plan.
5. `runtime/campaign_protocol.json`, source, benchmarks, and resource audit:
   simulation and recording implementation.
6. `review/README.md`: independent checks and material caveats.

Scientific figures and their exact values are in `figures/`,
[paper/figures](../paper/figures), and `degree/`. Network records are the units of
uncertainty; neurons, temporal blocks, and overlapping quadruples are not
independent network replicates.

## Simulation implementation

Dynamics use float64, midpoint integration, and an exact sparse-complement
representation of each strict Dale matrix. The dense population contribution
is retained as a rank-two action. No edges are pruned. An optional native
constant-weight sparse action is checked against SciPy and dense actions;
the portable SciPy action remains available. Float32 is used only for stored
trajectories. Covariance contractions and summary statistics use float64.

The campaign caps workers and BLAS threads, monitors aggregate resident
memory, streams covariance arrays through temporary files, and discards
regenerable raw trajectories after saving sufficient results. The complete
protocol and resource evidence take precedence over this short description.

The first campaign's written protocol specifies B500 with gap0 as the
primary four-window statistic and gap1 as sensitivity. Earlier historical
cohort comparisons used gap1. Both are retained and identified. Selected
longer-block and duration diagnostics test the recording procedure; they do
not prove a uniform stationary limit. The PR ratio remains a ratio estimator.

## Main paper, appendices, and limitations

The main paper centers on the conditional local-law-to-population-geometry
construction, the direction and size of premature averaging errors, and the
degree and composition tests. The appendix contains inspectable cavity
algebra and the estimator identity. Full first-order finite-size calculations,
the second-order inventory, joint fluctuation attempts, gain residual tests,
branch diagnoses, and crossover work remain in the earlier archive folders.

Measured-local-law substitutions, empirical source decompositions, and
population-axis projections are diagnostics. They are never relabeled as
parameter-only predictions. Further finite-size expansions and a new
critical crossover campaign are not prerequisites for the core paper.
