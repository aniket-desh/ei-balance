# Independent checks for the scoped EI round

This folder records a skeptical check of the degree extension, measurement
normalization, and eventual paper claims. It does not add an open-ended
finite-size project to Nischal's handoff.

## Checks completed before inspecting the new networks

The exact row-covariance derivation and finite-reservoir pair contraction are
in `degree_row_contraction.tex`. They expose an assumption behind the statement
that degree dispersion changes the one-site law while leaving the leading
dynamic and pair equations intact. The degree ensemble has within-row edge
correlations, even though different receiving rows remain independent.
Those correlations create low-rank population-projection terms in the pair
source. They are subleading when the covariance energy in these projections
is subextensive. A stable mean connectivity matrix alone does not prove this.

`check_degree_contraction.py` checks the block expansion against the literal
matrix trace at machine precision, then uses independent-row Monte Carlo as
a distinct check. The four Monte Carlo discrepancies are between -0.41 and
2.18 reported Monte Carlo standard errors. The algebra checks pass; a Monte
Carlo error bar is not an exact identity. A constructed coherent-mode example
shows why the additional projection condition is needed even when PR remains
extensive. This example is a counterexample to an insufficient assumption,
not a claimed EI network outcome.

The runtime now measures the coherent projection energy. It should be read
as a diagnostic of this assumption, rather than a complete stationary
finite-network correction or a fitted covariance amplitude.

The streamed runtime estimator's population contractions were independently
reviewed. Its block energy is normalized by total network size. With one
ordered E-I block reported, total energy is E_EE + 2 E_EI + E_II. Diagonal
neuron terms remain in the within-population blocks. The mean-free four-window
formula and the coherent-projection contraction have the correct factors.

`local_moment_estimator_checks.tex` gives compatible cross-window estimates
of static input variance and squared time-averaged gain. A full-record
variance across estimated neuronal means is positively biased by temporal
mean noise, a relevant diagnostic at nominally fixed degree. These identities
were sent to the runner before the new outcomes were inspected.

## Conditions that must remain visible in the paper

- Mean-free covariance energy is unbiased for independent stationary windows.
  Separated windows in one nonlinear trajectory are an approximation to that
  setup. Block and gap stability tests help assess it; they do not prove
  uniform mixing.
- The PR ratio remains biased, even when numerator and energy estimates are
  unbiased. Report both mean network PR and the ratio of ensemble mean
  ingredients. Independent networks, rather than windows or neuron pairs,
  determine network-level uncertainty.
- A gain-only substitution holds source statistics artificially fixed. Local
  sources and gains co-vary in the actual networks. Calibrated substitution
  diagnoses errors but does not become a parameter-only theory prediction.
- Nonzero temporal variance and drift do not establish chaos. The fast runner
  currently measures neither a largest Lyapunov exponent nor all return modes.
  Call these outcomes fluctuating unless a separate diagnostic supports a more
  specific dynamical classification.
- An ensemble match at one size does not prove asymptotic closure, and a
  statistically inconclusive residual is not quantitative agreement. Preserve
  uncertainty and distinguish observed trends from controlled remainder bounds.
- The new prospective cohort is the primary fresh test. The historical sample
  contains the two networks whose unusual values triggered the investigation;
  pooling all histories without stating their selection does not create a new
  prospective experiment.
- The formal first-order correction raises the 60:40 prediction. It has not
  explained the lower corrected historical mean. Its algebra and partial
  checks belong in the archive and appendix with this limitation visible.

New network and manuscript checks will be appended after the prespecified
campaign returns. No new data result is asserted by this initial review.

## Additional focused checks during the campaign

`focused_novelty_check.md` compares the specific claims with Clark et al.
(2023, 2025) and Clark (2026), using primary texts. Same-label spectral and
response products already appear in the 2025 heterogeneity framework. The
2026 typical-quenched linear-equivalence derivation treats i.i.d. couplings
and explicitly does not establish the structured ensemble studied here.
The supported contribution remains the conditional balanced EI specialization,
its population control identities and averaging inequalities, and the direct
tests, without a broad priority claim.

`fixed_degree_quadratic_variance.tex` derives an exact frozen-reservoir
fixed-subset quadratic variance, including coherent row means and the
linear–quadratic covariance. Exhaustive enumeration in
`check_fixed_quadratic_variance.py` agrees within 9e-16. The leading centered
formula confirms the proposed diagonal-source subtraction, but does not
complete the stationary finite-network variance coefficient. The independent
frozen row and zero mixed static/dynamic covariance assumptions must remain
visible. The degree branch's measured cross-fitted covariance between local
mean input and temporal variance is nonzero in the selected nominally
fixed-degree example, so treating them as independent is not an exact repair.

`audit_campaign_records.py` checks every completed network record for control
matrices, theory-selected degree drives, population normalization, symmetry,
and the PR ratio. Its JSON lists which records were complete at audit time;
the script must be rerun after the campaign finishes. This is an integrity
check, not evidence of stationarity or a nonlinear closure theorem.

## Completed nonuniform degree comparison

All 18 prespecified degree runs completed and remain in the record. The
population mechanism is substantial: changing dispersion from zero to two
reduces mean dimension by approximately 7.9-fold at N1000 and 7.1-fold at
N2000, compared with the 10.34-fold conditional prediction. This supports
transfer of the large degree effect to different E/I local laws. It does not
validate the absolute curve uniformly.

At N2000 the chi0 mean is 0.15618 versus 0.17389 predicted, with conventional
95% network-mean interval [0.15068, 0.16167]. The chi2 mean is 0.02190 versus
0.01681 predicted, with interval [0.01761, 0.02619]. Both retain quantitative
residuals under the adopted recording protocol. Three networks per group do
not establish a finite-size scaling law. In particular, the chi2 discrepancy
does not decline monotonically across the two tested sizes.

The raw-rate coherent projection energy fractions are small: roughly
0.3–0.9% across N1000 degree conditions, and 0.08%/0.30% at the N2000
chi0/chi2 endpoints. These assess a specific uniform-population sector.
They are not bounds on all response-weighted collective modes or on every
finite-size coefficient. The frozen quadratic-variance coefficient concerns
a filtered covariance S, so a measured fraction for unfiltered C cannot be
substituted without justification. With an O(1)-norm mean coupling row,
an O(1/N) projection-energy fraction can still contribute to its O(1/N)
variance coefficient; an O(1/N^2) fraction for the same bounded filtered
matrix would suppress that particular term by another power of N.

## Final primary campaign review

The full primary campaign now contains 40 composition networks and 24 degree
networks. The degree extension was allocated after inspecting the initial
endpoint residuals, with fresh outcomes retained at both N4000 endpoints.
It must be described as staged, not as 64 experiments all chosen before the
first pilot outcome.

The primary estimator remains T8000, B500, gap0 in the new campaign.
The older sixteen-network recording audit used gap1 for its tabulated
mean-free values. The final manuscript now states this distinction explicitly.

The forty-network composition aggregate preserves the mean of network PRs,
the ratio of ensemble mean moments, and the exact positive difference
identity as separate quantities. At N2000, twelve networks per composition
give relative mean errors below 3.4% for nu, energy, and PR. The six separate
95% mean-error intervals span an overall range from -12.58% to +14.19%.
This is a descriptive statement about pointwise finite-record intervals;
15% was not a predeclared equivalence margin, and no simultaneous 95%
accuracy guarantee follows. The four-network N4000 60:40 point has PR error
only +0.27% while numerator and energy errors are +5.11% and +10.52%.
It must not be used as evidence of monotone ingredient convergence.

At N4000 the degree chi0 mean PR remains 6.68% low and covariance energy
14.41% high. The energy interval [0.43429, 0.47693] excludes the leading
0.39822 prediction. At chi2 the mean PR is 14.84% high, with a much wider
interval [0.01234, 0.02627]. The completed degree test therefore validates
the large mechanism qualitatively and exposes remaining prefactor errors.
It does not make a uniform quantitative validation across all dispersions.

The normalized E-I energy is an exact Hilbert-Schmidt alignment of the two
time-centered temporal Gram matrices, equivalent to linear CKA. The exact
joint-dimension identity uses variance fractions w_a, not neuron fractions
f_a. Both derivations and their normalization were independently checked.
The empirical normalized allocation and CKA comparisons are secondary,
exploratory diagnostics. Ratios built from mean-free cross-energy estimates
are not exactly unbiased and need not obey every exact covariance bound
at finite recording duration.

The completed primary table and current main-text numerical statements were
checked against their JSON sources. All five auxiliary timestep records have
also completed. Comparing identical recording durations, four PR changes
have magnitude below 1.1%, while the N1000 80:20 seed3001 outlier changes by
+24.34%, with numerator -10.83% and energy -36.06%. Such a difference can
involve chaotic sampling, dynamical branch sensitivity, or timestep bias;
these possibilities have not been distinguished by this comparison alone.
No blanket assertion of negligible timestep dependence is justified.
At the N4000 chi0 check, halving the step changes PR by +1.05%, numerator
by -0.054%, and energy by -1.15%, so this selected stiffness check does not
remove the observed degree prefactor residual. Tangent diagnostics remain
a separate check rather than a classification of every primary realization.

The final record auditor checks all 72 completed run files: 64 primary
networks, five timestep auxiliaries, and three extended outlier diagnostics.
Population normalization errors are below 9e-17, and the independent
joint-dimension identity agrees within 3e-17. Those checks concern numerical
accounting, not stationary validity.

The three additional native-backend trajectories of the atypical N1000
80:20 seed3001 give T16000 PR fractions 0.0190851, 0.0186359, and 0.0170023
at steps 0.05, 0.025, and 0.0125. Their maximum-to-minimum spread is 12.25%.
At the original step 0.05 and T8000, the native backend gives 0.0183607
versus the original SciPy result 0.0156011, a 17.69% change despite action
agreement at roughly machine precision. Thus the original 24.3% half-step
change cannot be uniquely assigned to integration truncation. Arithmetic
phase, record length, timestep, and possible state sensitivity remain
entangled; no converged invariant covariance has been established for this
outlier. All primary and auxiliary outcomes remain visible.

The four selected N2000 tangent calculations have positive largest
finite-time exponents and positive values in all eight successive blocks.
They support a chaotic interpretation of those examples, rather than a
classification of all primary networks. The source-dimension concentration
identity and temporal-Gram geometry appendix were also independently checked;
their exact frozen-Gaussian and common-record premises remain explicit.
