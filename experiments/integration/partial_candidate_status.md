# Fixed-degree variance source: an incomplete diagnostic

The nominally fixed-degree networks develop finite-N variability of the
neurons' temporal input variances and temporal mean rates. This suggests a
specific source of static operating-point disorder even when the limiting
static variance is zero. We investigated this source without fitting the
measured participation ratio.

For a frozen reservoir, let S be its rate covariance after filtering both
time arguments by the common leak. An independent incoming row gives
q_i = J_i S J_i^T. Fixed-count sampling removes the random diagonal source
when S has a constant leading diagonal within each population. The partial
calculation subtracts that diagonal from the filtered population covariance
energy, then propagates the remaining variance through the static response.
The exact finite-subset derivation and enumeration checks are in
`../review/fixed_degree_quadratic_variance.tex`.

At N1000 the parameter-only frozen-row candidate gives Var(q_E), Var(q_I)
approximately .1219, .2868, and static variances .0242, .0371 before adding
rounding. In the selected degree seed4001 example, measured values are about
.2076, .5473 and .0607, .0965. The candidate is therefore retained as a partial
source calculation and rejected as a complete quantitative correction.

The missing mixed term is explicit. Expanding the conditional mean gives

    delta m = alpha delta mu + (beta/2) delta q + ...,
    Var(m) = alpha^2 Delta + alpha beta Cov(mu,q)
             + (beta^2/4) Var(q) + ... .

The selected network has nonzero cross-window Cov(mu,q), approximately
.0196, .0557. Discarding this term is not justified by its mean vanishing in
an independent frozen-row model. Reinserting a probe into an endogenous
network changes the source dependence at the order being investigated.

Coherent feedback also needs its own contraction. For J_i = bar j_i + delta J_i,
the variance contains a linear-source term

    4 bar j_i^T S Cov(delta J_i) S bar j_i,

as well as a linear–quadratic third-moment term when the row is not symmetric.
A small population-axis fraction of the unfiltered rate covariance is not
itself a bound on these terms. The filtered S projection and the N scaling
must be controlled.

One normalization error in the exploratory reasoning was corrected before
adopting any result: an individual mean incoming row has O(1) Euclidean norm;
the full mean connectivity matrix has O(sqrt(N)) operator norm. The exact
quadratic-variance note uses the correct row normalization. This candidate
does not establish a repaired 60:40 prediction, a complete O(1/N) stationary
coefficient, or a new prerequisite for the paper.
