# Full-cohort duration and centering audit

The prospectively specified duration campaign completed **all sixteen** existing
60:40, N1000 Bernoulli connectivity realizations in455 seconds with four
single-thread workers. Each network uses the original generator, dt=.05,
burn500, the same initial condition, sample_dt=.25, and a record of16000.
No realization was selected or excluded based on agreement. Seed101 reuses
the independently completed extension; every other network was rerun.

The first4000-time primary numerator, cross energy and PR of **every seed**
reproduce the supplied external record to8.9e-16 maximum absolute error. This
verifies all sixteen original Bernoulli records in this cohort, not the other
compositions or moment-controlled ensembles. All compact per-network outputs
and final statuses are retained in `cohort/`; raw temporary
trajectory arrays were removed after successful analysis, except seed101's
shared temporary trajectory used for earlier diagnostics.

## Duration alone is not the complete control

The standard eight-block estimate changes its block duration when the total
record grows. At fixed block duration, more data do not remove own-block mean
subtraction bias. All entries below use the same sixteen networks. SE and
95% Student-t intervals use connectivity realizations as independent units,
and include their remaining finite-record variation.

| Estimator | T | Mean d | SE | 95% t interval |
|---|---:|---:|---:|---|
| Standard8 blocks (B500) |4000|.02728095|.00168791|[.0236833,.0308786]|
| Standard8 blocks (B1000)|8000|.02662055|.00164854|[.0231068,.0301343]|
| Standard8 blocks (B2000)|16000|.02604132|.00164307|[.0225392,.0295434]|
| Standard fixed B500|8000|.02742408|.00171897|[.0237602,.0310880]|
| Standard fixed B500|16000|.02742703|.00171245|[.0237770,.0310770]|
| Mean-free four-window B500|4000|.02596408|.00163133|[.0224870,.0294412]|
| Mean-free four-window B500|8000|.02589123|.00162604|[.0224254,.0293571]|
| Mean-free four-window B500|16000|.02567848|.00162718|[.0222102,.0291467]|
| Mean-free four-window B1000|16000|.02568148|.00162576|[.0222163,.0291467]|
| Mean-free four-window B2000|16000|.02560045|.00159963|[.0221909,.0290100]|

Mean-free estimates use all three pairings of four distinct windows with at
least one whole unused block between each consecutive chosen window. Their
covariance energy is unbiased if the four stationary windows are independent;
the PR ratio need not be exactly unbiased. Overlapping quadruples are not
treated as independent observations or used to inflate the sample size.

## Specific source of the shift

At T16000, paired within-network differences between own-centered and
mean-free ratios are:

| B | Mean difference | Paired network SE | B times mean difference |
|---:|---:|---:|---:|
|500|.00174854|.00024473|.8743|
|1000|.00086483|.00014056|.8648|
|2000|.00044087|.00009283|.8817|

This nearly1/B behavior is the expectation from the derived centering bias.
It is a temporal-estimator effect, not an N^-1/2 physical correction.
The standard8 estimate decreases by .00123963 +/- .00026006 paired SE
between T4000 andT16000. At fixedB500 the corresponding change is only
-.00014608 +/- .00021710, consistent with stable biased estimation.

The mean-free ratios are stable under the available gap and block controls.
At B500/T16000, changing the minimum empty gap from500 to0 changes cohort
mean d by -3.81e-6 +/-5.73e-6 paired SE; changing it from500 to1000 changes
mean d by +4.93e-6 +/-5.87e-6. Changing B500 to1000 changes mean d by
+2.99e-6 +/-1.34e-5. These are useful controls at this coupling point,
not uniform guarantees of independence or stationarity near a transition.

The leading theory predicts a covariance-energy centering time6.946 and a
trace-centering time4.232 at60:40. Its asymptotic B500/T4000 calculation
therefore predicts approximately5.1% upward PR bias when ratio noise is
ignored. The observed same-record average standard/mean-free shift is of
similar size. This is not a precise finite-network prediction: the actual
cohort's effective energy-centering scale is larger, about9, and leading
physical numerator/energy errors can partially cancel in the ratio.

## Consequence for the paper

The earlier claim that the ensemble mean agrees with the leading/formal
first-order value must be qualified. The longer-record mean-free full-cohort
mean .0256785 has95% t interval ending at .0291467, below leading
d0=.0298935 and further below the formal first-order .0308713. This is
about14% below leading and17% below the first-order value. The finite sample,
residual temporal-ratio bias, and adopted closure assumptions preclude a
definitive infinite-time population statement, but the old short-record
agreement cannot be used as strong evidence of quantitative accuracy.

The two originally low networks remain the two lowest after the repair:
d=.0135124 and .0139611. The newly added fourteen give
.02738445 +/- .00130383 SE, with95% interval [.0245677,.0302012]. Thus
their atypicality still matters; it does not eliminate the remaining
finite-N ensemble bias. Both facts must appear together.

At mean-free B500/T16000 the cohort mean moments are
nu=.1440788 and E=.8551616. Their ratio is .0242746, while the mean of
network ratios is .0256785. The difference is+.0014039 and Corr(nu,E)=.6553.
The joint moment distribution therefore matters; it is not interchangeable
with a plug-in ratio of ensemble means. Delete-window temporal sensitivities
are recorded per network but are not a proven variance decomposition.

## What is and is not resolved

We have identified a concrete finite-record bias, derived its exact structure,
constructed an independent-copy repair, validated that repair on a Gaussian
test, and checked it on the entire original sixteen-network cohort with
matched record and estimator controls. This resolves the centering artifact
at the tested coupling point to substantially better precision than the
original method's several-percent shift.

We have not derived a uniform T(N,distance-from-onset) bound, completed a
finite-record theory for all ratio fluctuations, or established the full
critical sampling campaign. The theoretical long pair-correlation time is
not automatically the time governing centering bias or estimator variance.
The exact four-time gap term and critical collective correlations still
require separate control. None of these results justify adding one fitted
finite-record counterterm to every network or to all parameter regimes.
