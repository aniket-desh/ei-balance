# Completed conditional EI algebra

The final derivation is `final_analytics.tex`. It establishes the population
resolvent and PR formulas conditional on the stated nonlinear cavity
assumptions. It also proves the premature-averaging inequalities for a general
nonuniform population variance profile with common leak time, extending the
previous scalar demonstration. No new universality or literature-priority
claim is made.

The central observable is rate covariance after centering each neuron's
stationary temporal mean. It is not Pearson covariance, attractor dimension,
or a communication dimension. At a fixed point PR is undefined. The E–I
energy is the squared Frobenius norm of a cross-covariance block, normalized by
total network size; both E–I orientations contribute to the total energy.

## What was finished

- Conditional mean, gain, Price integral, Hermite residual, static and full
  temporal-kernel self-consistency with every averaging operation specified.
- Directed and shared-input cavity contractions reduced to
  `X = R diag(Q/f) R^dagger`, including the diagonal subtraction and all
  population fractions.
- Total and within-population PR, diagonal-heterogeneity versus pair-correlation
  factorization, population block positivity and cross-share bounds.
- A nonnegative path expansion proving that averaging spectra or gain before
  the required same-label moments raises PR at fixed local laws, for the
  nonuniform population model as well as the scalar case.
- Fixed-aggregate composition and degree-control paths distinguished.
- Coherent EI feedback retained in an exact finite-dimensional Woodbury
  comparison; endogenous-label regression explicitly conditioned on a fixed
  cavity reservoir; exact finite-network balance retains centered recurrent
  mean current.

The proof of the population averaging result concerns integrated zero-lag
energies. It does not order arbitrary unequal-lag maps or complex spectral
entries. It compares approximations to the same local laws, not distinct
self-consistent network ensembles.

## Numerical controls

`population_averaging_controls.py` uses the archived conditional laws and an
independent chunked population quadrature, preserving spectral masses and
same-neuron products. The reference baseline is reproduced to about 1e-16.

| E fraction | Conditional PR/N | Average spectra | Average gain | Both |
|---|---:|---:|---:|---:|
| 0.6 | 0.0298935242 | 0.0665594070 | 0.163923994 | 0.359971202 |
| 0.8 | 0.0343262252 | 0.0764508268 | 0.168013711 | 0.368451604 |

These values and all population energy blocks are in
`population_averaging_controls.json`; every blockwise inequality passes. The
large difference illustrates why replacing distributed gains by their mean
cannot be used to explain the original balanced-network geometry.

`check_final_analytics.py` adds independent constraints, recorded in
`analytics_checks.json`:

| Constraint | Largest error |
|---|---:|
| Price covariance vs direct 2D Gaussian quadrature | 4.1e-14 absolute |
| Full complex cavity equations, 1/2/3/5 populations | 4.0e-15 absolute |
| Separate SciPy discrete Lyapunov solve | 1.8e-15 absolute |
| Population path series vs direct resolvent quadrature | 1.8e-15 relative |
| Uniform-population fraction null | 8.9e-16 absolute |
| Isolated directed-edge phase | 2.8e-17 absolute |
| FFT lag convention vs direct sums | 1.2e-16 absolute |

Only one BLAS thread is used. The small checks do not run network simulations.
The standalone system `pdflatex` entry initially lacked its format; the paper
extraction agent owns the private repaired TeX environment and integrated
manuscript compilation. The failed initial standalone invocation remains in
`tmp/compile.log`, a regenerable local build artifact.

## Assumptions that remain assumptions

Algebra is complete once the conditional one-site and two-site closures are
specified. This does not prove conditional Gaussianity, reservoir decoupling,
uniform response bounds, or a nonlinear stationary limit. Gaussian row
regression is exact for fixed reservoir means independent of the incoming
probe row, not for arbitrary self-consistent full-network means. Degree
constraints require control of coherent population projections in addition
to stability of the mean EI matrix. The finite-rank trace comparison holds
local sources fixed; it cannot bound a full-rank change in nonlinear local laws.

The critical law additionally needs an isolated smooth mode, nonzero projected
curvature, and the appropriate order of limits. A homogeneous biased
operating point already supplies curvature; the three-halves exponent alone
is not evidence for disorder. These unresolved assumptions are the specific
objects tested by the network validation, rather than algebraic gaps hidden
by the final PR formula.

A fresh independent review found the matrix, population geometry, and
averaging arguments sound. Its requested qualification about the fixed
cavity reservoir in Gaussian regression has been incorporated. Remaining
physical assumptions are stated alongside the formulas.
