# Finite-count particle mean field

The particle closure recovers part of the operating-point spread that Gaussian theory misses at fixed in-degree. It does not systematically improve held-out predictions of gain or participation ratio. At χ = 1 and 2, allowing each cell its measured temporal variance substantially improves cross-half gain predictions. That result motivates solving for the joint distribution of operating point and temporal variance before replacing the existing theory curves.

## Held-out predictions

Each dispersion has three independent networks of 2,000 cells with seeds 5101–5103 and three independent particle pools per resolution with seeds 8101–8103. The same seed labels recur across settings. The following entries give E-only, I-only and joint PR fractions, each normalized by the number of cells in its readout. Simulation entries are means of per-network estimates. The particle entries average three separate 400-particle-per-type solves.

| χ | Measured E / I / joint | Gaussian E / I / joint | Particle E / I / joint | Particle joint range |
|---:|---|---|---|---|
| 0 | .1760 / .5378 / .1540 | .1993 / .5717 / .1739 | .1983 / .5693 / .1730 | .1728–.1733 |
| 1 | .0398 / .1480 / .0339 | .0406 / .1477 / .0343 | .0348 / .1292 / .0294 | .0225–.0333 |
| 2 | .0223 / .0852 / .0187 | .0201 / .0778 / .0168 | .0209 / .0777 / .0174 | .0122–.0233 |

At χ = 0 the mean absolute error across the 10th, 25th, 50th, 75th and 90th percentiles of the mean input falls from .141 to .085 for E and from .179 to .107 for I. These errors compare the mean predicted quantile with the mean observed quantile across the three networks. The predicted static variances are .0052 and .0086, compared with measured cross-block values of .0270 and .0412. At χ = 1 the mean quantile errors change from .059/.072 to .053/.104, and at χ = 2 they increase from .085/.102 to .205/.182. At χ = 1 the predicted mean squared gain is .0581/.0429, compared with .0522/.0401 in the networks and .0579/.0415 in Gaussian theory. The finite-count model still overestimates E gain at all three dispersions.

The nine held-out networks come from the [dense sweep](../dense_sweeps). Quantiles use each neuron's full 8,000-unit time average. Mean squared gain and static variance use distinct 500-unit blocks. PR uses the four-window covariance-energy estimator with gap zero. For class a the measured fraction is fₐνₐ²/Eₐₐ because Eₐₐ is normalized by total N. Finite networks used the archived degree-specific drive. Their mean E/I rates across seeds were .301/.302, .293/.295 and .306/.304 at χ = 0, 1 and 2. The predictions use a target of .3 throughout and do not ingest a measured rate, gain or kernel.

## Equations and finite source pools

The physical source pools have 1,600 E neurons and 400 I neurons. Each population is represented by P = 100, 200 or 400 sorted, equal-weight input offsets η. Each particle's conditional rate is mᵦⱼ = 2Φ[(hᵦ + ηᵦⱼ)/√(1 + qᵦ)], where hᵦ is chosen so the population mean rate equals .3. G and J⁰ come from the archived benchmark. The edge probability is pₐᵦ = (J⁰ₐᵦ)²/[(J⁰ₐᵦ)²+Gₐᵦ].

For each target particle and source type we draw the incoming count K with the simulation's own degree sampler. We choose exactly K physical sources without replacement. In the empirical representation each rate occurs Nᵦ/P times, so one multivariate hypergeometric draw gives the multiplicities Hᵢⱼ of all particle labels. The signed synaptic input is ∑ᵦⱼ J⁰ₐᵦ Hᵢⱼmᵦⱼ/(pₐᵦ√N). We subtract the within-type draw mean, sort the resulting offsets, and relax 45% toward them at each iteration. The offset h absorbs the mean input needed to meet the target rate. Keeping a pool's multiplicities fixed between iterations makes the residual a criterion for one sampled map. Independent pool seeds measure variation across maps.

For a fixed empirical distribution of μᵦⱼ = hᵦ + ηᵦⱼ, we solve the same temporal boundary problem as the original theory, with Kₐ″ = Kₐ − ∑ᵦGₐᵦfᵦ C̄ᵦ(Kᵦ), Kₐ′(0) = 0, Kₐ(100) = 0 and qₐ = Kₐ(0). Here C̄ᵦ(0) = 0. Price's formula gives its derivative

```text
C̄ᵦ′(c) = 2 / [π √((1+qᵦ)²−c²)] × P⁻¹ ∑ⱼ exp[−μᵦⱼ² / (1+qᵦ+c)].
```

We integrate this expression after substituting c = (1+qᵦ) sin θ. The solver adjusts q and both rate-constrained offsets and recomputes the full temporal kernels on each static iteration. Archived Gaussian solutions supply initial guesses only. For an admitted positive branch, the original two-site cavity integral takes the empirical conditional spectra, gains and equal particle weights and returns the E, I and joint dimensions. This retains population-shared temporal Gaussian noise and the leading-order pair response. The finite neighbor sums correct the static row law. They leave cell-specific temporal variance, feedback from shared loops and finite-N corrections to the pair response open.

## Numerical checks and pool variation

All nine P = 400 cases converged. Across the full grid, 26 of 27 cases converged in 61–417 iterations. At χ = 1, P = 100, seed 8103, the temporal collocation Jacobian became singular at iteration 38 with q near 10⁻⁹. We report this as an unresolved positive branch and omit its PR. At χ = 1 the mean joint fractions across successful pools were .0111, .0205 and .0294 at P = 100, 200 and 400, respectively. At χ = 2 they were .0157, .0103 and .0174. Three seeds and these resolutions leave substantial uncertainty in the limiting particle prediction.

* **Finite source law.** On 30,000 independent rows in an 80-member pool with p = .25, χ = 0 gave exactly 20 contacts, so a constant source had zero row-to-row variation. Across χ = 0, 1 and 2, empirical synaptic-sum variances were within 1.4 Monte Carlo standard errors of the exact variance Mp(1−p)[(M−χ_eff) Var(m)/(M−1) + χ_eff mean(m)²]. At χ = 1 this reduces to the independent Bernoulli variance Mp(1−p) mean(m²). The actual χ = 0 physical pools need stochastic rounding in three blocks, with effective dispersions up to .0032.
* **Shared kernel.** An exactly homogeneous source reproduced the χ = 0 Gaussian dynamic variances within 9.1 × 10⁻⁷. Direct evaluation of the original quadrature agreed with the empirical Price interpolation to 7.4 × 10⁻¹⁴ on a separate heterogeneous example. On accepted solves the maximum static residual was below 2 × 10⁻⁵, the off-mesh temporal-equation error below 7.7 × 10⁻⁶, the boundary error below 3 × 10⁻¹² and the target-rate error below 6.1 × 10⁻¹⁵.
* **Two-site integration.** All 26 solutions passed the response-radius, spectral-mass and temporal-tail checks. The largest radius was .9624 and the largest relative kernel at lag 80 was 6.7 × 10⁻⁷. For one P = 400, χ = 1 pool, refining the FFT lag step from .10 to .08 changed the three fractions by at most 2.2 × 10⁻⁹ at fixed frequency cutoff 12. This checks lag sampling for that pool. Both transforms use a horizon of 100 and frequency spacing π/100.

## Cell-specific temporal variance

The shared-kernel model assigns every neuron in one population the same q. To test the effect of that assumption separately, we took each cell's empirical mean μ from the first half of a held-out recording, computed its Gaussian gain with either that half's population-average q or its cell-specific q, and predicted the cell's measured gain on the other half. We then exchanged halves. This diagnostic uses measured inputs and does not contribute to the parameter-only results above.

| χ | E gain RMSE, shared → cell q | I gain RMSE, shared → cell q | Opposite-half mean–q rank correlation, E / I |
|---:|---:|---:|---:|
| 0 | .0159 → .0155 | .0133 → .0132 | .106 / .024 |
| 1 | .0206 → .0129 | .0179 → .0119 | .424 / .279 |
| 2 | .0201 → .0112 | .0192 → .0113 | .407 / .285 |

Cell-specific q reduced gain error in all 18 network-by-type comparisons. The effect is largest at χ = 1 and 2. Predicting the joint distribution of μ and q would test whether the improvement survives when q must be inferred from connectivity.

## Reproduction

Run `plot.py` from the command below to redraw the figure from the saved validation and particle results. To repeat the held-out comparison in `evaluate.py`, first restore or regenerate the nine per-neuron arrays from the [degree sweep](../dense_sweeps).

```sh
python3 -B experiments/particle_pilot/particle.py
python3 -B experiments/particle_pilot/evaluate.py
python3 -B experiments/particle_pilot/plot.py
```

The 27 solves took 45 seconds of summed case runtime locally using one numerical thread, with a 301 MiB process peak under a 1.5 GiB watchdog. Each [run](runs) retains its seed, numerical residuals or failure, iteration history and checksummed solver version. Successful solves include the empirical inputs, rates, gains, count samples and kernels. The [validation record](validation.json) contains all per-network measurements, input hashes, gain diagnostics and pool ranges at all resolutions. The [source snapshot](source.tar.gz) bundles the solver, validation code, underlying equations and checked initialization files. The [figure](particle_validation.pdf) and [caption](particle_validation_caption.txt) can be used independently of the primary manuscript figures.
