# Degree dispersion and E/I composition at 2,000 and 4,000 neurons

Across 12 degree dispersions at 2,000 neurons, the mean activity dimension per neuron fell from 0.1540 to 0.0187. At eight newly sampled interior settings, the parameter-only theory tracked the curvature better than a straight line in χ joining its predicted log dimensions at χ = 0 and 2. In the separate composition sweep, the three paired network seeds did not resolve the predicted E–E decrease and I–I increase from 80% to 88% excitatory neurons.

[Degree sweep](../../../degree/degree_validation.pdf) · [Composition sweep](../../../figures/composition_main.pdf) · [Population participation ratios](../../../../figures/population_pr.pdf)

| Paired change at 2,000 neurons | Seed 1 | Seed 2 | Seed 3 | Mean change | 95% interval |
|---|---:|---:|---:|---:|---:|
| Joint dimension per neuron, χ = 2 minus χ = 0 | −0.1319 | −0.1359 | −0.1382 | −0.1353 | [−0.1432, −0.1275] |
| E–E squared covariance per distinct pair, 88% minus 80% E | −0.1059 | −0.0370 | +0.1484 | +0.0018 | [−0.3249, +0.3285] |
| I–I squared covariance per distinct pair, 88% minus 80% E | +0.2506 | −0.0217 | +0.4783 | +0.2357 | [−0.3861, +0.8576] |

The pair statistic is the mean of squared rate covariance over distinct ordered pairs within a population, multiplied by total network size. From 80% to 88% E, the mean rose from 0.6582 to 0.6600 for E–E and from 0.6717 to 0.9075 for I–I. The intervals use a two-sided paired Student t statistic with two degrees of freedom. The signs of the three mean changes in the table persisted with a shorter recording and a larger gap between estimator windows. The E dimension divided by total network size was 0.0336, 0.0299 and 0.0359 at E fractions 0.80, 0.88 and 0.94. Neither adjacent change resolved against its paired interval, so these measurements do not establish a maximum near 0.88. [Seed-level results and intervals](../../summary.json)

## Where the degree prediction agrees

At χ = 0 the measured dimension was 0.1540 ± 0.0016 (mean ± one SE across three networks), below the theory's 0.1739. At χ = 2 it was 0.0187 ± 0.0008, above the theory's 0.0168. The new interior settings distinguish the proposed curve from its endpoint trend. For each seed we computed the root-mean-square error in the natural logarithm of dimension per neuron, giving the eight new settings equal weight. The two seeds run after this comparison was specified gave errors of 0.141 and 0.131 against theory, compared with 0.455 and 0.428 against the endpoint line. The first seed gave 0.129 and 0.446. Four measurements for that first seed were available when this supplementary comparison was specified. [Theory values](../../../degree/theory_curve.json) · [Comparison and recording-length checks](../../summary.json)

## What was measured

The 81 fresh networks comprise 45 degree and 36 composition runs. At 2,000 neurons, three independently seeded networks were simulated at each of 12 χ values and nine E fractions. At 4,000 neurons, three were simulated at each of three values of each control. Within a family the seed labels are matched across settings. Each family uses distinct seeds, including at its common Bernoulli reference. The earlier 24 degree and 40 composition networks remain identifiable in the figures. The degree manipulation holds the population fractions and coupling matrices fixed and uses the predicted mean-input offset at each χ. The composition manipulation holds the presynaptic-column-weighted coupling matrices fixed as the E fraction varies. [Definition and controls](EXPERIMENT.md) · [81 configurations](../../jobs.json)

Each run used midpoint integration with step 0.05, a burn-in of 500 leak times and a recording of 8,000 leak times sampled every 0.25. The primary energy estimate uses four distinct 500-time-unit windows. All 81 records matched their assigned configurations. Independently reconstructing the mean temporal variance per neuron in each population and the diagonal covariance energies from the per-neuron records across every available duration, block length and gap gave maximum absolute differences below 3.5 × 10⁻¹⁶ across 87 saved sets of statistics. None of the measured windows triggered the checks for nonpositive or undefined estimates. Finite recording length and the precision of a three-seed comparison still limit the biological interpretation. [Raw network records](../../runs) · [Independent numerical checks](../../capsule_checks.json)

The Mac completed 21 networks. Its first four-worker attempt stopped at the 4 GiB memory limit during covariance estimation. The same seeds completed under an 8 GiB limit, with peak combined resident memory of 4.19 GiB. The three CBAI workers each completed 20 disjoint cases. All three were released, with an estimated combined worker compute cost of $0.63 from allocation to release, excluding disk and the existing coordinator. Their records and per-neuron arrays were checksummed on the workers and independently verified after download. [Cloud source snapshot](cloud_simulation_source.tar.gz) · [Execution and checksums](cloud_execution.json)

An independent check reproduced the endpoint contrasts, confidence intervals and interior-curve errors from the network records and matched all 129 cloud checksum entries to the downloaded files.

## Reproduction

The [original protocol](EXPERIMENT.md) records the commands used on the workers. From the repository root, replace `final/next_round/` with `experiments/` and `final/pr-plots/` with `figures/` when following those commands. The plots and [summary](../../summary.json) use the tracked network JSON records. To reproduce the per-neuron checks, restore the omitted `*_neuron_statistics.npz` arrays to [runs](../../runs), or move that directory aside in a disposable checkout and rerun the simulation before calling [check_capsules.py](../../check_capsules.py). The launcher skips draws with completed JSON records.
