# Larger cohort for the fixed-point mode comparison

Written before any of these networks were simulated. The measurement code, root-selection rule, subspace definition and both permutation controls are those of the [24-network replication](../../spectral_replication/PROTOCOL.md), called unchanged through `experiments/spectral_replication/measure.py`. Only the cohort differs.

**Cohort.** Bernoulli graphs at 80% E with the reference couplings and target rates 0.3: 32 new networks at each of 250, 500 and 1,000 cells and 16 at 2,000 cells (seeds 46001 onward). At 1,000 cells, 16 networks at degree dispersion 0 and 16 at dispersion 2 (seeds 46101 onward), using the degree sampler and the parameter-only drive of the degree sweep. Recordings are 2,000 leak times after 500 of burn-in, midpoint step 0.05, sampled every 0.25.

**Comparisons.** The same three as the replication, reported by size and by dispersion: the Spearman correlation between a mode's real eigenvalue and its projected input variance against the median within-population shuffle; the fraction of squared rate-covariance energy inside the leading tenth of fixed-point directions against the gain-decile shuffle; and the PR of the orthogonal complement relative to the full recording. The unit is a network. Networks that settle to a fixed point, root searches that fail and cases that exceed one hour are counted and reported, never replaced.

**What is new.** Sizes of 1,000 and 2,000 cells and the two non-Bernoulli dispersions were not in the replication, so their outcomes are not predicted by it. The N = 250 and 500 cells repeat the replication with different seeds.
