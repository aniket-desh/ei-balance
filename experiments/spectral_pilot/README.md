# Fixed-point modes and recorded input variance

Three of the four Bernoulli E/I networks fluctuated during recording. In all three, directions with larger real eigenvalues of the fixed-point coupling tended to have larger projected input variance. The fourth network settled to a stable fixed point and its stored trajectory had zero temporal variance. The experiment adapts the observable in [Clark, Abbott and Litwin-Kumar, Fig. 2b](https://arxiv.org/pdf/2207.12373#page=3) to these four small E/I networks.

| Cells | Seed | Unstable modes | Full-record ρ | First / second half ρ | Rank agreement between halves | Largest shuffled ρ |
|---:|---:|---:|---:|---:|---:|---:|
| 250 | 7101 | 6 | .400 | .390 / .403 | .983 | .138 |
| 250 | 7102 | 0 | — | — | — | — |
| 500 | 7101 | 8 | .503 | .473 / .526 | .950 | .044 |
| 500 | 7102 | 8 | .313 | .306 / .319 | .949 | .112 |

Each row uses the first verified root in the fixed search order. The unstable count includes both members of a complex-conjugate pair. The rank statistics take one representative from each pair and all real modes. The full-record statistic is the Spearman correlation between the real eigenvalue and projected variance. Rank agreement compares the variance of each mode between recording halves. For each of 32 controls we permuted coordinates within E and within I using the same permutation for all eigenvectors, which preserves their geometry and population weights. These permutations assess association within one network. The four draws and the alternate fixed points limit inference across networks.

## Network and observable

Every network has E fraction .8, target population rates .3, and independent Bernoulli edges (χ = 1). The variance profile is [[16, 9], [25, 12]] and the signed mean connectivity is [[1, −6], [2, −10]], with rows indexing recipients. The population input offsets come from the archived composition reference at the same fraction. The midpoint integrator used a .05 step and a 500-unit burn-in, then saved the input every .25 units for 2,000 time units. Each half has 4,000 samples and is centered independently.

For a verified fixed point x*, we diagonalized A = J diag φ′(x*) and calculated v = u† Cₓ u for each unit-norm complex right eigenvector u. Cₓ is the equal-time covariance of the recorded inputs. The continuous-time linearization is −I + A, so Re λ > 1 gives an unstable fixed-point direction. The directions are nonorthogonal and their projected variances need not sum to the network variance. Fixed-point instability and fluctuations during the recording do not establish a positive Lyapunov exponent.

The direct root solver started at the recording mean, its first sample, and the population offsets. Only the nonfluctuating network yielded a verified root directly. For each remaining network we used pseudo-arclength continuation of −x + a + s[Jφ(x) + drive − a] = 0 from s = 0 to full strength, with the recording mean and population offsets as separate anchors. We retained only full-strength solutions with maximum absolute drift below 10⁻⁸. One search for N = 500, seed 7101 reached s = .9833 at the step limit. Its other anchor succeeded. Across all four networks the six distinct roots had drift residuals below 3.1 × 10⁻¹⁴. Two networks had a second verified root. Their full-record correlations changed from .400 to .534 (N = 250, seed 7101) and from .313 to .456 (N = 500, seed 7102). Root choice affects the size of the association.

## Reproduction and checks

Run these commands sequentially from the project root. The simulation reuses completed, parameter-matched runs with retained trajectories. All numerical stages ran locally, one numerical thread per process. The simulation and analysis monitored a 1.5 GiB process limit, with observed peaks of 204 and 468 MiB, respectively.

```sh
python3 -B experiments/spectral_pilot/simulate.py
python3 -B experiments/spectral_pilot/analyze.py
python3 -B experiments/spectral_pilot/plot.py
```

[summary.json](summary.json) records all root attempts, per-network hashes of the connectivity, trajectory and run record, and per-root statistics. [source.tar.gz](source.tar.gz) contains the simulation, projection, continuation and direct-check source files with hashes in the summary. The [run records](runs) and [fixed points with their measured projections](modes) are included. Running `simulate.py` regenerates the local trajectories in `tmp/` for `analyze.py`. Regenerating the connectivity reproduced each recorded hash, and the recorded population means agreed with direct trajectory averages. The worst right-eigenvector residual was 1.4 × 10⁻¹⁰. An independent calculation using the explicit spatial covariance agreed with streamed projections within 5.5 × 10⁻¹⁴ in absolute variance.
