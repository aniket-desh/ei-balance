# Fixed-point modes and population rate covariance

We tested whether the directions singled out by a heterogeneous E–I network's fixed point also organize the fluctuations of its rates. The frozen protocol specified 24 independent Bernoulli networks at 80% E, including 12 draws at each of 250 and 500 cells. All 24 completed. Four 250-cell networks converged to stable fixed points. Their late stored inputs were constant, their final drift was below 4 × 10⁻¹⁵ RMS, and the leading real coupling eigenvalue of the occupied root was below one. The other 20 draws contributed to the variance comparisons. The experiment adapts the fixed-point mode plot from the trivial fixed point in an i.i.d. model to numerically verified nontrivial fixed points in balanced networks with heterogeneous operating points. [Clark, Abbott and Litwin-Kumar, Fig. 2b](https://arxiv.org/pdf/2207.12373#page=3) · [Protocol](PROTOCOL.md)

## What replicated

In every fluctuating network, modes with a larger real eigenvalue tended to carry more input variance. The median rank correlation was 0.348, compared with −0.081 across within-population shuffle medians. The prespecified median *paired* difference was 0.446. All 20 differences were positive (exact one-sided sign test, $p=2^{-20}$). The median agreement between the two recording halves in the variance ranking was 0.983. All 34 roots passed the drift threshold. Nine fluctuating networks had a second verified root, which changed the input correlation by up to 0.141. The effect stayed positive at every verified root of every fluctuating network. [Network contrasts and root sensitivity](summary.json)

| Size | Fluctuating / simulated | Median input rank correlation | Median observed rate energy fraction | Median gain-shuffled energy fraction |
| --- | --- | --- | --- | --- |
| 250 | 8 / 12 | 0.272 | 0.671 | 0.140 |
| 500 | 12 / 12 | 0.397 | 0.389 | 0.081 |

The leading tenth of fixed-point directions, mapped into rates, contained a median 53.1% of the recorded rate variance and 39.5% of its squared covariance energy. The latter exceeded the gain-matched shuffle in all 20 networks, with a median paired difference of 30.9 percentage points. Taking the orthogonal complement increased PR in 18 of 20 recordings. The median complement-to-full PR ratio was 1.20, compared with 1.01 for the gain-matched shuffles. A subset of the local linear directions thus identifies where the nonlinear recording concentrates covariance, even after stratifying shuffled cells by gain. These measurements support a geometric interpretation at the tested sizes and 80:20 composition. They leave open whether fixed-point information alone can predict whole-network PR or the response to degree dispersion. [Vector figure](spectral_replication.pdf) · [Editable SVG](spectral_replication.svg) · [Caption](caption.txt)

## Measurements

For each verified fixed point $x^*$, we diagonalized $A=J\operatorname{diag}\phi'(x^*)$. The leak shifts the eigenvalues of the continuous-time linearization by $-1$. For a unit right eigenvector $u$, the color in the spectrum is $u^\dagger C_xu$, where $C_x$ is the covariance of the recorded inputs after subtracting their temporal means. Rank correlations use one member per conjugate pair and each real mode. We compute them for the whole recording and for each separately centered half. Each of 64 null permutations shuffles cell labels within E and within I, using the same permutation on all modes in one draw. The first verified root in the fixed search order defines the primary comparison. The search uses the recording mean among its starting points. [Per-network measurements, roots, attempts and null draws](measurements) · [Saved roots, spectra, mode variances and bases](modes)

For the rate comparison we select eigenvalues with the largest real parts, completing complex-conjugate pairs at the cutoff. This gives 25–26 or 50–51 real directions according to network size. A local input displacement $u$ maps to a rate displacement $\operatorname{diag}\phi'(x^*)u$. The columns of $Q$ orthonormalize the real span of the selected rate displacements. For the directly measured rate covariance $C$, we evaluate

$$
t=\frac{\operatorname{tr}(Q^\top C Q)}{\operatorname{tr}C},
\qquad e=\frac{\lVert Q^\top C Q\rVert_F^2}{\lVert C\rVert_F^2},
\qquad R=\frac{D((I-QQ^\top) C (I-QQ^\top))}{D(C)},
\qquad D(C)=\frac{(\operatorname{tr} C)^2}{\lVert C\rVert_F^2}.
$$

The energy fraction $e$ counts the covariance block entirely inside the chosen subspace. The complement can have higher PR if projection preferentially removes dominant covariance directions. We compare each measurement to its median in 64 within-population coordinate permutations and in 64 permutations further restricted to deciles of the fixed-point gain within each population. Each paired difference uses the observed and shuffled subspaces of the same network. Individual modes are nonorthogonal, so the block calculation uses $Q$ directly. The PR here is the geometry of the observed recording under ordinary centering. The manuscript's cross-window network estimator answers a different finite-record question.

## Numerical checks

Some cells have nearly vanishing gain. A direct eigensolve followed by rescaling its returned right eigenvectors produced physical-coordinate residuals as high as 3.73 × 10⁻⁵. We used an orthogonal real Schur reduction before recovering the eigenvectors and recomputed every spectrum and null. The maximum residual over 34 roots was below 1.53 × 10⁻¹⁴. Removing modes with $|\operatorname{Re}\lambda|\leq10^{-8}$ as a numerical sensitivity check gave a positive association in all 20 fluctuating networks (median 0.426). Explicit projections of the time series agreed with spatial quadratic forms for eight probes at each root within 3.74 × 10⁻¹⁴. The independent check reconstructs the primary Spearman coefficient, rate fractions, complement PR, and a gain-matched shuffle directly from the saved arrays. [Validation](validation.json)

## Reproduction

From the repository root, run the following with Python 3.11 and the installed project requirements. The first command regenerates local trajectories and records. The second reuses verified analyses with matching source, record, and trajectory hashes. The numeric summary and figure can be regenerated directly from the published compact measurements and mode files. The independent check also reads the local trajectories.

```sh
python experiments/spectral_replication/run_networks.py --workers 2
python experiments/spectral_replication/measure.py --workers 2
python experiments/spectral_replication/summarize.py
python experiments/spectral_replication/plot.py
python experiments/spectral_replication/check.py
```

The simulation used 500 time units of burn-in, 2,000 units of recording, a midpoint step of 0.05 and samples every 0.25. Each worker used a single numerical thread and the existing 1.5 GiB process watchdog. The 24 simulation records took 73 summed seconds including their archived block estimators. The compact results retain the network and trajectory checksums, search attempts, verified roots, split-half measures, and shuffle distributions. [Simulation records](runs) · [Source hashes and independent checks](validation.json)
