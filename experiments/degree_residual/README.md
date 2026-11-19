# Which covariance terms explain the degree residual?

At low degree dispersion, distinct E cells have more squared covariance than predicted, which makes measured activity dimension smaller than the theory predicts. At high dispersion, the relative errors in rate variance and covariance energy vary across cohorts. The graph-conditioned closure below tests one possible source of the discrepancy, heterogeneity in a cell's temporal input variance.

[Degree residual figure](degree_residual.pdf) · [Caption](degree_residual_caption.txt) · [Per-network terms and sensitivity checks](residuals.json)

## Degree accounting

For each network, let $d=D/N$, $\nu=\mathrm{tr}(C)/N$, and $E=\mathrm{tr}(C^2)/N$, with subscript 0 denoting the Gaussian prediction. The identity $d=\nu^2/E$ gives

$$
\log(d/d_0)=2\log(\nu/\nu_0)
-\sum_{b} (E_b-E_{b0})\frac{\log(E/E_0)}{E-E_0}.
$$

The common coefficient integrates $1/E$ along the straight segment between the predicted and measured block energies. Its equal-energy limit is $1/E_0$. The terms partition the exact log error for each draw. They describe covariance accounting, and do not identify which synaptic process produced an error.

| Dispersion and cohort at 2,000 cells | Log dimension error | Rate-variance term | E–E | 2 × E–I | I–I |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0, dense | −0.121 | +0.089 | −0.147 | −0.055 | −0.008 |
| 0, original | −0.108 | +0.093 | −0.141 | −0.052 | −0.008 |
| 2, dense | +0.104 | −0.046 | +0.082 | +0.058 | +0.011 |
| 2, original | +0.262 | +0.383 | −0.102 | −0.017 | −0.002 |
| 2, held out | +0.070 | +0.114 | −0.034 | −0.008 | −0.001 |

At dispersion 0, the distinct-pair part of the dense E–E contribution is −0.132 and its diagonal contributes −0.016. At dispersion 2 the contributions are +0.084 and −0.002, respectively. Shortening the recording to 4,000 time units or requiring one empty block preserves the signs of the endpoint log residuals in both cohorts. Every size, setting, seed and sensitivity estimate is retained. The original 2,000-cell cohort only sampled the endpoints. The dense 2,000-cell cohort sampled twelve dispersions, and each network-size comparison uses its own draws.

Among 45 dense degree networks, using each neuron's measured temporal variance with its measured mean input reduced opposite-half gain RMSE in every one of 90 population-by-network comparisons. For the 2,000-cell networks, within each seed the rank correlation across the twelve dispersions between the E temporal-variance error and the E–E log-error contribution ranged from −0.83 to −0.28. Both quantities change with dispersion. These comparisons motivate a connectivity-based prediction and leave the causal effect of temporal variance on off-diagonal covariance open.

## Predicting heterogeneous means and temporal variances

The prospective test uses three new connectivity seeds at each of three dispersions and the primary protocol frozen before accessing their activity. The calculation takes the specified graph, transfer function, target rate, and parameter-only Gaussian kernel for its dispersion. For each graph let $X_{ij}=J_{ij}-J^0_{ab}/\sqrt{N}$ for target type $a$ and source type $b$. With $m_i=2\Phi(\mu_i/\sqrt{1+q_i})$, solve

$$
\mu_i=h_a+\sum_j X_{ij}m_j,\qquad \langle m_i\rangle_a=0.3 .
$$

The graph-shared control sets every cell's $q_i$ to $q_{0a}$, the leading population variance. For the joint closure, keep the leading normalized temporal shape $r_a(t)=K_{0a}(t)/q_{0a}$ and compute $c_j(t)$, the covariance of the transfer function of two Gaussian inputs with mean $\mu_j$, variance $q_j$ and correlation $r_b(t)$. The even Green function of the leaky filter gives the presynaptic contribution $s_j=\int_0^\infty e^{-t}c_j(t)\,dt$. Assuming independent source fluctuations in the centered input gives the update

$$
u_i=\sum_j X_{ij}^2s_j,\qquad q_i=q_{0a}\frac{u_i}{\langle u_i\rangle_a}.
$$

Mean input and temporal variance are solved together. Both graph models enforce the same predicted population variance and average target rate. This isolates variation between cells in one population. The joint closure omits terms from distinct presynaptic covariances and cell-specific temporal shapes. It takes the archived population averages as input, so it cannot address an error in those averages. For PR, the empirical conditional spectra and squared gains enter the existing class-level cavity calculation. The graph supplies its empirical one-site distribution, while the pair calculation keeps the ensemble response.

The solver uses 48 squared Hermite coefficients for the transfer covariance. Independent Price quadrature checks the local expansion. Separate tests verify that numerical spectral masses are nonnegative to tolerance and that the homogeneous zero-dispersion limit recovers the three Gaussian dimensions. For these nine graph predictions the two modes both converged. The per-cell predictions and graph hashes can be inspected independently of any recordings. [Frozen evaluation protocol](PROTOCOL.md) · [Prediction checksums](prediction_hashes.json)

## Held-out results

The joint model predicted single-cell gains more accurately than the shared-variance graph model in all twelve population-by-network comparisons at χ = 1 and 2. It increased gain error in all six comparisons at χ = 0. At χ = 1 the E/I rank correlations of predicted with observed cell variances averaged 0.515/0.300, and at χ = 2 they averaged 0.519/0.311. At χ = 0 they were 0.031/0.031. Over the three dispersions, the paired differences in mean squared gain error, joint minus shared, were −0.000315 for E (95% interval −0.000464 to −0.000167) and −0.000208 for I (−0.000314 to −0.000101). Each pointwise interval uses three independent seed labels after averaging the settings within each seed.

Each entry below gives the three dimension fractions $D_E/N_E$, $D_I/N_I$, and $D/N$, in that order.

| Dispersion | Measured | Gaussian theory | Graph, shared variance | Graph, cell-specific variance |
| --- | --- | --- | --- | --- |
| 0 | .1836 / .5454 / .1602 | .1993 / .5717 / .1739 | .1985 / .5697 / .1732 | .1983 / .5688 / .1731 |
| 1 | .0396 / .1521 / .0339 | .0406 / .1477 / .0343 | .0475 / .1682 / .0406 | .0583 / .1883 / .0495 |
| 2 | .0215 / .0842 / .0181 | .0201 / .0778 / .0168 | .0293 / .1062 / .0248 | .0359 / .1192 / .0302 |

Improving the cell-level prediction worsened the collective prediction. Averaged across dispersions within each seed, switching from shared to cell-specific variance increased mean absolute log PR error by 0.137 for E, 0.077 for I, and 0.133 for joint activity (pointwise 95% intervals 0.114–0.161, 0.055–0.098, and 0.110–0.155). At χ = 1 and 2 the Gaussian ensemble prediction gave smaller mean absolute log errors than either graph-conditioned closure for all three dimensions. The Gaussian baseline remains appropriate for the degree figure. The gain improvement identifies a predictable part of heterogeneous cell dynamics, while the pair response and the connection between local inputs and covariance energy remain open. The zero-dispersion deficit in distinct E–E pairs repeats in the fresh cohort (−0.096 log units). At χ = 2 the fresh positive log-PR error, +0.070, is chiefly a numerator effect as in the original cohort. That last decomposition is a secondary comparison on the independent recordings.

All nine regenerated graph hashes match the simulated graphs exactly. Both graph models converged and passed the spectral checks on every draw. The largest absolute local covariance error against direct Price quadrature in a separate stress check was 1.77 × 10⁻⁵. The independent homogeneous test reproduced each Gaussian PR to 4.22 × 10⁻⁷. The per-network file retains the local mean-squared errors, gain moments, variance rank correlations, and E/I/joint log PR errors. [Held-out figure](joint_variance.pdf) · [Caption](joint_variance_caption.txt) · [All scores](holdout_scores.json)

## Reproduction

Run from the repository root. The residual summary requires the local per-neuron arrays for the dense degree cohort. The plotting script only requires the published JSON.

```sh
python experiments/degree_residual/residuals.py
python experiments/degree_residual/plot.py
python experiments/degree_residual/plot_holdout.py
python experiments/degree_residual/predict.py --out /tmp/ei-degree-predictions
python experiments/degree_residual/evaluate.py
```

The prediction command writes all 18 graph closures into a scratch directory. The evaluation command scores the originally frozen cell arrays in this directory against the nine runs in `experiments/degree_holdout/runs`. Seed labels are paired across the three dispersions. The two halves of a recording and the neurons within it do not supply independent networks.

Independent integration along the covariance-energy path reproduced all 207 per-network block decompositions (three recording conditions on 69 networks) to 5.6 × 10⁻¹⁷. Recomputing all 36 held-out gain errors directly from the block means and the frozen mean/variance predictions gave maximum disagreement below 6.6 × 10⁻¹⁹. [Source snapshot](source.tar.gz) · [Input and source provenance](provenance.json)
