# Prospective composition findings

Completed primary measurements: 40 of 40 prespecified composition jobs.

The primary estimator is B500, T8000, gap0. All planned statuses are retained. The independent statistical units are seed clusters. Conventional network-mean t intervals describe the finite-record estimator; they do not prove stationarity, window independence, or asymptotic approximation bounds.

| fE | N | Networks | Mean PR/N ± network SE | 95% mean interval | Leading prediction | Relative mean error |
|---|---:|---:|---:|---:|---:|---:|
| 0.6 | 1000 | 4 | 0.02410 ± 0.00316 | [0.01404, 0.03415] | 0.02989 | -19.4% |
| 0.8 | 1000 | 4 | 0.03032 ± 0.00684 | [0.00857, 0.05208] | 0.03433 | -11.7% |
| 0.6 | 2000 | 12 | 0.02942 ± 0.00111 | [0.02697, 0.03187] | 0.02989 | -1.6% |
| 0.8 | 2000 | 12 | 0.03316 ± 0.00143 | [0.03001, 0.03631] | 0.03433 | -3.4% |
| 0.6 | 4000 | 4 | 0.02997 ± 0.00047 | [0.02848, 0.03146] | 0.02989 | +0.3% |
| 0.8 | 4000 | 4 | 0.03349 ± 0.00163 | [0.02830, 0.03868] | 0.03433 | -2.4% |

## Ingredient errors remain visible

| fE,N | nu error | Covariance-energy error | Relative PR-error interval | Mean ratio minus ratio of mean moments |
|---|---:|---:|---:|---:|
| f0.6_N1000 | -13.1% | -5.5% | [-53.0%, +14.2%] | 0.000206 |
| f0.8_N1000 | -2.5% | +10.0% | [-75.0%, +51.7%] | 0.000629 |
| f0.6_N2000 | -0.6% | +1.7% | [-9.8%, +6.6%] | 0.000363 |
| f0.8_N2000 | -1.2% | +2.9% | [-12.6%, +5.8%] | 0.000551 |
| f0.6_N4000 | +5.1% | +10.5% | [-4.7%, +5.3%] | 0.000091 |
| f0.8_N4000 | +0.7% | +4.1% | [-17.5%, +12.7%] | 0.000081 |

A broad interval containing the leading prediction is not an equivalence result. Any accuracy statement must give the observable, network size, and uncertainty. The formal first-order correction has not been validated as a finite-network repair.

## Static and temporal variance

| fE,N | Population | Mean Delta | Mean q | Mean Delta+q | Leading Delta+q | Across-network corr(Delta,q) |
|---|---|---:|---:|---:|---:|---:|
| f0.6_N1000 | E | 4.774 | 1.645 | 6.419 | 5.643 | -0.782 |
| f0.6_N1000 | I | 6.915 | 2.509 | 9.424 | 8.646 | -0.949 |
| f0.8_N1000 | E | 4.025 | 1.904 | 5.929 | 5.643 | -0.899 |
| f0.8_N1000 | I | 6.680 | 2.913 | 9.593 | 8.646 | -0.987 |
| f0.6_N2000 | E | 3.782 | 1.929 | 5.710 | 5.643 | -0.634 |
| f0.6_N2000 | I | 5.955 | 2.956 | 8.911 | 8.646 | -0.728 |
| f0.8_N2000 | E | 4.063 | 1.907 | 5.970 | 5.643 | -0.436 |
| f0.8_N2000 | I | 6.135 | 2.920 | 9.055 | 8.646 | -0.437 |
| f0.6_N4000 | E | 3.642 | 2.038 | 5.681 | 5.643 | -0.817 |
| f0.6_N4000 | I | 5.542 | 3.127 | 8.670 | 8.646 | -0.989 |
| f0.8_N4000 | E | 4.049 | 1.943 | 5.992 | 5.643 | -0.695 |
| f0.8_N4000 | I | 6.077 | 2.985 | 9.061 | 8.646 | -0.587 |

These empirical correlations are exploratory, particularly in four-network groups. Total input variance is not held fixed by the measured networks. The exact conditional fixed-total-variance derivative and parameter-only degree tangent are documented separately; they do not provide a full quenched correction.

## Population covariance geometry

The following secondary diagnostic normalizes the ordered EE,EI,IE,II energy matrix by its total. It tests allocation independently of the raw amplitude, whose errors remain above. The error is the Frobenius distance to the predicted normalized matrix divided by the norm of that prediction.

| fE,N | Error of mean normalized network matrix | Error of normalized ensemble mean energy matrix | Mean fraction touching population-uniform axes |
|---|---:|---:|---:|
| f0.6_N1000 | 3.91% | 3.61% | 0.263% |
| f0.8_N1000 | 4.21% | 4.03% | 0.537% |
| f0.6_N2000 | 1.37% | 1.43% | 0.112% |
| f0.8_N2000 | 2.31% | 2.46% | 0.161% |
| f0.6_N4000 | 0.64% | 0.68% | 0.028% |
| f0.8_N4000 | 0.91% | 0.94% | 0.043% |

A small population-uniform sector rules out that particular subspace as the entire source of excess energy. It does not exclude nonuniform collective modes or quantify all coherent feedback effects.

## Paired size comparisons

Common-seed contrasts use only the prespecified first four seed clusters. Additional N2000 networks increase within-point precision; they are not inserted as independent unmatched observations into a size contrast.

- f0.6_1000_to_2000: paired PR/N change 0.00633; 95% paired interval [-0.00430, 0.01697]; 4 seed clusters.
- f0.6_2000_to_4000: paired PR/N change -0.00046; 95% paired interval [-0.00464, 0.00372]; 4 seed clusters.
- f0.8_1000_to_2000: paired PR/N change 0.00124; 95% paired interval [-0.02274, 0.02521]; 4 seed clusters.
- f0.8_2000_to_4000: paired PR/N change 0.00193; 95% paired interval [-0.00687, 0.01073]; 4 seed clusters.

## E and I dimensions

Population fractions use d_a=f_a nu_a^2/E_aa; absolute dimensions are D_a=N_a d_a. The cross-population rectangular block has no PR here.

| fE,N | Mean dE ± network SE | Mean dI ± network SE | Mean DE ± network SE | Mean DI ± network SE | Predicted DE,DI |
|---|---:|---:|---:|---:|---:|
| f0.6_N1000 | 0.03475 ± 0.00444 | 0.06431 ± 0.00821 | 20.85 ± 2.66 | 25.72 ± 3.29 | 25.29, 32.42 |
| f0.8_N1000 | 0.03515 ± 0.00796 | 0.14097 ± 0.02621 | 28.12 ± 6.37 | 28.19 ± 5.24 | 32.45, 29.54 |
| f0.6_N2000 | 0.04187 ± 0.00164 | 0.07890 ± 0.00270 | 50.24 ± 1.96 | 63.12 ± 2.16 | 50.59, 64.84 |
| f0.8_N2000 | 0.03898 ± 0.00167 | 0.14525 ± 0.00523 | 62.38 ± 2.67 | 58.10 ± 2.09 | 64.90, 59.07 |
| f0.6_N4000 | 0.04244 ± 0.00064 | 0.08084 ± 0.00152 | 101.85 ± 1.54 | 129.34 ± 2.44 | 101.18, 129.68 |
| f0.8_N4000 | 0.03933 ± 0.00178 | 0.14712 ± 0.00776 | 125.85 ± 5.68 | 117.70 ± 6.21 | 129.80, 118.14 |

The separately normalized cross-population energy E_EI/sqrt(E_EE E_II) is also retained in the JSON. It is a temporal-Gram overlap diagnostic, not communication capacity or a rectangular PR.


## What this does not establish

The campaign does not derive every finite-size vertex or predict the full distribution of network-specific outliers. The conditional Gaussian local and pair closures remain assumptions about the fluctuating branch. Finite-window gain and static-variance checks on selected retained networks diagnose estimator and local-law effects but are not independent parameter-only predictions. Report a residual when ingredient errors or sampling uncertainty prevent a stronger accuracy claim.
