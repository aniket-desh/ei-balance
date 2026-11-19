# Completed nonuniform EI degree test

All 24 declared degree networks completed and are retained. The original18-network test and its results are preserved in initial_18/. Six N4000 endpoint networks were declared as a staged extension after the N2000 residuals, before observing their own outcomes. Every parameter point has three fresh seed labels. The leading connectivity and external-input path is specified in degree_control_derivation.tex and adopted_protocol.json.

| N | chi | theory PR/N | network mean | network SE | mean relative error |
|---:|---:|---:|---:|---:|---:|
| 1000 | 0 | 0.173891 | 0.149156 | 0.003542 | -14.2% |
| 1000 | 0.5 | 0.055942 | 0.041364 | 0.004108 | -26.1% |
| 1000 | 1 | 0.034326 | 0.030165 | 0.002399 | -12.1% |
| 1000 | 2 | 0.016811 | 0.018960 | 0.000788 | +12.8% |
| 2000 | 0 | 0.173891 | 0.156176 | 0.001276 | -10.2% |
| 2000 | 2 | 0.016811 | 0.021896 | 0.000997 | +30.3% |
| 4000 | 0 | 0.173891 | 0.162277 | 0.001482 | -6.7% |
| 4000 | 2 | 0.016811 | 0.019306 | 0.001620 | +14.8% |

The degree effect transfers to genuinely nonuniform E/I local laws. Ratios of endpoint cohort means are:

- N1000: 7.87-fold, compared with 10.34-fold predicted.
- N2000: 7.13-fold, compared with 10.34-fold predicted.
- N4000: 8.41-fold, compared with 10.34-fold predicted.

These are ratios of cohort means, without a fitted amplitude or an uncertainty calculation that treats coupled seed labels across conditions as independent. The mean PR, ratio of mean moments, and conventional network intervals remain separate in comparison.json.

The test also identifies quantitative limits. At the largest tested size:

- chi0: PR mean error -6.7%; E/I temporal input variance errors +3.3% / +3.5%.
- chi2: PR mean error +14.8%; E/I temporal input variance errors +4.2% / +4.3%.

A large directional effect is not a uniform quantitative validation. The N2000 chi2 mean was 30% above the leading prediction and is retained even if later size points improve. Local-law and population-energy ingredient checks determine what may be claimed; a favorable final ratio alone does not.

Mean gap1-minus-gap0 shifts are at most 0.000131; T4000-minus-T8000 shifts are at most 0.000461. These diagnostics constrain recording effects for the reported observables but do not prove the infinite-time limit. B1000 is checked only on the selected endpoint seed4001 examples.

The figure degree_validation.pdf/png/svg shows the retained degree cohort, conditional theory, raw finite-T input-mean variances, temporal variances, and all three population energy contributions. Hollow chi1 symbols are a distinct composition-campaign reference with exactly the same Bernoulli connectivity law and simulation/estimator settings; they are not counted as extra degree replicates or pooled into the primary cohort.

- Same-law reference N2000: n=12, mean PR/N=0.033159, network SE=0.001431.
- Same-law reference N4000: n=4, mean PR/N=0.033491, network SE=0.001630.

Corrected local-moment diagnostics cover 14 of the 24 degree records. Coverage is explicitly recorded in ingredient_diagnostics/status.json; these are not additional networks. The remaining primary summaries retain raw finite-T local moments.

The selected chi0 N1000 seed4001 diagnostic shows that measured rate-mean heterogeneity inserted into the exact finite-pool row contraction predicts its observed static spread within about 2%. A local Gaussian expansion involving measured Var(q) and Cov(mu,q) explains much of the rate heterogeneity but retains a 7–12% residual in that example. It is a calibrated explanation, not a parameter-only finite-size PR correction. The later-size diagnostics are also retained.

The exact finite-pool covariance and third-cumulant identities, numerical refinements, static susceptibility, measured-source substitutions, and rejected assumptions are retained in this folder. Main-paper material, appendix algebra, and limitations are identified in manuscript_placement.md. No complete stationary finite-N theory, universal monotone convergence, or attractor probability is claimed.
