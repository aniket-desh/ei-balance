# How composition changes a fixed 100-cell readout

At 2,000 neurons, increasing the excitatory fraction from 0.80 to 0.88 gives a leading prediction of **+0.46 dimensions for a 100-cell E readout and −1.17 for a 100-cell I readout**. The prediction takes the ratio of the expected squared trace and expected covariance energy of a uniformly sampled readout. In the four pilot seed pairs, taking the mean of the eight individual ratios changes the estimated contrast, relative to that ratio of moments, by +0.01 E dimensions and +0.12 I dimensions on average. The pilot’s measured E change itself was −1.29 dimensions. The independent experiment tests whether the small positive theoretical contrast survives in simulated networks. [Theory values and curve](predictions.json) · [Pilot records and sizing](pilot_and_sizing.json)

## The finite subset

Fix one network and its stationary rate covariance matrix $C$. Sample $K$ cells uniformly without replacement from a population of $M$ cells. Write $v_i=C_{ii}$, $m_1=M^{-1}\sum_i v_i$, $m_2=M^{-1}\sum_i v_i^2$, and $P=N\sum_{i\ne j} C_{ij}^2/[M(M-1)]$. For subset $S$, let $T_S=\operatorname{tr} C_{SS}$ and $Y_S=\operatorname{tr}(C_{SS}^2)$. Its PR is $T_S^2/Y_S$.

Each cell is included with probability $K/M$ and each ordered distinct pair with probability $K(K-1)/[M(M-1)]$. It follows that

$$
\mathbb E_S T_S=K m_1,\qquad
\mathbb E_S Y_S=K\left[m_2+(K-1)P/N\right].
$$

Expanding the squared trace separates $K$ diagonal terms from $K(K-1)$ products of different variances. Their mean product is $(M m_1^2-m_2)/(M-1)$. The exact variance of the subset trace and the resulting ratio of moments are

$$
\operatorname{Var}_S T_S=K\frac{M-K}{M-1}(m_2-m_1^2),\qquad
R_K=\frac{\mathbb E_S T_S^2}{\mathbb E_S Y_S}
=\frac{K m_1^2+\frac{M-K}{M-1}(m_2-m_1^2)}
{m_2+(K-1)P/N}.
$$

These identities condition on a fixed covariance. For numerical theory we substitute the conditional one-site moments for the empirical $m_1,m_2$ and the leading two-site result for $P$. Along the fixed-aggregate Bernoulli composition path, the one-site moments stay fixed while $P_{aa}(f_E)=c_{aE}/f_E+c_{aI}/(1-f_E)$ changes. Sampling 100 of the 120 I cells at 94% E most strongly suppresses the variance of the subset trace. Network-to-network fluctuations of the moments and finite-record errors remain in a simulation comparison.

| Excitatory fraction | E readout $R_{100}$ | I readout $R_{100}$ | $P_{EE}$ | $P_{II}$ | Total E dimension $D_E/N$ | Per-E-cell $D_E/N_E$ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.80 | 28.438 | 30.243 | 0.5826 | 0.6317 | 0.03245 | 0.04056 |
| 0.88 | 28.900 | 29.076 | 0.5589 | 0.7027 | 0.03398 | 0.03862 |
| 0.94 | 28.268 | 25.556 | 0.5916 | 0.9584 | 0.03241 | 0.03448 |

The analytic maximum in total E dimension occurs at $f_E=0.88744$. On the grid of attainable population sizes the E readout ratio peaks at 0.8825, near the E–E pair-energy minimum. E dimension per E cell decreases across all three settings. Ignoring the variance of the subset trace would predict changes of +0.455 E and −1.099 I dimensions on the lower flank, compared with +0.462 and −1.166 when the variance is retained.

## How much the ratio approximation changes the answer

With $X=T_S^2$, $Y=Y_S$ and $D=X/Y$, the exact difference is $\mathbb E D-R_K=-\operatorname{Cov}(D,Y)/\mathbb E Y$. The one-site diagonal moments and the mean distinct-pair energy alone leave that covariance undetermined. A four-cell example makes this explicit. Give each cell variance one. In one network, set all six off-diagonal entries above the diagonal to 0.2. In another, give one pair covariance $\sqrt{6}\,(0.2)$ and the others zero. Both covariance matrices are positive definite. For a two-cell readout they give the same $R_2=1.92308$, while enumerating the six subsets gives mean PRs of 1.92308 and 1.93548. [Exact enumeration](validation.json)

For each pilot network and population we used the eight measured traces and four-window energies to compute both averages. The difference between their 80%→88% E contrasts was +0.009 E dimensions (paired 95% interval −0.257 to +0.276) and +0.125 I dimensions (−0.197 to +0.447). Intervals use four independent seed differences. Eight overlapping subsets give a finite estimate within each network, and the separate four-window estimates have recording error. These numbers quantify the size of the approximation in the pilot. Applying them as universal corrections to new networks would require a model for the joint fluctuations of numerator and denominator. [Per-network ratios, fluctuations, and expansion remainder](pilot_and_sizing.json)

## Size of the independent test

The design uses 624 fresh matched seed triplets, 20001–20624, at 80%, 88%, and 94% E. The primary endpoint is the network-paired change in the mean of eight actual 100-cell E-readout PRs from 80% to 88% E. Each network supplies one mean per class. The secondary endpoints for the mechanism are the 100-cell I-readout change and both full-population distinct-pair energies on that lower flank. Testing the total-E-dimension peak uses both flanks. A lower-flank covariance shift or a readout PR change by itself does not establish a downstream task benefit.

| Endpoint (upper minus lower fraction) | Theory change | Pilot paired SD | Seeds for 80% power | Expected interval half-width at 624 | Power at 624 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E readout PR, .88−.80 | +0.462 | 2.196 | 180 | 0.173 | .9995 |
| I readout PR, .88−.80 | −1.166 | 3.729 | 123 | 0.386 | >.9999 |
| $P_{EE}$, .88−.80 | −0.02371 | 0.10841 | 248 | 0.01121 | .9980 |
| $P_{II}$, .88−.80 | +0.07099 | 0.20759 | 104 | 0.02147 | >.9999 |
| $D_E/N$, .88−.80 | +0.001533 | 0.005505 | 154 | 0.000569 | >.9999 |
| $D_E/N$, .94−.88 | −0.001570 | 0.010082 | 485 | 0.001043 | .9038 |

The primary interval is two-sided 95%. The five secondary intervals are each 99%, giving at least 95% simultaneous coverage by Bonferroni. The powers in the table are the individual rejection probabilities at these thresholds, assuming independent, normally distributed seed differences, the four-pair pilot SD, and the theoretical change. Requiring both total-E flanks to resolve in the predicted directions gives at least 90.3% power by the union bound under these assumptions. The same lower bound for both pair-energy directions is 99.7%. No independence across endpoints is needed for either bound. The primary test has 80% power for a change of 0.247 dimensions at the pilot SD. One whole E dimension, a reference unit for interpreting precision, would require 40 seeds at 80% power. A biological minimum effect would need a task-defined criterion.

The small pilot makes the variability estimate uncertain. If the primary SD is 1.5 or 2 times its pilot value, the 624-seed power for +0.462 falls to .938 or .747. For the upper-flank test of total E dimension the corresponding powers fall to .504 and .262. The sizing record includes the pointwise 95% widths and the prespecified widths, together with SD sensitivity and both flanks for every endpoint. For six pilot condition runs, the historical full-population and readout recordings used the same graph but different numerical trajectories. Each endpoint’s pilot SD comes from its own four matched seed differences. The independent experiment measures the endpoints on the same trajectory. [Pilot method](../readout_pilot/README.md) · [Sizing calculation](calculate.py)

## Reproduction

From the project root, run the following commands in the installed Python environment.

```sh
python experiments/readout_theory/calculate.py
python experiments/readout_theory/validate.py
```

The first script writes the three anchor predictions, 901 composition settings, and the pilot and sizing records. The second enumerates every subset size in an eight-cell covariance matrix and verifies the four-cell example. When the local pilot blocks are available, it also reconstructs one 100-cell I readout directly from the packed lower triangles and all 5,460 four-window pairings. The published validation records the checksum, trace, energy, and PR. All theory and sizing inputs are small tracked files. The raw pilot blocks are needed only for the independent reconstruction.
