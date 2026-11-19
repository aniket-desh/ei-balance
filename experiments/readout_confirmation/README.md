# How composition changes a fixed-size readout

Increasing the excitatory fraction from 80% to 88% raised the mean participation ratio (PR) of a random 100-cell excitatory readout from 27.405 to 27.997. The same change lowered the mean PR of a 100-cell inhibitory readout from 29.393 to 28.660. Squared covariance between distinct excitatory cells fell, while squared covariance between distinct inhibitory cells rose. These changes support a trade-off in activity dimension at fixed readout size in this balanced network model. [Figure](composition_readout.pdf) · [Caption](caption.txt) · [Plot source](plot.py)

## Prediction and measurement

Each row gives the change from the lower to the higher excitatory fraction. All six comparisons use the same 624 fresh seed triplets. The primary E-readout comparison uses a two-sided paired $t$ interval at 95% confidence. The five secondary paired $t$ intervals use 99% confidence apiece, targeting 95% simultaneous coverage by Bonferroni. $P_{aa}$ is $N$ times the mean squared covariance over distinct ordered pairs in population $a$. [Predictions](../readout_theory/README.md) · [Measured contrasts and pointwise intervals](summary.json)

| Quantity | E fraction | Theory | Measured change | Paired interval |
| --- | --- | ---: | ---: | ---: |
| 100-cell E readout PR | 80% → 88% | +0.462 | +0.592 | [+0.330, +0.854] |
| 100-cell I readout PR | 80% → 88% | −1.166 | −0.734 | [−1.118, −0.350] |
| E–E pair energy $P_{EE}$ | 80% → 88% | −0.02371 | −0.02049 | [−0.03451, −0.00648] |
| I–I pair energy $P_{II}$ | 80% → 88% | +0.07099 | +0.06849 | [+0.05017, +0.08682] |
| Total E dimension $D_E/N$ | 80% → 88% | +0.001533 | +0.001797 | [+0.001165, +0.002429] |
| Total E dimension $D_E/N$ | 88% → 94% | −0.001570 | −0.001733 | [−0.002364, −0.001101] |

The theory predicts the direction of all six changes. It overestimates the absolute E-readout PR at all three fractions, while its predicted increase from 80% to 88% lies inside the primary interval. Its predicted I-readout decrease is larger than the measured decrease and falls outside the adjusted interval. Total E dimension per network neuron was highest at 88% among the three fractions tested. Dividing by the number of E cells instead gives a decreasing sequence of 0.0398, 0.0382 and 0.0339. These three settings bracket the predicted maximum of $D_E/N$ near 88.7% E. Locating the maximum more precisely or establishing a downstream decoding benefit would require a separate test.

## Measurements and reproduction

Each seed produced three 2,000-cell networks with matched seed labels, fixed aggregate couplings and fixed target rates. In each network we averaged the PRs of eight uniform, potentially overlapping 100-cell subsets per population before computing seed-level differences. Full-population and readout measurements use the same recording. Across the four recording-length and block-gap choices, the mean E-readout change ranged from +0.585 to +0.603 dimensions. The independent validator checked all 1,872 network records and 7,488 window estimates and regenerated 36 connectivity matrices to verify their hashes. [Frozen design](EXPERIMENT.md) · [Seed-level measurements](network_estimates.csv) · [Validation](validation.json)

From the repository root in the installed Python environment, rebuild the measurements from the compact records, verify them, and redraw the figure.

```sh
python experiments/readout_confirmation/summarize.py
python experiments/readout_confirmation/validate_results.py
python experiments/readout_confirmation/plot.py
```

The twelve compressed archives contain per-subset traces and energies and full-population summaries. All twelve Runpod worker allocations were released. Estimated worker compute was $2.89, excluding storage and the existing coordinator. [Records](data) · [Worker log and archive checksums](cloud_execution.json) · [Simulation source hashes](simulation_sources.json)
