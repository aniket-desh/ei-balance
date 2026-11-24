# Heterogeneous operating points in balanced E–I networks

Excitatory and inhibitory neurons can share a transfer function and still operate at different mean inputs. This project asks how those heterogeneous operating points shape neural covariance and the effective dimension of activity. The model, simulations, numerical records, and editable manuscript live in this repository.

[Read the paper](paper/main.pdf) · [LaTeX source](paper/main.tex) · [Overleaf source ZIP](docs/ei-balance-overleaf.zip) · [Figure storyboard](docs/E-I%20Paper%20Storyboard.html)

## Results

A larger campaign reran every simulation comparison with more settings, more networks and sizes from 500 to 16,000 neurons (1,694 networks). The conditional theory follows the degree sweep, the composition path from 50% to 94% excitatory cells, the single-neuron variance and gain laws, and the two-lag pair moments. Its largest error, a 10% deficit in dimension at fixed in-degree and 2,000 neurons, is a finite-size effect: at fixed in-degree it falls to 2.2% at 16,000 neurons. [Larger sweeps](experiments/paper_campaign/README.md) · [All figures](experiments/paper_campaign/figures/index.html) · [Overview figure](experiments/paper_campaign/figures/overview.pdf)

Increasing the excitatory fraction from 80% to 88% raised the participation ratio of a fixed 100-cell excitatory readout by 0.59 dimensions and lowered it by 0.73 for an inhibitory readout of the same size. Across 624 fresh matched network triplets, distinct E–E pairs lost squared covariance and I–I pairs gained it. Total excitatory dimension per network neuron was highest at 88% among the three tested fractions. The theory predicted all six contrast directions and overestimated the inhibitory readout change. [Independent composition experiment](experiments/readout_confirmation/README.md)

In the original 81-network sweep, increasing in-degree dispersion reduced joint activity dimension per neuron from 0.154 to 0.0187 at 2,000 neurons. The nonlinear theory tracked the intermediate settings. Excess squared covariance between distinct excitatory cells explains most of the low-dispersion discrepancy. [Degree sweep](experiments/degree/degree_validation.pdf) · [Dense sweep measurements](experiments/dense_sweeps/experiments/dense-ei/README.md) · [Degree residuals](experiments/degree_residual/README.md)

The follow-up experiments supply editable figures, standalone captions, and the measurements behind them.

| Research question | Results and figure |
| --- | --- |
| Which covariance terms account for the error in the degree sweep? | [Network-level decomposition](experiments/degree_residual/README.md) · [Degree figure](experiments/degree_residual/degree_residual.pdf) |
| Does predicting each cell’s temporal variance improve the local gains and population dimension? | [Nine held-out networks](experiments/degree_holdout/README.md) · [Gain and dimension comparison](experiments/degree_residual/joint_variance.pdf) |
| How does a fixed 100-cell readout differ from counting all excitatory dimensions? | [Derivation](experiments/readout_theory/README.md) · [624-triplet result](experiments/readout_confirmation/README.md) · [Readout figure](experiments/readout_confirmation/composition_readout.pdf) |
| Do fixed-point modes identify directions of high rate covariance? | [24-network replication](experiments/spectral_replication/README.md) · [Spectral figure](experiments/spectral_replication/spectral_replication.pdf) |

| Directory | Contents |
| --- | --- |
| [src/eicavity](src/eicavity) | Network model, mean-field solvers, covariance theory, and participation-ratio calculations |
| [tests](tests) | Numerical checks of the model and theory against exact limits and small reference calculations |
| [experiments](experiments) | Simulation code, experiment definitions, seed-level measurements, and validation figures |
| [figures](figures) | Population PR and degree/composition plots, with plotting scripts and saved values |
| [paper](paper) | Manuscript, tables, references, and compiled PDF |
| [docs](docs) | Figure storyboard, editable cavity diagram, and Overleaf handoff |
| [archive](archive) | Earlier derivations, handoff files, and reference data used by the current analyses |

The [fixed-size readout](experiments/readout_pilot/README.md), [fixed-point spectral analysis](experiments/spectral_pilot/README.md), and [particle mean-field comparison](experiments/particle_pilot/README.md) each have their own methods, results, and reproduction commands. The links named `next_round` and `recovered` support paths recorded by earlier simulation and plotting code.

## Install and reproduce

From the repository root, install the Python package and run the numerical tests. Python 3.10 or newer is required.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests
```

To redraw the PR plots from the published values and network records, run the following commands. The scripts write their plots in [figures](figures).

```sh
python figures/pr_controls.py
python figures/population_pr.py
```

With TeX Live and `latexmk` installed, rebuild the manuscript from the same root.

```sh
latexmk -pdf -cd paper/main.tex
```

## Data and provenance

The repository includes the run configurations, seeds, compact theory kernels, plot inputs, and source snapshots. The [readout records](experiments/readout_confirmation/data) reproduce the confirmatory contrasts, and the [held-out cell measurements](experiments/degree_holdout/measured) reproduce the primary gain and variance comparisons. Large trajectory recordings and covariance-block arrays stay local. The [dense-sweep definition](experiments/dense_sweeps/experiments/dense-ei/EXPERIMENT.md) and [checksums](experiments/dense_sweeps/experiments/dense-ei/cloud_execution.json) identify the original runs.

To regenerate the dense sweep's per-neuron arrays, use a separate disposable checkout. The runner skips networks with completed JSON records, so move its published output directory aside there before running. The command below runs all 81 cases under an 8 GiB memory budget. The native backend requires a C++ compiler.

```sh
mv experiments/dense_sweeps/runs experiments/dense_sweeps/published-runs
python experiments/dense_sweeps/run.py --shard all --workers 2 --memory-gib 8 --scratch-gib 20 --native
python experiments/dense_sweeps/check_capsules.py
```

For the separate 100-cell readout pilot, move `experiments/readout_pilot/measurements` aside in that disposable checkout, then run `python experiments/readout_pilot/run.py` to regenerate its block arrays. Replays with different numerical backends can follow different trajectories on the same connectivity.
