# Compact local block moments from the prospective campaign

These files preserve local moments from existing v1 network trajectories.
No additional network simulation was run. Each NPZ contains one row per
completed 500-time-unit block and one column per neuron:

- `mean_x`, `second_x`: temporal mean input and input square.
- `mean_rate`, `second_rate`: temporal mean rate and rate square.
- `mean_gain`, `second_gain`: mean analytical slope and slope square.

All reductions use float64, starting from the same float32 recorded inputs
used by the runtime analyzer. Metadata records the block length, sampling
interval, neuron count, E fraction, and verified written duration. The
watcher reads only prefixes published after a trajectory flush. It opens
read-only mappings while simulations run, so subsequent POSIX file deletion
does not invalidate the attached mapping. Each completed block is captured
once and the compact output is replaced atomically.

`coverage.json` distinguishes captured records from raw trajectories already
deleted before attachment. The watcher was added late in the campaign to
investigate finite-record input-mean noise, particularly at nominally fixed
degree. Coverage is therefore explicitly incomplete. The available N1000
endpoint examples are retained seed4001 records; they are selected diagnostics.
All six N2000 degree endpoint records were attached while running. The four
retained composition seed3001 records were also captured. None of these
summaries creates another independent network realization.

Crossblock local-law estimates require stationary independent windows.
For example, for block input means `mu[b,i]`, the average over distinct blocks
of `mean_i[(mu[b,i]-mean_i mu[b,i])*(mu[c,i]-mean_i mu[c,i])]` estimates static
input disorder without the positive same-window temporal-mean noise.
Mean input square minus the distinct-block product of input means estimates
temporal input variance. Gain second moments analogously use products of
means from distinct blocks, rather than squaring one finite-record estimate.
Negative corrected finite-sample results must not be clipped.

The four-window diagonal rate-energy source uses independent estimates
`v[b,c,i] = (second_rate[b,i]+second_rate[c,i])/2
             - mean_rate[b,i]*mean_rate[c,i]` and averages
`v[b,c,i]*v[d,e,i]` over four distinct blocks and all three matchings.
Degree-agent diagnostics also retain block-gap comparisons, corrected
heterogeneity of temporal input variance, and input-mean/variance covariance.
Those results live in the [degree diagnostics](../../degree/ingredient_diagnostics).

`validation.json` checks capsule dimensions, completed prefixes, and direct
raw-block agreement where the retained raw file remains available. It also
compares an independent elementary-symmetric-polynomial expression for the
four-window diagonal source with explicit four-block matching calculations.
The code is `../check_local_block_capsules.py`; the capture code is
`../local_block_capsules_watcher.py`. Missingness and selection qualifications
remain applicable even when every captured prefix has passed validation.
