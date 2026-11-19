# Fixed-size readout and population geometry in fresh matched networks

## Question

At fixed aggregate coupling and target rates, how much does a 100-cell excitatory readout change when the excitatory fraction rises from 0.80 to 0.88? Do distinct E–E and I–I pair energies move in opposite directions, and does total E dimension decrease again at 0.94?

## Hypothesis

Increasing the E fraction to 0.88 raises mean E readout PR by approximately 0.46 dimensions under a ratio-of-moments approximation. The corresponding I prediction is a decrease. E–E pair energy decreases and I–I pair energy increases. Total E dimension divided by network size has opposite signed slopes on the two flanks of the predicted optimum.

## Method

Use 624 fresh connectivity seed labels 20001 through 20624, each at all three E fractions. Use N=2000, the original Bernoulli population law, fixed aggregate couplings, mean target rate 0.3 and the archived input offsets. Use midpoint step 0.05, burn 500 leak times, record for 8000, sample every 0.25. For each class take eight uniform 100-cell subsets using the pilot's shared class-relative priorities across fractions. Calculate each readout's centered covariance with four distinct 500-time-unit blocks and gap zero. Average the eight PRs within each network. Compute full population energies on the identical trajectory. Retain block-gap and half-duration sensitivities and realized connectivity counts.

## Inputs

The frozen protocol.json fixes seeds, endpoints, estimator windows, and budget. Original graph parameters and offsets come from ../dense_sweeps/jobs.json. The design uses the existing four-seed readout pilot for variance only. Its seed labels are excluded from confirmation.

## Commands

From the repository root run python experiments/readout_confirmation/run.py --part i/12 --workers 20 on each CPU worker, substituting its one-based stripe index i. Fraction triplets stay on the same worker. Summarize every planned seed together after collection.

## Expected outputs

One compact JSON per network, including per-subset traces, energies and PR, full block and diagonal energies, measured counts, connectivity and code hashes, and numerical checks for both recording durations and gaps. Archive these losslessly for publication with a checksum manifest. Trajectories and large intermediate matrices are scratch.

## Controls

Verify the selected-cell block mean rates and variances against the same cells in the independently computed full-network estimator. The original engine checks a full four-window energy contraction on a spatial tile. Record three fractions for every connectivity label, without selecting outcomes or reusing pilot observations. Preserve stable or numerically degenerate outcomes; report an undefined endpoint on its own terms instead of replacing a seed.

## Success and failure criteria

Primary inference uses the two-sided paired 95% t interval for the E readout change 0.88 minus 0.80. An interval above zero supports an increase; compare the interval with the predicted 0.46 separately. Secondary contrasts are I readout, both distinct-pair energies from 0.80 to 0.88 and total E dimension on both flanks. Report pointwise and simultaneous Bonferroni intervals over these five contrasts. Opposite pair directions or a local maximum require the corresponding two signs to resolve together. Report realized precision and trajectory-length sensitivity even for a null. This tests geometry at the specified fractions and readout size.

## Time and cost

Up to twelve 32-vCPU workers, at most four hours and $4.48 in worker compute per allocation. CPU3c is $0.96/hour and CPU5c is $1.12/hour. Up to $53.76 total worker compute, excluding disks and the existing coordinator, with release after verified collection. The nine independent degree simulations and spectral replication have separate local cohorts.

## If something fails

Retry the same assigned cases on a verified named worker, lower concurrency if needed, and validate complete records before release. Retain failures and diagnose any invalid estimator. Never replace seeds or stop on an effect estimate.

## Questions asked before launch

The user authorized multiple CPU workers and the full experiment without a spending limit. The chosen design fixes a finite cohort and resource cap.

## Review

The primary comparison uses independent network labels and averages overlapping subsets before estimating uncertainty. Sizing uses the theory contrasts and four pilot labels. Three composition settings test specified flanks and cannot identify an exact maximum. Preliminary predictions, paired standard deviations and sensitivity to uncertain variance are in ../readout_theory/.

## Amendments

Before inspecting any fresh readout outcomes, independent review found that 384 triplets give about 68% power for the upper total-E-dimension flank at the five-way Bonferroni threshold. The cohort was increased to 624 triplets to target about 90% power at the pilot variance. If any selected group has zero varying rates or nonpositive covariance energy, retain the raw moments and label its PR undefined. For each contrast report complete paired counts and the separate excluded stationary/nonpositive cases. Under incomplete pairing report a sensitivity analysis across recording windows and the range of the mean contrast when undefined physical dimensions are allowed to lie between 1 and readout size (or 1 and class size for whole-class PR).

