# Testing the joint-dimension optimum along the composition path: design

Written on 27 November 2026, before any network of this cohort was simulated. The only simulations run before this file was frozen were the ten check networks of `check_jobs.json` (seeds 59001–59010, outside the cohort), which test the unequal-rate drive; the amendments section at the end records anything changed after that.

## Question

The note `../joint_dimension_optimum.tex` predicts, within the conditional mean-field theory and along the composition path (presynaptic columns rescaled so that J⁰ diag(f) and G diag(f) stay fixed):

1. **(i) Structure.** The block covariance energies obey E_ab(f) = f_a f_b Σ_c T_abc / f_c with an f-independent tensor T, and the per-population rate variances v_E, v_I do not depend on f.
2. **(ii) Form.** Hence D/N = (v_I + (v_E − v_I) f)² / (α/f + β/(1 − f) + p f + q (1 − f)).
3. **(iii) Location.** A single interior maximum, at f* = 0.851 for the reference parameters, below the optimum 0.887 of total E dimension.
4. **(iv) Rate dependence.** f* moves strongly with the two target rates.

Which of these can simulations of finite networks test decisively, and do they hold?

## What can and cannot be tested decisively

The existing data say why a model-free location test is weak. The predicted curve is flat at its top (within 1% of the maximum from f = 0.80 to 0.89), the single-network spread of D/N at N = 2,000 is about 14%, and at that size the finite-size offset in D/N is about 1% and varies with f. So an argmax over cell means, or a two-point comparison of 0.851 with 0.887, either needs thousands of networks per fraction or is confounded by an f-dependent offset of the same size as the effect.

The note's structure gives a much stronger handle. Each block energy is a linear function of two known basis functions of f, so the six T_abc can be estimated from block energies at several fractions; v_a is a constant. The optimum is then computed from fitted coefficients by the note's quintic. This pools every network in a rate pair and size into one estimate, and it separates the numerator (v) from the denominator (T), so finite-size errors in either can be seen. Its cost is that the resulting f* is only as good as the structure it assumes, which is why (i) is tested directly and a model-free contrast is kept as a secondary check.

Prediction (iv) is the most decisive test available, because it moves the predicted optimum by much more than either the noise or the plausible finite-size error. At the reference couplings:

| Rate pair (m_E, m_I) | Label | f* | d(f*) | 1% plateau | v_E | v_I |
| --- | --- | --- | --- | --- | --- | --- |
| (0.3, 0.3) | ref | 0.8510 | 0.03466 | 0.799–0.892 | 0.1440 | 0.1664 |
| (0.2, 0.4) | lo | 0.7866 | 0.03041 | 0.730–0.835 | 0.0934 | 0.1941 |
| (0.4, 0.2) | hi | 0.8948 | 0.03188 | 0.852–0.926 | 0.1615 | 0.1101 |

So the predicted shifts are f*_lo − f*_ref = −0.0644 and f*_hi − f*_ref = +0.0437, in opposite directions. All three curves have a single stationary point in (0, 1). These numbers come from `predictions.py` (`predictions.json`), which solves the theory with no network data: the tensor uses `check.tensor_for`, the routine of the note's unequal-rate table, and reproduces `theory.composition_tensor` at the reference to 1e−9.

The two unequal pairs were chosen to bracket the reference in both directions with moderate rates. Pairs with a rate of 0.1 move f* further (to 0.69 or 0.93) but sit where the campaign found the theory off by tens of percent at N = 2,000 (target rate 0.15: rate variance +46%), and 0.93 lies where only a handful of I→I edges are absent per row.

## Predictions tested, with their values

All values are parameter-only (`predictions.json`).

* **P-i (structure).** At each rate pair, the fitted T_abc and v_a equal the predicted ones (table in `predictions.json`, key `tensor`), the slope of v_a against f is zero, and the block cell means are fitted by the two-term linear forms within their noise. Reference tensor: T_EEE = 0.4800, T_EIE = 0.4079, T_IIE = 0.3698, T_EEI = 0.00772, T_EII = 0.02034, T_III = 0.0950.
* **P-iii (location and uniqueness).** f*_ref = 0.851 with a single interior maximum; f*_ref < 0.887.
* **P-iv (rate dependence).** f*_lo = 0.787, f*_hi = 0.895; shifts −0.0644 and +0.0437.
* **Model-free tilt.** τ = log d(0.94) − log d(0.70) is −0.0158 (ref), −0.1832 (lo) and +0.0900 (hi); the differences from the reference are −0.1675 (lo) and +0.1058 (hi).

## Cohort

Composition path of the paper campaign: reference couplings G⁰ = [[16, 9], [25, 12]] and J⁰ = [[1, −6], [2, −10]] at f⁰ = (0.8, 0.2), presynaptic columns rescaled by f⁰_b/f_b, Bernoulli edges, positive-erf rates, strict Dale signs, midpoint step 0.05, 500 leak times of burn-in, 8,000 recorded and sampled every 0.25, four-window mean-difference estimator with window 500 and no gap (the primary window), archived engine `runtime/engine_v4.py` with its native constant-weight action. The order-one drive h of each rate pair is the parameter-only mean input from the same solve. No extras are measured beyond the per-neuron moments and the sampled autocovariances that the runner always records.

| Size N | Rate pairs | Fractions | Networks per cell | Networks |
| --- | --- | --- | --- | --- |
| 2,000 | ref, lo, hi | 10 | 64 | 1,920 |
| 4,000 | ref, lo, hi | 10 | 16 | 480 |
| 8,000 | ref, lo, hi | 10 | 16 | 480 |
| 16,000 | ref | 10 | 4 | 40 |

Fractions: 0.60, 0.70, 0.75, 0.79, 0.82, 0.85, 0.88, 0.90, 0.92, 0.94 (N f is an integer at every size). Seeds: one label per network, 50001 upward in manifest order (`prepare.py`, `jobs.json`), so networks in different cells are independent. The reference pair uses the campaign's `composition_config` unchanged with scalar target rate 0.3; the unequal pairs pass `mean_rate = [m_E, m_I]`, which `make_network` already uses elementwise in the balance drive −√N J⁰ (f ∘ m) + h. No simulator code was changed (checks below).

The largest size that carries every rate pair is N = 8,000; N = 16,000 extends the reference only, as a check on how the location and the tensor move with size.

## Estimators

Per network, from the primary window: E_EE, E_EI, E_II (normalized by total N; E–I counted twice in the total), ν_E, ν_I, and d = (f ν_E + g ν_I)² / (E_EE + 2E_EI + E_II).

**Structural fit** (`estimators.py`). For each rate pair and size, separately per block,

    E_EE(f) = f T_EEE + (f²/g) T_EEI,   E_EI(f) = g T_EIE + f T_EII,   E_II(f) = (g²/f) T_IIE + g T_III,

fitted to the ten cell means by weighted least squares with weights n_k/μ_k², where μ_k is the parameter-only block value at f_k (equal relative error across cells, as the pilot data show; the weights do not depend on the data). v_a is the mean of ν_a over all networks of that rate pair and size. f̂* is the root of the note's quintic in (0, 1) with the largest d, computed from (T̂, v̂) by `check.interior`; the number of interior roots is recorded.

**Uncertainty.** Stratified bootstrap over networks within cells, 4,000 draws, percentile 95% intervals. Different rate pairs and sizes are independent cohorts, so the interval for a shift uses independent bootstrap draws of the two cohorts. Goodness of fit per block uses χ² of cell means against the fit with the empirical cell standard errors (8 degrees of freedom); f-independence of v_a uses the weighted slope of cell means against f.

**Model-free tilt.** τ = log d̄(0.94) − log d̄(0.70), with d̄ the ratio of cell means (mean ν)²/(mean E), the quantity the theory predicts; bootstrap intervals as above. The mean of per-network d is reported beside it.

## Comparisons and decision rules

**Primary 1, rate dependence (P-iv), at N = 8,000.** The shifts Δ_lo = f̂*_lo − f̂*_ref and Δ_hi = f̂*_hi − f̂*_ref with 95% intervals.
* *Held* if each interval excludes zero with the predicted sign and contains the predicted shift.
* *Direction held, magnitude not* if each excludes zero with the predicted sign but at least one excludes the predicted shift.
* *Failed* if either interval contains zero or has the wrong sign.
The same comparison at 2,000 and 4,000 is reported to show whether it depends on size.

**Primary 2, location at the reference (P-iii).** f̂*_ref with its 95% interval at N = 8,000 and N = 16,000.
* *Held* if the interval at the largest size contains 0.851 and excludes 0.887, and the estimates at 8,000 and 16,000 agree within their joint uncertainty.
* *Failed* if the interval at the largest size excludes 0.851 and the estimates do not move toward it with size.
* Otherwise *not resolved*. Uniqueness: the fraction of bootstrap draws with exactly one interior stationary point, reported per cohort.

**Primary 3, structure (P-i), at every size.** For each rate pair: the ratio T̂_abc/T_abc and v̂_a/v_a with intervals, the v-slope, and the block χ².
* The structure is taken to *hold up to finite-size error* if (a) at the largest size every ratio that moves f* by more than 0.002 per 5% (T_EEE, T_EIE, T_EEI, v_E, v_I, and T_IIE for lo) is within 10% of one, (b) its deviation from one shrinks from N = 2,000 to the largest size, and (c) the v-slopes and block χ² show no misfit at the largest size beyond what is present at 2,000. The 10% threshold is a judgment, not a derived bound: it is roughly the energy excess seen at N = 2,000 in the campaign.
* It *fails* if any of these deviations grows with N or stays above 10% at the largest size.
Because N = 2,000 cells hold 64 networks and N = 8,000 cells 16, the χ² tests are much more sensitive at the smaller size; misfit at N = 2,000 is read as a finite-size statement unless it persists.

**Secondary.**
1. The tilt τ per rate pair and its difference from the reference, against −0.1675 (lo) and +0.1058 (hi), at every size. This is model-free and should agree in sign and roughly in size with the structural shifts.
2. Decomposition: f* from measured T with predicted v, and from predicted T with measured v, to say which ingredient moves the optimum at finite N.
3. Window sensitivity: the structural f̂* and the tilts recomputed from the 4,000-leak-time window (gap 0) and the 8,000 window with one empty block between windows.
4. Measured population mean rates against their targets in every cell (this checks the unequal-rate drive on the whole cohort).
5. Fixed-point and settled networks per cell.
6. The existing campaign networks (13 fractions × 48, N = 2,000, reference rates), fitted the same way, as an independent replication at N = 2,000.

**What would count against the theory.** A rate-pair shift whose interval contains zero or has the wrong sign at N = 8,000; a reference optimum whose interval excludes 0.851 at the largest size and does not move toward it; fitted T or v that drift away from prediction as N grows; v_a that depends on f at large N; or a fitted curve with more than one interior maximum in most bootstrap draws.

## Power

Sizing used a residual bootstrap on synthetic cohorts (`validate_estimator.py`, `estimator_validation.json`): each synthetic network is the parameter-only cell mean times a multiplicative residual vector drawn from the 624 existing N = 2,000 composition networks, so the heavy tails (single networks reach twice the cell-median energy) and the correlations between the five measurements are those observed. Residuals are scaled toward one for larger networks; the existing size series gives relative spreads of block energies of about 0.14 at N = 2,000, 0.10–0.12 at 4,000 (12 networks) and 0.04 at 8,000 (6 networks), so scales 0.75 (4,000) and 0.30–0.45 (8,000) were used.

| Synthetic design | SD of f̂* (ref, lo, hi) | bias of f̂* | 95% interval coverage | SD of Δ_lo, Δ_hi |
| --- | --- | --- | --- | --- |
| N = 2,000, 64 per cell | 0.008, 0.007, 0.010 | < 0.001 | 0.92–0.94 | 0.011, 0.014 |
| N = 4,000, 16 per cell (scale 0.75) | 0.013, 0.009, 0.013 | < 0.001 | 0.90–0.96 | 0.015, 0.018 |
| N = 8,000, 10 per cell (scale 0.30) | 0.006, 0.005, 0.007 | ≤ 0.001 | 0.91–0.92 | 0.008, 0.009 |

The planned 16 per cell at N = 8,000 gives SDs about √(10/16) of the last row, or about 1.4 times that row's values if the spread there is at the pessimistic scale 0.45 (that design is also in `estimator_validation.json`). So the predicted shifts, −0.064 and +0.044, are 4 to 9 standard errors from zero at N = 8,000 on any of these spreads, and the reference f̂* has a standard error of 0.005 to 0.008, enough to separate 0.851 from 0.887 (0.036 apart). The percentile intervals cover at 90–94% rather than 95% in these simulations (cells of 10–16 networks with heavy tails); this is stated with the results instead of being corrected after the fact. With zero noise the estimator returns T, v and f* to rounding (1e−12 and 1e−9).
Before this design, the structural fit was applied to the existing reference networks (13 fractions × 48, N = 2,000) to see how it behaves on real data. It gave f̂* = 0.855 (bootstrap SD 0.008), T̂/T between 0.97 and 1.09 and v̂/v of 1.027 (E) and 1.002 (I), and block χ² of 20–23 on 11 degrees of freedom over the full range (7.7–12 on 5 for f ≥ 0.70). That fit was seen before the design was fixed; it is reported as secondary comparison 6, not as evidence for the primary comparisons.

## Controls and checks before renting

1. Generator: `make_network` with `mean_rate = [0.3, 0.3]` returns arrays bit-identical to scalar 0.3; with unequal rates the drive equals −√N J⁰ (f ∘ m) + h computed by hand after a JSON round trip, and the connectivity is unchanged (`validate.py`).
2. Ten N = 2,000 check networks through the copied runner (`check_jobs.json`), at both unequal pairs and the reference: population mean rates must lie within 0.02 of their targets (the campaign's reference networks lie within about 0.005 on average, with a spread of 0.01 per network).
3. Estimator on synthetic data from a known tensor: exact recovery with zero noise, small bias and near-nominal interval coverage at the planned sizes.
4. Runner: replay one archived campaign network (`replay.json`, `pc_composition_f0.86_N2000_seed43020`) on a pod with the archive's numpy and scipy and require identical D/N and block energies.
5. Collection: SHA-256 of every record and array on each pod compared with the local copy before the pod is stopped.

## Reporting rules

The unit of uncertainty is an independently generated network. Every planned network is kept. A network whose final drift is below 1e−10 has no stationary dimension; it is counted per cell and excluded from the averages, not replaced. Results are reported with 95% intervals whatever their outcome, and comparisons not listed above are labelled post hoc.

## Results of the checks run before freezing

* Check 1 passed: bit-identical arrays for `[0.3, 0.3]` against 0.3, the unequal-rate drive equal to the hand value to 1e−12 after a JSON round trip, and unchanged connectivity.
* Check 2, run on 27 November before this file was frozen but after the 0.02 threshold above was written: the measured population rates follow the targets in both directions (for example 0.2017/0.4016 and 0.3987/0.1992 against 0.2/0.4 and 0.4/0.2), so the drive works, but 2 of the 10 networks miss the threshold. Both are at (0.2, 0.4) and f = 0.94, with E rates 0.229 and 0.242 (I rates 0.422 and 0.430); their rate variances are also 28–31% above prediction. Check 1 shows the drive itself is exact, so this is the network's finite-size departure from balance at the most extreme fraction, where inhibitory cells are 6% of the network. The design is unchanged; secondary comparison 4 reports rates against targets in every cell and size, and the size control is what decides whether this matters.
* Check 3 passed (table above).
* Check 4 runs on the first pod with the cohort.

## Compute

The planned hardware was 32-vCPU `cpu5c` pods, but in EUR-IS-1 at launch only two 32-vCPU machines were available (`cpu3c`, $0.96/h, about half the per-core speed). The cohort therefore runs on 22 16-vCPU `cpu5c` pods ($0.56/h each, 32 GB RAM, 120 GB container disk) and the two 32-vCPU `cpu3c` pods, all secure cloud, each running one of 24 stripes of the manifest (a 32-vCPU `cpu3c` pod has about the throughput of a 16-vCPU `cpu5c` pod). A 16-vCPU `cpu3c` pod ($0.48/h) runs only check 4 and is then stopped. Total rate $14.24/h; a scheduler simulation of the runner's limits suggests two to three hours of wall time, roughly $35–45. Each pod carries a 6-hour self-stop timer armed and verified before anything else, and is stopped (never terminated) after its outputs are verified locally.

## Amendments

**27 November 2026, about 16:31 UTC (operational, no change to the cohort or the analysis).** Twenty-five minutes into the run the runner's resource model was holding most pods to three to seven concurrent networks: it budgets about 4.3 GiB of memory and 9 GiB of scratch per N = 8,000 network, while the measured resident size was 1.0–1.4 GiB and scratch use was under 2 GB per pod. The two 32-vCPU pods, which also carry two N = 16,000 networks each, had finished 4 networks in 20 minutes. Each stripe's runner was therefore replaced (`checks/pod_scripts/restart.sh`): the old runner was killed with SIGKILL so that its networks in progress kept running to completion, and a new runner was started on the same stripe with `--only` set to exactly the networks that were neither finished nor running, with `--memory-gib 56 --scratch-gib 400` and 15 workers in total on the 16-vCPU pods (`112` and 31 on the 32-vCPU pods). The manifest, seeds, code and per-network computation are unchanged; a network's result does not depend on how many others run beside it (the replay check reproduced an archived network bit for bit on different hardware). Because a new runner's worker cap does not count the networks it inherited, the same restart was repeated every eight minutes or so on pods running fewer than 13 networks (`checks/pod_scripts/refill.sh`). At about 17:00 UTC nine stripes still had 27–51 queued networks while nine pods had emptied their queues, so half of each long queue (alternate networks in queue order) was handed to a runner on an idle pod with `--part S/24 --only <names>` (`checks/pod_scripts/split.py`); the heavy pod records the handed-over names in `given.txt` and its later restarts exclude them, so no network can run on two pods. A second split at about 17:10 UTC for four stripes (17, 19, 21, 22) did not exclude the names handed over in the first split, so some networks were started on two pods. This was caught within two minutes: every runner for those stripes was killed on every pod, the running and finished networks were collected from all pods, the younger copy of each network running twice was killed (nine copies), and the remaining queued networks were dealt to disjoint runners (`checks/pod_scripts/rebalance.py`). Any network that nevertheless finished on two pods is listed with the collection checks in `cloud_execution.json`; both copies come from the same seed and code, and one is kept. Each pod's `restart.log` and the extra audit files in `audit/` record the restarts.
