---
name: dense-ei
title: Degree dispersion and covariance geometry in balanced E/I networks
status: done
cpu: 32
count: 1
hours: 4
budget_usd: 5.00
disk_gb: 128
data:
secrets:
---

# Degree dispersion and covariance geometry in balanced E/I networks

## Question

Does changing the variance of incoming degree lower the dimension of balanced network activity across the previously unsampled part of the degree sweep? At fixed single-neuron input laws in the leading theory, does increasing the E fraction from 0.80 to 0.88 reduce covariance among pairs of E neurons while increasing covariance among pairs of I neurons?

## Hypothesis

The parameter-only stationary closure predicts that joint participation ratio per neuron falls from 0.17389 at common degree dispersion χ = 0 to 0.01681 at χ = 2. On the separate composition path, it predicts opposite changes in the two populations' per-pair squared covariances between f_E = 0.80 and 0.88. With P_aa = N times the mean of C_ij² over distinct ordered pairs in a, its I–I minimum is at f_E = 0.76758 and its E–E minimum at 0.88245. At a fixed N, the predicted total E dimension D_E has a separate maximum at 0.88744. These are predictions along the prescribed rescaling of coupling strength with population fraction. Testing a downstream task or the transfer of a conserved quantity would require a different observable.

## Method

Use the positive-erf rate model φ(x) = 2Φ(x), unit leak and strict presynaptic Dale signs from the archived campaign. The reference populations are f⁰ = (0.8, 0.2), G⁰ = [[16,9],[25,12]] and J⁰ = [[1,−6],[2,−10]], with the two target mean rates both 0.3. A postsynaptic row of type a draws K_ab presynaptic cells uniformly without replacement conditional on the count. Degree dispersion is Var(K_ab) divided by the Bernoulli count variance N_b p_ab(1−p_ab), with finite integer rounding as implemented in the archived sampler.

| Family | N = 2,000 | N = 4,000 | Seed labels at each setting |
|---|---|---|---|
| Common χ at f_E = 0.80 | 0, 0.025, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1, 1.25, 1.50, 2 | 0.10, 0.75, 1.50 | 5101, 5102, 5103 |
| Bernoulli f_E at χ = 1 | 0.60, 0.65, 0.70, 0.75, 0.80, 0.84, 0.88, 0.91, 0.94 | 0.70, 0.88, 0.94 | 6101, 6102, 6103 |

The 81 draws comprise 45 degree and 36 composition networks. Within a family, the same three random-number seeds identify contrasts across settings and sizes. Each setting generates its own connectivity. The χ = 1, f_E = 0.80 networks in the two families have the same ensemble law and different sets of seeds. Keep both cohorts identifiable.

The degree sweep fixes G⁰, J⁰ and f⁰. Use the archived, parameter-only h(χ) for each point before simulating. The leading balance drive is fixed while its finite offset varies. The composition sweep rescales the presynaptic columns of G⁰ and J⁰ by f_b⁰/f_b to preserve G diag(f) and J⁰ diag(f). Use the χ = 1 reference offset throughout. This preserves the conditional one-site law in the leading closure and tests its finite-size accuracy experimentally.

Integrate midpoint dynamics in float64 with step 0.05. Discard 500 leak times for burn-in, record 8,000, and sample every 0.25. Use the primary four-distinct-window, mean-difference covariance-energy estimator with window length B = 500 and minimum empty gap 0. Evaluate T = 4,000 versus 8,000 and gap 1 sensitivity for every network. At N = 4,000 also evaluate B = 1,000 on the first seed of each of the six anchor settings. Store float32 trajectory samples only as temporary data. Accumulate covariance and block statistics in float64.

## Inputs

The workspace snapshot supplies the runtime and recovered model source, the reference matrix controls and offsets in `final/next_round/degree/theory_curve.json`, and the χ = 1 conditional kernel in `final/next_round/degree/chi1_one_site.npz`. The prediction for the composition slice is computed from this kernel by `final/pr-plots/population_pr.py`. The deterministic manifest `final/next_round/dense_sweeps/jobs.json` defines each case, offset, seed, estimator and shard. Preserve its hash and the source revision or source hashes used for collection. The inputs are synthetic and require no external data, account keys or credentials on workers.

## Commands

From the workspace root containing `final/`, prepare the complete manifest once before network execution. Run the two disjoint shards under their respective resource limits. A one-case pilot with reduced duration can check installation and is excluded from the 81 primary draws.

```bash
python3 final/next_round/dense_sweeps/prepare.py
python3 final/next_round/dense_sweeps/run.py --shard local --workers 4 --memory-gib 8 --scratch-gib 20 --native
```

Stage the project at `/workspace/ei-balance-n19` on the verified coordinator. Start three workers with the same approved scientific design and a distinct 1-based stripe. For each worker, replace `K` with its assigned integer, 1, 2 or 3. The driver applies the stripe to the ordered remote manifest before testing for completion.

```bash
python3 final/next_round/dense_sweeps/run.py --shard remote --part K/3 --workers 32 --memory-gib 48 --scratch-gib 110 --native
```

Publish the files directly under `final/next_round/dense_sweeps/runs/` in an experiment-scoped directory on each worker, including per-network JSON, every B500 or B1000 neuron-statistics capsule, and the host audit. Collect each stripe separately to the Mac, verify its exact manifest membership and hashes against its worker, then merge the disjoint files. Verify the final 81 names, estimator configurations and NPZ capsules before producing figure files. Retain the data locally before releasing workers.

After collecting both shards, run the aggregation in the workspace root.

```bash
python3 final/next_round/dense_sweeps/summarize.py
```

## Expected outputs

Keep `final/next_round/dense_sweeps/jobs.json`, per-case `runs/<name>.json` under that directory and its `summary.json`. Per-case records must identify the job, actual population sizes, seed, coupling/offset, connectivity hash, estimator settings, status and finite-size degree statistics. Save the population mean variances, full 2×2 energy and four-window diagonal energies wherever the contract calls for P_aa. Record missing inputs or failed cases in the summary. Saving one final mean per condition would lose the uncertainty unit and the ability to recover each contrast.

## Controls

For centered temporal rate covariance C, let B_ab = ||C_ab||²_F/N, ν_a = tr(C_aa)/N_a and f_a = N_a/N. Check d_a = f_a ν_a²/B_aa and d = (Σ_a f_a ν_a)²/(B_EE + 2 B_EI + B_II). Given Ddiag_a = Σ_{i∈a} C_ii²/N from matching four-window estimates, use P_aa = N²(B_aa−Ddiag_a)/[N_a(N_a−1)]. This normalization keeps a change in the number of pairs from masquerading as a change in pair covariance. Preserve numerator and denominator separately, and compute cohort means of individual-network PRs.

Compare the Bernoulli anchors with both the archived predictions and the archived ensemble, displaying old and new samples separately. Check realized versus requested count dispersions, especially at χ = 0. The smallest incoming pool for the degree grid is 400 and every requested χ is at most 2, within the count sampler's domain. At f_E = 0.94 the I pool has 120 or 240 neurons. The expected missing I inputs to an I target are only 4.17 and 8.34 at the two sizes, respectively. Retain this high-fraction point and its size sensitivity with the results. Coherent excursions, atypical draws, weak fluctuations and sensitivity to observation duration are outcomes to report. A fluctuating trajectory alone does not establish chaos.

## Success and failure criteria

Use N = 2,000 for both prespecified comparisons. For the degree test, compute the difference in joint d at χ = 2 minus χ = 0 for each degree seed. For the composition test, compute P_EE(0.88)−P_EE(0.80) and P_II(0.88)−P_II(0.80) within each composition seed. Show all three differences, their mean, and a pointwise two-sided 95% Student-t interval with two degrees of freedom. The predicted degree change is resolved if its interval is wholly negative. The paired geometry prediction is resolved if the E interval is wholly negative and the I interval wholly positive. A resolved wrong sign counts against its respective hypothesis. Intervals crossing zero are inconclusive. Three independent seed labels provide limited precision for the predicted few-percent E effect. A sign reversal between the prespecified window or length checks makes the corresponding claim sensitive to recording protocol.

At each N = 2,000 seed, compare log-dimension RMSE over the eight new interior χ values (0.025, 0.05, 0.10, 0.20, 0.30, 0.75, 1.25, 1.50). Compare the fixed nonlinear prediction with the straight secant in χ between the fixed predicted log dimensions at 0 and 2. Give each value equal weight, report theory RMSE minus secant RMSE by seed with its paired t interval, and check the recording-length and gap sensitivity. Negative differences favor the nonlinear shape. This tests shape against that comparator. The three-seed composition sweep is a pilot for the few-percent effect. Use its paired variation to size an independent confirmatory experiment if one is warranted.

For the remaining points, report individual networks, mean ± one network SE, the parameter-only theory, and finite-size and finite-record residuals. As a secondary comparison, inspect both flanks of D_E/N at f_E = 0.80, 0.88 and 0.94. The discrete grid can support a bracket for a maximum, not a precise empirical estimate of 0.88744. The three N = 4,000 composition anchors do not include 0.80, so comparisons at that size must use their actual settings and explicitly label any historical reference. Report each primary case even if it disagrees with theory. Flag unresolved estimates, including nonpositive estimated energies, and do not clip them or count neurons, pairs or windows as replicates.

## Time and cost

The local shard takes the first seed at all 21 N = 2,000 settings. Two or three 32-vCPU workers split the remaining 60 draws, including every N = 4,000 anchor. The top block applies separately to each worker. Each may run for at most four hours, with a $5 ceiling including container disk. The campaign ceiling is $10 for two, or $15 for three. Quoted compute rates are $1.12/h for cpu5c or $0.96/h for cpu3c, plus approximately $0.08 for 128 GB of disk over four hours. Actual rates and stock must satisfy the cap at placement. With the historical runner, median elapsed time per network was approximately 180 seconds at N = 2,000 and 337 seconds at N = 4,000 on its prior Linux host. These timings do not establish throughput on a new host. Set BLAS to one thread per worker process and let live RAM and free disk limit concurrency. The Mac shard caps its process memory budget at 8 GiB and scratch at 20 GiB. Each cloud driver caps these at 48 and 110 GiB and must respect the resources actually provisioned.

## If something fails

Keep completed records and retry an interrupted job with the same config and seed. Lower concurrency if the host approaches a memory or scratch limit. Use the tested SciPy path if native compilation is unavailable, record the backend, and retain the same equations and estimator. Preserve completed cloud results before releasing a worker. A fallback to the memory-capped Mac may run the remaining named jobs after the cloud shard stops. Do not run the same case on both machines concurrently or pool duplicated outputs. A missing or invalid case is a technical failure for that cell. Show partial coverage and leave its planned contrast inconclusive pending recovery.

## Questions asked before launch

The user authorized the CPU campaign, requested multiple pods with the best available CPU, and authorized simultaneous work on the Mac under its memory limit. The hardware and per-worker cap above implement that instruction. Verify the CBAI account and the provider's current case-sensitive `aniket-` pod prefix before accessing or changing a pod.

## Review

Independent review identified overlapping default remote commands, an endpoint-only degree criterion and limited precision for the composition contrast. We added deterministic worker stripes, explicit collection with checksums, the eight-point curvature comparison and the pilot interpretation. The reviewer accepted the revisions before cloud execution. The four local records already available when the interior comparison was added are disclosed under Amendments. The aggregation script is checked against the raw records before closeout.

## Fixed rules

Keep the manifest and seeds fixed, retain every draw, and report protocol-sensitive contrasts.

## Amendments

The first local attempt reached the 4 GiB combined-RSS guard during covariance estimation. The same draws were replayed under an 8 GiB cap on the 48 GiB Mac. The additional shape criterion was specified after the first four local degree records became available and before the independent cloud seeds ran. Treat the two cloud seeds as the prospective check on that supplementary criterion.

## Outcome

All 81 prespecified networks completed. At N = 2,000, the paired degree endpoint change was −0.1353 (95% interval −0.1432 to −0.1275). The nonlinear curve had smaller interior log errors than the endpoint secant in both prospectively evaluated seeds. Neither population’s paired composition contrast excluded zero. All three workers were released after verification. Elapsed worker times were 14 min 31 s, 13 min 22 s and 10 min 3 s. Estimated worker compute totaled $0.634. See the [full results](README.md) and [worker receipts](cloud_execution.json).
