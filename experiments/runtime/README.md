# Optimized, bounded simulation campaign

The simulations run in this remote Linux workspace, not on the user's Mac.
The current host reports 9 logical CPUs and approximately 10 GB memory. The
included launcher is designed for macOS as well: it uses multiprocessing
`spawn`, limits BLAS to one thread per worker, and caps worker count using
available memory and CPU count. macOS execution itself has not been tested in
this session.

## What was optimized and verified

Each strict-Dale connectivity block has one possible nonzero edge weight. For
an edge probability above one half, write its Bernoulli adjacency as one minus
the absent-edge adjacency. The deterministic part is a population sum. Store
the less frequent event in a CSR matrix and apply the remaining population
term as a rank-two operation. This represents the original sampled matrix
exactly; it does not prune weak synapses or approximate the model.

For these configurations, the sparse representation stores approximately
9.6%–18.1% of matrix entries. The initial single-worker benchmark found a
3.9-fold speedup at N1000, 2.2-fold at N2000 (60:40), and 4.2-fold at N4000
(80:20), relative to single-thread dense midpoint integration. Giving dense
BLAS 2 or 4 threads did not beat the sparse single-thread alternative in the
N2000 comparison. Each midpoint step is vectorized across neurons through
SciPy CSR operations and vectorized Gaussian CDF evaluations. Independent
connectivity realizations run in separate processes.

The optional native kernel exploits the four constant block weights and
vectorizes their sums. Its SIMD build reorders floating additions; it does
not change edge weights, the transfer function, drive, or midpoint update.
Short action and 50-step trajectory checks have errors below 4e-14. Long
chaotic trajectories are expected to diverge from reordered floating sums;
no bitwise long-time equality is claimed. Native benchmarks were taken while
the initial campaign ran, so their timings are under-load measurements.
The default portable launcher uses SciPy and needs no C++ compiler. `--native`
requests a local build and falls back to SciPy if compilation is unavailable.
No prebuilt Linux binary is needed on a Mac.

The previous estimator held every N×N covariance block in resident arrays.
The new estimator writes covariance blocks to temporary files and contracts
spatial row tiles. All four-distinct-window matchings are retained. No random
projection or sampled-pair approximation is used. Literal population formulas
and the prior estimator agree to about 1e-16 in the independent-Gaussian test.
Full N1000/N2000 formula checks and small spatial-tile checks are retained.
The covariance-energy repair remains unbiased only for mutually independent
stationary windows. PR is a ratio and is not exactly unbiased.

## Resource limits

- At most four network workers, one BLAS thread each, leaving two logical CPUs
  unused when that restriction is tighter.
- Total network-process resident-memory budget is the smaller of 4 GiB and
  40% of initially available memory. The driver monitors aggregate RSS and
  stops if the configured limit is reached. A conservative per-worker array
  estimate can reduce the worker count before starting.
- The adopted jobs' temporary file sizes remain below the 20 GiB campaign
  ceiling. The data are memory mapped and streamed; disk-backed arrays are not
  loaded in full. Selected trajectories are retained on scratch for independent
  checks. Other trajectories and covariance files are deleted after compact
  summaries have been atomically saved.
- Two N4000 cases run in the first extension wave together with smaller cases.
  Their measured resident peaks determine worker count for the remaining large
  cases. The gate uses memory only, not agreement with theory.

The resource audit files contain actual measured memory, worker allocation,
thread controls, and platform details. They distinguish observed limits from
estimated requirements.

## Prespecified comparisons and exact normalization

`campaign_jobs.json` contains 16 fresh composition cases and 18 nonuniform
EI degree cases. `extension_jobs.json` adds 16 composition cases at N2000,
8 composition cases at N4000, and 6 degree endpoint cases at N4000. The primary estimator is B500, zero empty blocks, T8000. One-empty-
block estimates and T4000 prefixes are sensitivity checks. Selected B1000
analyses were listed before their outcomes. All listed outcomes are retained.
Repeated parameter cases sharing a seed are paired controls, not independent
replicates across control settings. Different seeds within a setting provide
network variation.

Population covariance energies are divided by the total network size:

    E_ab = sum_{i in a,j in b} C_ij^2 / N.
    E_total = E_EE + 2 E_EI + E_II.
    nu_total = sum_a f_a nu_a.
    d = nu_total^2 / E_total.

The population-uniform projection diagnostic uses U with columns
`1_E/sqrt(N_E)` and `1_I/sqrt(N_I)`. For two covariance estimates C and D its
energy is `(2 tr((CU)^T(DU)) - tr((U^T C U)(U^T D U)))/N`. This is an exact
orthogonal decomposition of covariance energy into entries touching the two
uniform population axes and the perpendicular sector. It does not describe
the full covariance spectrum.

V1 reports whole-record gain and input-mean moments as finite-record
diagnostics. V2 additionally retains input/gain block means, computes cross-
block gain and static-variance estimates, and saves compact neuron statistics.
Selected V1 trajectories receive independent postprocessing. Do not silently
label every V1 ingredient as a bias-corrected estimate.

## Running on another machine

From the repository root, install the project and run its campaign launcher.

    python -m pip install -e .
    python experiments/runtime/run_campaign.py --jobs experiments/runtime/campaign_jobs.json --workers 4

The launcher skips completed saved records by default. Use `--rerun-complete`
only when intentionally reproducing them, and `--native` for the optional
locally compiled kernel. The launcher replaces stored host-specific native
library paths with a local path or the SciPy fallback. Reruns write to the
same result names; keep a copy of prior outputs if comparing implementations.

The simulation remains float64; trajectory storage is float32, matching the
existing protocol. Covariance construction and contractions use float64.
Selected midpoint-tangent and half-step-size checks are specified in the
queue; their completed records are required before treating them as validation.
A positive tangent exponent
supports chaos in that tested case; other networks are described as
fluctuating rather than automatically classified as chaotic.

For a fixed-point mode plot, give a new pilot job a distinct name and set
`retain_temporary_trajectory` to `true`. After the run, use its record from
the repository root.

    python3 experiments/runtime/mode_variance.py experiments/runtime/runs/<name>.json

The script rebuilds the same connectivity, checks the recorded population
means, and solves for a fixed point from the temporal mean input. You can
supply a candidate input vector with `--fixed-point point.npy`. Both paths
require a maximum absolute drift below or equal to `1e-8`. The NPZ stores
the eigenvalues of `J diag(phi'(x*))`, with stability boundary `Re(lambda)=1`,
and the full-, first-half-, and second-half variances along each unit right
eigenvector. `--variable rate` maps the directions through `diag(phi'(x*))`
and normalizes them before projecting the rates. Each variance is centered
within its own window. Nonorthogonal directions can share variance. A root
identifies one equilibrium of the sampled network, and the split estimates
show how much the measurement depends on the recording window.

## Files

- `engine.py`: unchanged first-campaign implementation, source hash recorded.
- `engine_v2.py`: all current extensions, with extra local-law diagnostics.
- `engine_v4.py`: portable export of V3, also handling unsupported madvise and
  using a conservative macOS memory estimate. Fresh-directory compilation was
  tested without changing the active native library.
- `engine_v3.py`: identical estimator arithmetic with raw-map residency reduced;
  612 N4000 numerical fields match V2 exactly, peak reanalysis RSS 645 MB.
  V4 is the portable launcher default; the already active large-case queue
  continues under V2 with two workers to avoid interrupting simulations.
- `optimization_benchmark.json`, `native_*benchmark.json`: timings and action checks.
- `estimator_validation.json`, `v2_estimator_validation.json`: independent checks.
- `campaign_protocol.json`, `extension_protocol.json`: sampling and method decisions.
- `runs/*.json`: one atomically saved record per requested case.
- `runs/*neuron_statistics.npz`: compact V2 local statistics, not raw trajectories.
- `integration_check_jobs.json`: four N1000 half-step reruns and one selected
  N4000 degree-endpoint half-step rerun, the latter recording T4000.
- `tangent_checks.py`: selected finite-time tangent checks.
- `runtime_rejections.md`: unsuccessful implementation attempts and their correction.
- `tmp/`: regenerable raw work arrays and native binaries, excluded from deliverables.

## Final numerical validation outcome

All 64 primary cases, five half-step-size controls, three targeted outlier
follow-ups, and four selected tangent checks are complete. The four tangent
exponents and every one of their eight duration blocks are positive. These
checks support the chaotic branch for those selected cases only.

Four of five half-step PR comparisons change by at most 1.05%. The N1000
80:20 seed3001 exception changes by24.3%; three T16000 same-network follow-ups
show material arithmetic/phase, duration, and timestep sensitivity. Its PR
estimate is not established as converged. See `outlier_diagnostic_findings.md`;
all original data remain in the primary cohort and this exception is a
limitation. No additional numerical or basin campaign is launched.

The maximum observed aggregate RSS in monitored covariance-simulation phases
was 2,102,587,392 bytes. All original network pools and controllers finished.
`runtime_completion_summary.json` records complete counts, exact control
comparisons, memory, backend versions, hashes, and the outlier qualification.
