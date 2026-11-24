# Larger sweeps for the paper: design

Written on 24 November 2026 before any of these networks were simulated, except the amendments at the end, which record what changed during execution.

## Question

Do the paper's simulation comparisons hold when each is repeated with more settings, more networks and a wider range of sizes? In particular: is the 10% dimension deficit at fixed in-degree a finite-size effect, does the theory follow the whole composition curve and not only three fractions, and do the single-neuron and lag-resolved predictions hold when measured directly?

## Method

The model, ensembles and estimator are those of the earlier campaigns: positive-erf rates φ(x) = 2Φ(x), unit leak, strict presynaptic Dale signs, reference couplings G⁰ = [[16, 9], [25, 12]] and J⁰ = [[1, −6], [2, −10]] at f⁰ = (0.8, 0.2), target rates 0.3, midpoint step 0.05, 500 leak times of burn-in, 8,000 recorded and sampled every 0.25, and the four-window mean-difference covariance-energy estimator with window 500 and no gap. All seed labels are new. The parameter-only predictions are computed first and fix each network's order-one drive; no network statistic enters them.

| Family | Settings | Networks per setting | Seeds |
| --- | --- | --- | --- |
| Degree | dispersion χ in {0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3}; 80:20; N = 2,000 | 24 | 41001–41024 |
| Size | N in {500, 1,000, 4,000, 8,000} at χ in {0, 0.5, 1, 2} (80:20) and at 60:40 Bernoulli | 48, 32, 12, 6 | 42001– |
| Composition | f_E in {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94}; Bernoulli; presynaptic columns rescaled so G diag(f) and J⁰ diag(f) stay fixed; N = 2,000 | 48, matched across fractions | 43001–43048 |
| Record | χ in {0, 1, 2} at 80:20 and 60:40 Bernoulli; 32,000 leak times with windows 500, 1,000 and 2,000; the same graphs and initial conditions again at steps 0.025 and 0.125 | 8 | 44001–44008 |
| Generality | χ = 1, 80:20, coupling variance sG⁰ with s in {0.5, 0.75, 1.5, 2, 3}; or target rate 0.15 or 0.5 at s = 1; N = 2,000 | 16 | 45001–45016 |

The N = 2,000 points of the size series are the degree and composition networks at the same settings. Dispersions 2.5 and 3, coupling scales other than one and rates other than 0.3 have not been simulated before.

Extra measurements on the retained trajectory, fixed here: readouts of 25, 50, 100 and 200 cells per class in the composition family and 100 cells elsewhere (eight draws, the confirmation experiment's priority rule, nested in K); rate and input autocovariances of 192 sampled neurons per population out to 40 leak times; cross-fitted two-lag pair moments on those neurons for the first eight seeds of each setting (lags −24 to 24); whole-record covariance eigenvalues for the first eight seeds at N ≤ 2,000; the largest Lyapunov exponent (one tangent vector, 2,000 leak times) for the first six seeds of each degree, composition and generality setting and a few size settings; and each neuron's time-averaged input, input variance, rate variance, gain and realized in-degree for every network.

## Controls

Before renting anything: replay one archived dense-sweep network through the new runner and require an identical record; check a whole-population readout against the engine's block energy; check the lag estimator against a brute-force loop; check the theory helper against the archived predictions. During collection: compare SHA-256 of every file on each machine with its local copy before stopping the machine.

## Reporting rules

The unit of uncertainty is a network. Every planned network is kept. A network that settles to a fixed point has no defined PR; it is counted and reported, not replaced and not averaged in. Composition contrasts are paired by seed and reported with two-sided 95% Student-t intervals. The readout contrast between 80% and 88% E was sized at 624 triplets in the earlier confirmation; 48 pairs are not expected to resolve it, and this campaign reports its interval without reinterpreting that result. Finite-size trends are described with fitted exponents over the sizes run, without claiming an asymptotic law.

## Time and cost

Seven or eight 32-vCPU `cpu5c` machines in EUR-IS-1 at $1.12 per hour each, with a 2.5-hour self-stop timer on each. Estimated 96 core-hours. Machines are stopped after verified collection and left for the owner to delete.

## Amendments during execution

* One machine's host was at least five times slower than the others and had produced no records after three minutes. It was stopped and its share of the manifest was run on a replacement machine; nothing from the slow machine was kept.
* After the size series to N = 8,000 showed the fixed in-degree deficit shrinking with size, twelve networks at N = 16,000 were added (χ = 0, 1, 2; four seeds each; `jobs_extension.json`). This decision used the N ≤ 8,000 outcomes.
* The fixed-point mode analysis was repeated on 144 networks with the replication's code unchanged; its design is in [spectral/PROTOCOL.md](spectral/PROTOCOL.md).
* The size-series figures use across-network medians with bootstrap standard errors instead of means, chosen after seeing that a few weakly fluctuating 500-neuron networks have ill-conditioned PR ratios. Means are kept in `summary.json`.
* The self-stop timer was armed on only one machine at launch because the launch script's "already armed" test matched its own command line. The other six ran about 40 minutes without a timer before this was noticed and corrected; no machine ran unattended.
