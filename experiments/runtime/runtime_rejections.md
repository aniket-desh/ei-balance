# Runtime attempts and checks

The original all-block covariance estimator required nb*N*N float64 storage
in resident arrays. It was replaced by an exact streamed spatial-tile Gram
contraction using covariance files on scratch disk. No random projection or
pair sampling was introduced. Full population identities were checked against
literal dense formulas and the earlier estimator.

A first v2 ingredient test detected a variable-name collision between gain
block means and the inhibitory index group. The gain array was renamed before
any v2 network jobs ran. The corrected independent-Gaussian test agrees with
the earlier mean-free estimator to 6e-17. This was an implementation error,
not evidence for or against the theoretical closure.

Dense BLAS with 2 or 4 threads per network was slower overall than four
single-thread sparse-complement workers. The original scalar C++ action only
modestly improved larger cases and could lose at N1000 from Python-call
validation overhead. Caching fixed array pointers removed that overhead.
The optional SIMD version reorders double-precision sums. Its random-vector
action errors are below 2e-14 and 50-step trajectory errors below 4e-14 in the
benchmarks. Those short checks do not imply equality of long chaotic
trajectories. Both scalar and SIMD benchmark results and source remain.

The degree and composition campaigns use mathematical predictions fixed before
outcomes. Runtime improvements do not change coupling matrices, transfer
function, drives, the midpoint scheme, or primary recording estimators.

V2 first N4000 analysis used 1.424 GB peak resident memory. This was safe for
the first two large cases but would restrict four-case parallelism. V3 releases
raw map pages after the first mean and FFT passes, releases written covariance
pages immediately, and uses in-place covariance scaling. All 612 numerical
fields match V2 bitwise in a complete N4000 reanalysis, with peak RSS 645 MB.
The active V2 first wave was not changed. A replacement continuation controller
was prepared, but the original had already advanced to the two-worker V2 pool.
The replacement was canceled before it launched any jobs. The original safe
two-worker pool continues; no active network was interrupted. V3 remains a
validated lower-memory alternative used by the portable launcher, not the
backend claimed for the current large-case results. Both source versions,
memory checks, and controller record remain.

A fresh-export audit found the optional native builder assumed `tmp/` existed.
The builder now creates its output directory. A clean temporary-directory build
and action check passed without modifying the active native library. The
portable V4 export also catches an unsupported `madvise` OSError and applies a
more conservative macOS worker-memory estimate. Its numerical calculations are
unchanged from the memory-validated V3. macOS execution remains untested.
