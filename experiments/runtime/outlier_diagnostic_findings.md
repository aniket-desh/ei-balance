# A bounded numerical check of the atypical N1000 network

All three follow-ups use the original network, original initial condition,
original drive, and the native float64 action. The primary cohort is unchanged.

| dt | T4000 PR/N | T8000 PR/N | T16000 PR/N |
|---|---:|---:|---:|
| 0.05 | 0.01819187 | 0.01836071 | 0.01908510 |
| 0.025 | 0.01791820 | 0.01939911 | 0.01863594 |
| 0.0125 | 0.01769614 | 0.01560838 | 0.01700227 |

The T8000 half-step result reproduces the earlier half-step control exactly in
nu, energy, and PR. Replacing only the mathematically equivalent sparse action
at dt=.05 changes T8000 PR/N from the original SciPy value 0.015601 to native
0.018361. The action equivalence checks are accurate to approximately 1e-14;
long chaotic trajectories need not agree after floating-sum reordering.

At T16000, the three PR estimates still span 0.017002 to 0.019085. The finest
step does not establish a clean convergence sequence. Trace and covariance
energy also vary materially. Thus the original 24% half-step change cannot be
assigned solely to integration bias. These tests expose arithmetic/trajectory,
record-length, and possibly attractor-state sensitivity in this one atypical
N1000 network; they do not separate those causes or establish a converged
invariant covariance.

The result remains a limitation. Every primary and auxiliary case is retained.
No extra basin or numerical-convergence campaign is launched. In contrast, the
selected N4000 degree endpoint changes PR by 1.05% under a half step, and the
other three selected N1000 PR checks change by at most 0.49%. Those successful
checks do not erase this exception or prove every realization is converged.
