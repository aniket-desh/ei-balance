# Validation and interpretation corrections

The first completion-validator draft assumed the campaign, full extension,
and degree-only extension manifests were disjoint. The full extension also
contains its six degree cases. That assertion failed before any scientific
comparison ran. The corrected validator unions names, requires identical
model and measurement settings for duplicate schedule references, and checks
64 unique primary realizations. The initial source is retained as
`validate_completed_campaign_v1_manifest_assumption.py`.

The partial fixed-degree variance source is rejected as a complete finite-N
repair. Its algebra, numerical undershoot, omitted mixed covariance, and
coherent/endogenous terms are retained in `partial_candidate_status.md` and
`chi0_partial_static_source.tex`.

The selected half-step checks do not establish negligible time-step error in
every N1000 network. One outlier changes substantially. Three longer native
recordings show comparable same-step backend phase/record sensitivity and
still do not establish step convergence. The primary cohorts are unchanged;
all checks and the remaining ambiguity are reported.
