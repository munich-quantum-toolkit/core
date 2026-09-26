# Native-cost routing review cleanup

Status: complete. Ownership and pricing changes have separate commits, contract
tests, and matched comparisons against `b284be69e`.

## Scope and decisions

Keep routing methods in the mapping pass. Store invocation data in a local
context and share it read-only across trials. The numerical table has one owner;
each tracker retains its bounded local cache. Initialize trackers with their
routing states. Refinement remains unscored.

Represent standalone SWAP cost and signed prefix adjustment explicitly. The
adjustment must use the same separate-run bound as tracker accounting. Retain
uniform-cost guidance, bounded node storage, stable addresses, layout capacity
reuse, and the distance-reducing fallback. Add no public options, per-edge cost
tables, fidelity scoring, cache tuning, or operation renaming.

## Completed work

- [x] Localize ownership and tracker construction without changing routes.
- [x] Simplify node initialization and correct positive prefix adjustments.
- [x] Validate pass reuse, cold/hot agreement, numerical costs, and semantics.
- [x] Run affected suites, GCC/Clang checks, and required lint.
- [x] Compare archived quality, timing, and Grover resource cohorts.

## Acceptance

Preserve successful compilations and semantics. Against `b284be69e`, permit at
most 1% aggregate native-count regression separately for flat and structured
inputs, and 3% aggregate wall-time regression separately for both timing
cohorts. Report individual losses, SWAPs, depth, CPU, and RSS. Keep benchmark
artifacts outside the repository and leave untracked work intact.

All 1001 affected tests and three CLI checks pass. The positive-adjustment
regression fails on the previous calculation in both operand orders. Changed
implementation and test files compile with GCC and Clang. Whole-file C++ lint
and repository lint pass.

Both commits preserve emitted IR, native counts, SWAP counts, and depth across
868 flat, 106 structured, and 45 separately reported largest paired successes.
The 24 shared unroll-limit cases remain unchanged. All semantic checks pass.

Aggregate wall time increases by 2.03% in the seven-case cohort and 2.60% in the
92-case sweep. CPU changes by -0.20% and +0.91%; peak RSS by -0.03% and -0.11%.
Both timing gates pass. Individual losses remain visible: the largest parallel
Grover setting rises from 3.91 to 4.22 seconds (+7.93%), with CPU +3.05% and RSS
-0.07%. Three repetitions do not establish statistical significance.

Raw records, hashes, exclusions, plots, and the full report are outside the
repository in `/home/nvidia/.codex/experiments/pr2607-review-feedback-20260924`.
The original untracked files are preserved.
