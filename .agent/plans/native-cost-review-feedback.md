# Native-cost routing review cleanup

Status: in progress. Ownership and pricing changes require separate commits,
contract tests, and matched comparisons against `b284be69e`.

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

## Work remaining

- [x] Localize ownership and tracker construction without changing routes.
- [ ] Simplify node initialization and correct positive prefix adjustments.
- [ ] Validate pass reuse, cold/hot agreement, numerical costs, and semantics.
- [ ] Run affected suites, GCC/Clang checks, and required lint.
- [ ] Compare archived quality, timing, and Grover resource cohorts.

## Acceptance

Preserve successful compilations and semantics. Against `b284be69e`, permit at
most 1% aggregate native-count regression separately for flat and structured
inputs, and 3% aggregate wall-time regression separately for both timing
cohorts. Report individual losses, SWAPs, depth, CPU, and RSS. Keep benchmark
artifacts outside the repository and leave untracked work intact.

The ownership change passes all 125 mapping tests, including pass-manager reuse
across target environments, seeds, and threading modes. Broad comparison and
final lint remain pending.
