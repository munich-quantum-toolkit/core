# MLIR audit resolution

Status: complete. The remaining comparator, replay, and test-matrix findings in
PR #2502 are implemented and locally validated.

Structural comparisons use upstream `OperationEquivalence` with consistent SSA
mapping, including forward references across blocks. Parser/print round trips
use it directly. The permutation helper retains wire, allocation, and
independent operation reordering; yielded values follow mapped parent results.
Strict comparison is also its fast path. Neither comparator applies numerical
tolerance. Greedy permutation fallback retains its documented ordering and
complexity limits.

Every transform used by driver pipelines is registered, including shrink and QIR
cleanup/metadata passes, so late base/adaptive QIR failures replay. Malformed
modifier coverage stays in the QC verifier: all 16 operation kinds, three
modifiers, and direct/nested bodies remain. The redundant outer
allocation-origin axis is removed; separate capture cases remain.

Ponytail Review removed redundant identity mappings. All 1,575 affected CTest
entries, repository lint, and C++ lint against the fixed PR base pass. The
[matched benchmark][benchmark] records the structural fast path improvement and
its limits. The [audit record][audit] retains regression locations, supported
boundaries, and current validation separately from historical results.

[audit]: ../audits/mlir-tests-diagnostics.md
[benchmark]: ../benchmarks/pr2502-comparator/README.md
