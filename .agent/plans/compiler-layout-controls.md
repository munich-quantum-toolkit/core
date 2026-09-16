# Native layouts and SDK layout interchange

Status: complete. Draft PR #2553 is the publication record.

## Scope and ownership

Extend #2553 to resolve #2070. Keep the existing native initial-placement API
and detached `MappingResult`. Add frontend-neutral serialized metadata for
logical input resources, their initial physical positions, an optional routing
permutation, source register groups, and the physical output order. The existing
version-specific adapter alone reads and reconstructs SDK objects.

Support initial and final layouts, partial assignments, physical gaps, ancillary
inputs, and physical/output orders that differ from logical register order.
Reject references to missing resources and inconsistent metadata. Keep the
existing supported circuit-operation and register-membership boundary.

The layout APIs accept the shared `CompilationOptions` from target compilation,
including seed, timing, statistics, mapping trials, refinement iterations,
routing lookahead, and search memory limits. `initialLayout` remains an explicit
input assignment; `MappingResult` remains a detached output.

The branch builds directly on `main` after #2551. Low-level pipeline builders
accept `MappingOptions`; `runWithCompilationOptions` applies the shared seed and
instrumentation when the pass manager runs. Layout preservation uses the shared
`runWithPassManager` helper.

## Lifetime and output rules

Copies, MLIR serialization, and plain QC/QCO conversions preserve imported
layout metadata. Resource-changing and arbitrary transformation pipelines
conservatively invalidate it. Invalidated layouts cannot be exported as valid
SDK layouts. Provide an explicit discard operation for callers who no longer
need provenance. OpenQASM, QIR/LLVM, and jeff output reject retained layout
metadata until the caller discards it. Low-level native pipeline entry points
must use the same invalidation and output checks as the program APIs.

The layout snapshot represents circuit-wire provenance, not an executable gate
or a persistent identity for SSA values. A transformation that preserves a
layout must preserve its resource correspondence; arbitrary external IR edits
must update or invalidate this discardable metadata.

## Tracking without changing compilation

Native tracking records source-index arrays on allocations in the existing
target-preparation pass, avoiding an extra pass and its IR verification. Tensor
shrinking selects the entries for retained slots. Discovery keeps its ordinary
wire order; removed inputs occupy unused permutation entries and can follow
routing workspace swaps without extra operations. All-to-all placement creates
only live wires and retains the indexed-placement path. Publication moves the
completed snapshot to the caller only after compilation succeeds.

Automatic tracking must preserve the ordinary compiled circuit. Regression
checks compare exact IR for routed and all-to-all workloads, including wide idle
tensors and both payload profiles. A complete-unitary test includes an
optimized-away source slot. Explicit placement and empty or entirely idle
allocations retain their source slots in the returned snapshot.

## Validation

The compiler, translation, and QDMI Python suites pass 642 tests. Native
validation passes 242 compiler tests, 115 mapping tests, 38 metadata tests, and
two tensor transform tests, including shared compilation options, routing with
zero lookahead and zero or small search memory budgets, failure publication, and
schema checks. The compiler suite includes four CLI GoogleTests; all three CLI
CTests pass. Both published Python examples execute. Stub generation, repository
lint, and full changed-file C++ lint against `origin/main` pass.

The implemented metadata uses a validated `mqt.layout` dictionary and a mutually
exclusive `mqt.layout_invalidated` unit marker. The version-specific adapter
supports `TranspileLayout`, including implicit output order; bare `Layout`
values are rejected. No opaque Python state enters the compiler.

Earlier native patch coverage at `3178e825c` was 419/438 production lines
(95.7%), including all 130 lines of `QubitLayout.cpp`. The shared-options update
adds regression coverage without changing coverage thresholds or exclusions.
