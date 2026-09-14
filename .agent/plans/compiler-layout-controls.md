# Native layouts and SDK layout interchange

Status: complete. Draft PR #2553 is the publication record.

## Scope and ownership

Extend #2553 to resolve #2070. Keep the existing native initial-placement API
and detached `MappingResult`. Add frontend-neutral serialized metadata for
logical input resources, their initial physical positions, an optional routing
permutation, source register groups, and the physical output order. The existing
version-specific SDK adapter alone reads and reconstructs Qiskit objects.

Support initial and final layouts, partial assignments, physical gaps, ancillary
inputs, and physical/output orders that differ from logical register order.
Reject references to missing resources and inconsistent metadata. Keep the
existing supported circuit-operation and register-membership boundary.

The layout APIs accept the shared `CompilationOptions` from target compilation,
including seed, timing, statistics, mapping trials, refinement iterations, and
routing lookahead. `initialLayout` remains
an explicit input assignment; `MappingResult` remains a detached output.

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

## Validation

The compiler and Qiskit translation suites pass 525 tests with Qiskit 2.5.2. All
18 layout round-trip cases pass with Qiskit 2.5.0, the adapter minimum. Native
validation passes 237 compiler tests and 35 metadata tests, including shared
compilation options, routing, failure publication, and schema checks. All four
CLI CTests pass, including saved-pipeline replay and explicit discard. The
compiler suite also includes the two shared-options CLI regressions. Both
published Python examples execute. Stub generation, repository lint, and full
changed-file C++ lint against `origin/main` pass.

The implemented metadata uses a validated `mqt.layout` dictionary and a mutually
exclusive `mqt.layout_invalidated` unit marker. The Qiskit 2.5 adapter supports
`TranspileLayout`, including implicit output order; bare `Layout` values are
rejected. No opaque Python state enters the compiler.

Earlier native patch coverage at `3178e825c` was 419/438 production lines
(95.7%), including all 130 lines of `QubitLayout.cpp`. The shared-options update
adds regression coverage without changing coverage thresholds or exclusions.
