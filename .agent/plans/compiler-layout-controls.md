# Native layouts and SDK layout interchange

Status: implementation and validation complete. Draft PR #2553 is the
publication record for this extension.

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

## Completion

- [x] Shared metadata representation, validation, and explicit discard API.
- [x] Supported SDK import/export forms with no compiler dependency on Qiskit.
- [x] Transformation invalidation and unsupported-format checks.
- [x] Native and SDK regression tests, docs, changelog, and stubs.
- [x] Final full-file C++ lint.

## Validation

Existing native target/mapping and Python layout regressions cover placement,
routing, tensor order, idle slots, and failure publication. Add issue #2070's
round-trip, transformation, partial-layout, ancilla, and output-format cases.
Run the supported Qiskit adapter at its minimum and installed patch versions.

The implemented metadata uses a validated `mqt.layout` dictionary and a mutually
exclusive `mqt.layout_invalidated` unit marker. The Qiskit 2.5 adapter supports
`TranspileLayout`, including its implicit output order; bare `Layout` values are
explicitly rejected. No opaque Python state enters the compiler.

Validation: 516 Python compiler/translation tests passed with the native
SC-provider registry, 18 layout tests passed on Qiskit 2.5.0, 234 compiler tests
and 34 metadata tests passed under coverage instrumentation, and CLI
layout/discard checks passed. Generated stubs are current. Final checks also
cover direct native serializer rejection and isolated CLI pipelines.

Final native patch coverage is 419/438 executable production lines (95.7%),
including all 130 executable lines of `QubitLayout.cpp`. No threshold or
exclusion changes were needed. Both published usage examples execute.
