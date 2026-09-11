# Add target-aware Qiskit export

Status: historical implementation record.

## Goal and scope

MQT target compilation produces programs whose static qubits refer to sites of a
`CompilerTarget`. A caller can now pass that target to `QCProgram.to_qiskit`.
The exporter then returns a canonical physical Qiskit circuit with one register
named `q`, one qubit for every target site, and dense Qiskit indices that match
the order of `CompilerTarget.sites`. Generic export without a target keeps its
existing behavior.

## Constraints

- A target site ID is not a Qiskit qubit index. A target may use sparse IDs such
  as 10 and 4294967296 while containing only two sites. Evidence:
  `CompilerTarget::vertexForSite` maps those IDs to compiler vertices 0 and 1.

- Qiskit uses one owning quantum register named `q` for its canonical physical
  circuit form. Layout metadata is independent of that register representation.

- Target compilation does not store the routing layout in the returned
  `QCProgram`. Correct Qiskit layout metadata therefore needs a separate design
  and is outside this work.

- CUDA-Q represents device topology with a module-level `quake.wire_set` and
  records mapping permutations in the function attributes `mapping_reorder_idx`
  and `mapping_v2p`. This is a precedent for a separate routing-layout retention
  design; it does not change the target-aware export contract in this plan.

## Decisions

- Make `CompilerTarget` an optional argument to export instead of adding target
  metadata to the MLIR module. Rationale: the target already owns the number and
  order of its sites, and the caller can supply the same immutable target used
  for compilation.

- Map `qc.static` site IDs through `CompilerTarget::vertexForSite`. Rationale:
  Qiskit physical-qubit indices are consecutive and zero-based, while target
  site IDs can be sparse.

- Reject dynamic quantum allocation during target-aware export. Rationale: a
  dynamic qubit has no target site and therefore no defined physical-qubit
  index.

- Do not emit Qiskit layout metadata. Rationale: a canonical physical register
  describes the circuit's physical-qubit space but does not describe the
  compiler's initial or final logical-to-physical permutation.

## Outcome and validation

Target-aware export now uses the supplied `CompilerTarget` directly and stores
no target data on MLIR modules. The binding build succeeded. All 141 tests in
the two affected Python files passed, including the five focused target-aware
cases. All 133 compiler tests, generated-stub comparison, repository lint, and
`git diff --check` also passed. The warning-as-error documentation build and
focused include-cleaner checks passed.

## Code and ownership

`mlir/include/mlir/Compiler/Target.h` defines `mlir::CompilerTarget`. Its
`sites()` method returns sites in compiler-vertex order, `numQubits()` returns
their count, and `vertexForSite()` maps a target-defined site ID to that dense
order.

`bindings/mlir/qiskit/QiskitExport.cpp` collects quantum and classical resources
from a `QCProgram` and constructs a Qiskit `QuantumCircuit` through Qiskit's C
API. `bindings/mlir/qiskit/Qiskit.h` declares the native exporter.
`bindings/mlir/register_mlir.cpp` exposes it as `QCProgram.to_qiskit`.
`python/mqt/core/mlir.pyi` is the generated Python type stub.

Mapping replaces dynamic QCO allocations with `qco.static` operations that use
target site IDs. QCO-to-QC conversion preserves those IDs as `qc.static`.
Cleanup can remove unused static qubits, so the remaining program alone cannot
describe the full target. The target argument supplies the authoritative site
set and order during export.

## Acceptance

The end-to-end test must return a five-qubit Qiskit circuit after compilation
uses only two sites of a five-site target. The circuit must have exactly one
quantum register named `q`, and `circuit.layout` must be `None`.

For a target with sites 10 and 4294967296, an MLIR operation on
`qc.static 4294967296` must use Qiskit qubit index 1 in a two-qubit circuit. A
`qc.static` site outside the target must fail with a controlled error. A
target-aware export containing `qc.alloc` or a quantum `memref.alloc` must also
fail. Existing target-free round-trip tests must continue to pass.

The generated stub must contain the keyword-only optional target argument. The
documentation build, focused test suites, repository lint, and
`git diff --check` must pass or have a recorded environment limitation.

## Interfaces

`mqt::bindings::qiskit::exportCircuit` accepts
`const mlir::CompilerTarget* target = nullptr`. The Python method accepts a
keyword-only `CompilerTarget | None`. The target is immutable and remains alive
for the synchronous export call; the exporter does not retain it.

The implementation uses existing `CompilerTarget::numQubits()` and
`CompilerTarget::vertexForSite()` methods. It uses the existing Qiskit 2.5 C API
translation layer and introduces no dependency.
