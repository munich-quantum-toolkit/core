# Preserve reusable Qiskit gates in QC

Status: complete.

## Goal and scope

Import nonstandard Qiskit `Gate` definitions as private QC unitary functions and
preserve their applications as `qc.call`. Export supported private unitary
functions and calls as Qiskit Gates. Nested and repeated definitions must stay
linear in the size of the definition graph and preserve parameters, public
names, global phases, qubit order, and supported modifiers.

The normalized boundary is `bindings/mlir/qiskit/QiskitTranslation.h`. The
Qiskit 2.5 implementation is in `bindings/mlir/qiskit/Qiskit2_5.cpp`;
format-independent import and export are in `QiskitImport.cpp` and
`QiskitExport.cpp`. Behavioral coverage is in
`test/python/test_mlir_qiskit_translation.py`.

Generic Qiskit `Instruction` definitions remain flattened. They can contain
classical bits, measurement, reset, and control flow, for which the unitary QC
function ABI is not valid. Gate functions cannot contain classical bits,
standalone classical variables, barriers, or nonunitary operations.

## Decisions

- Use Qiskit's public `Gate` type as the reusable-function boundary. Determine
  built-in gates from Qiskit's standard-gate identity, not from user-controlled
  names.
- Build a function from the bound definition circuit's remaining free
  parameters. Qiskit specializes copied definitions during parameter binding, so
  recovering an erased generic template would require private provenance.
- Intern definitions by source name and parameter hash, then use object identity
  and Qiskit circuit equality within that bucket. Qiskit deep-copies a Gate and
  its definition when appending it, so pointer identity alone duplicates equal
  functions. Signature buckets avoid a global quadratic equality scan.
- Unique colliding MLIR symbols deterministically and retain the Qiskit name in
  `mqt.source_name`. The first available symbol keeps the source name.
- Preserve inverse, closed-control, and finite numeric power modifiers around
  custom Gate calls. Reject open controls and symbolic powers that Qiskit 2.5
  cannot bind reliably.
- Use MLIR `CallGraph` and LLVM SCC traversal for callee-first export order and
  recursion rejection. Reject unreachable functions because a Qiskit circuit
  cannot retain unused Gate declarations.
- Retain the 64-level definition and exported call-depth limits. Qiskit exposes
  mutable Python definition graphs with arbitrary cycles, and parameterized
  `QuantumCircuit.to_gate(parameter_map=...)` recursively copies nested
  definitions. The export limit also keeps emitted circuits within the
  importer's supported range.
- Construct custom Gates in the version-specific bridge through the existing
  deferred-placeholder pattern. The Qiskit C API does not append arbitrary
  Python Gates or `AnnotatedOperation` values.
- Do not add a classical callable ABI. Standalone variables and control-flow
  captures remain entry- or block-owned and do not enter unitary functions.

## Validation

From the repository root, build and test with:

    cmake --build --preset release --target mqt-core-mlir-bindings -j2
    pytest -n0 -q test/python/test_mlir_qiskit_translation.py
    uvx nox -s stubs
    uvx nox -s cpp-lint
    uvx nox -s lint

The focused file passes all 294 tests. Stub generation, repository lint, and C++
lint pass; the C++ lint session performs a clean build before running
clang-tidy. Hosted CI is separate evidence and must run on the final published
commit.

## Outcome

The implementation uses the existing QC function model, Qiskit public Gate
model, MLIR call graph, and normalized translation boundary. It adds no dialect
operation, external dependency, generic call-graph framework, or classical
function ABI. A parameterized QC helper can split into specialized helpers on a
Qiskit round trip because Qiskit's public circuit retains the specialized
definitions rather than their erased template.
