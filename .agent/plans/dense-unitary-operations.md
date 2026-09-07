# Preserve dense unitary operations across the compiler collection

Status: historical implementation record.

## Goal and scope

MQT Core must represent a numeric unitary matrix as one frontend-neutral
operation instead of asking Qiskit to synthesize that matrix during import.
After this change, users can import and export Qiskit `UnitaryGate` operations
without losing the dense matrix. QC and QCO programs can also verify, convert,
serialize, simulate, modify, and synthesize the same operation through their
existing unitary interfaces.

## Constraints

- QC and QCO interpret operation operand zero as the most significant matrix
  bit. Qiskit interprets the first `UnitaryGate` qubit as the least significant
  matrix bit. Import and export must therefore apply the same row-and-column bit
  permutation at both boundaries.

- existing native synthesis accepts compile-time one- and two-qubit matrices
  through `qco::UnitaryOpInterface`. Mapping does not accept operations wider
  than two qubits. Wider matrices remain representable and exportable but cannot
  yet pass target compilation.

- deterministic unitarity checking is cubic in the matrix side length. The
  verifier therefore rejects operations wider than eight qubits before
  materializing or scanning matrix entries.

- Qiskit 2.5's native instruction accessor aborts on wrapped dense unitaries
  because their Python matrix parameter is not scalar. The importer must
  identify and reject those wrappers before calling the native accessor.

## Decisions

- Store matrices as rank-two dense `complex<f64>` tensor attributes in row-major
  order. Verify shape, finite entries, dimension, and unitarity at the operation
  boundary. Rationale: this is the native MLIR representation used by the
  existing matrix and synthesis infrastructure.

- Preserve dense operations instead of expanding Qiskit's circuit definition.
  Rationale: definition expansion is frontend-specific, increases IR size, and
  becomes redundant once QC and QCO can represent the matrix.

- Canonicalize only exact identity matrices. Rationale: tolerance- based
  rewriting can change program semantics.

- Accept dense operations on one to eight qubits and verify `U^dagger U`
  entry-wise with absolute tolerance `1e-10`. Rationale: eight qubits bounds the
  deterministic cubic check while covering the required Qiskit and Quantum
  Volume workloads.

## Outcome and validation

MQT Core now preserves verified dense numeric unitaries on one to eight qubits
as frontend-neutral QC/QCO operations. Qiskit import and export retain one-,
two-, and three-qubit matrices and explicitly convert the Qiskit/QC bit-order
convention. Existing target synthesis handles one- and two-qubit operations;
wider operations remain representable, serializable, and exportable but are not
yet accepted by mapping.

Validation passed with 334 QC IR tests, 485 QCO IR tests, 3 QC/QCO round-trip
tests, 23 target-synthesis tests, and 102 Qiskit translation tests. A QV100
smoke test preserved all 5,000 dense operations and compiled for FakeTorino in
20.431 seconds with target validation succeeding. The full repository lint
session and `git diff --check` also passed.
