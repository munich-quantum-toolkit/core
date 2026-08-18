# Preserve dense unitary operations across the compiler collection

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core must represent a numeric unitary matrix as one frontend-neutral
operation instead of asking Qiskit to synthesize that matrix during import.
After this change, users can import and export Qiskit `UnitaryGate` operations
without losing the dense matrix. QC and QCO programs can also verify, convert,
serialize, simulate, modify, and synthesize the same operation through their
existing unitary interfaces.

## Progress

- [x] (2026-08-17 18:00Z) Audited the existing matrix, unitary-interface,
  decision-diagram, native-synthesis, and Qiskit bridge support.
- [x] (2026-08-17 21:20Z) Added verified `qc.unitary` and `qco.unitary`
  operations and builders.
- [x] (2026-08-17 21:25Z) Preserved dense unitaries during QC-to-QCO and
  QCO-to-QC conversion.
- [x] (2026-08-17 21:35Z) Imported and exported Qiskit dense unitaries with
      explicit qubit-order conversion.
- [x] (2026-08-17 21:45Z) Added dialect, conversion, synthesis, simulation,
  modifier, serialization, and Qiskit regressions.
- [x] (2026-08-17 21:55Z) Built the affected C++ and Python targets, ran the
  focused and full affected tests, confirmed that no stub signature changed,
  and ran the repository lint session.

## Surprises & Discoveries

- Observation: QC and QCO interpret operation operand zero as the most
  significant matrix bit. Qiskit interprets the first `UnitaryGate` qubit as the
  least significant matrix bit. Import and export must therefore apply the same
  row-and-column bit permutation at both boundaries.
- Observation: existing native synthesis accepts compile-time one- and two-qubit
  matrices through `qco::UnitaryOpInterface`. Mapping does not accept operations
  wider than two qubits. Wider matrices remain representable and exportable but
  cannot yet pass target compilation.
- Observation: deterministic unitarity checking is cubic in the matrix side
  length. The verifier therefore rejects operations wider than eight qubits
  before materializing or scanning matrix entries.
- Observation: Qiskit 2.5's native instruction accessor aborts on wrapped dense
  unitaries because their Python matrix parameter is not scalar. The importer
  must identify and reject those wrappers before calling the native accessor.

## Decision Log

- Decision: Store matrices as rank-two dense `complex<f64>` tensor attributes in
  row-major order. Verify shape, finite entries, dimension, and unitarity at the
  operation boundary. Rationale: this is the native MLIR representation used by
  the existing matrix and synthesis infrastructure.
- Decision: Preserve dense operations instead of expanding Qiskit's circuit
  definition. Rationale: definition expansion is frontend-specific, increases IR
  size, and becomes redundant once QC and QCO can represent the matrix.
- Decision: Canonicalize only exact identity matrices. Rationale: tolerance-
  based rewriting can change program semantics.
- Decision: Accept dense operations on one to eight qubits and verify
  `U^dagger U` entry-wise with absolute tolerance `1e-10`. Rationale: eight
  qubits bounds the deterministic cubic check while covering the required Qiskit
  and Quantum Volume workloads.

## Milestones

### Add dialect operations and conversions

Define `qc.unitary` and `qco.unitary` in the QC and QCO operation tables. Add
shared verification for a nonempty square matrix whose dimension is `2^n`, whose
entries are finite, and whose conjugate transpose times the matrix is the
identity within the documented verifier tolerance. Implement builders and
preserve the matrix and operand order in both QC/QCO conversion directions.
Focused QC, QCO, and conversion tests must parse valid operations and reject
wrong shapes, nonfinite values, and nonunitary matrices.

### Connect Qiskit and the existing unitary infrastructure

Read and write Qiskit's native matrix through its C API. Convert the matrix bit
order at the boundary so Qiskit and QC/QCO apply the same operator to the same
qubits. Keep one-, two-, and three-qubit matrices dense during a Qiskit
round-trip. Prove one- and two-qubit target synthesis, modifier handling, and
decision-diagram semantics with focused tests. A seeded Quantum Volume circuit
must import without expanding its two-qubit matrices.

### Validate the complete change

From the repository root, build the affected MLIR and Python binding targets.
Run the focused QC, QCO, conversion, native-synthesis, decision-diagram, and
Python Qiskit tests. Run `uvx nox -s stubs` when binding signatures change, then
run `uvx nox -s lint` and `git diff --check`. Record concise pass or failure
evidence below.

## Outcomes & Retrospective

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
