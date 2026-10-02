# Export structured QC control flow to Qiskit

Status: historical implementation record.

## Goal and scope

MQT Core can import Qiskit control flow into QC MLIR, but flat-only export loses
structured `if`, `for`, `while`, and `switch` operations. After this change,
`QCProgram.to_qiskit()` recreates supported structured operations recursively.
It preserves root qubits, classical bits, scalar parameters, measurement
destinations, instruction order, and nested captures. Unsupported MLIR fails
before the exporter returns a partial Qiskit circuit.

Qiskit and OpenQASM logical AND and OR use short-circuit evaluation. QC MLIR
represents these expressions as a single-result `scf.if`: AND evaluates its
right operand only in the then region and yields false from else; OR yields true
from then and evaluates its right operand only in else. The exporter accepts
only these two result-bearing forms. It does not reconstruct general Boolean
selection or multiple `scf.if` results.

## Constraints

- OpenQASM import already emits the required result-bearing `scf.if` shapes.
  Evidence: `emitCondition` in
  `mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` emits lazy AND and
  OR regions.

- The old Qiskit importer mapped logical and bitwise operations to the same
  eager `arith.andi` and `arith.ori` operations. This evaluated both logical
  operands and disagreed with Qiskit and OpenQASM semantics.

- MLIR's result-bearing `scf.if` is the native structured form for conditional
  evaluation. `arith.select` selects already-computed SSA values and cannot
  provide short-circuit evaluation.

## Decisions

- Keep the recursive normalized circuit model and version-specific Qiskit
  writer. Rationale: Qiskit 2.5 exposes public Python constructors for
  control-flow objects but no equivalent stable C API. The generic translator
  remains independent of Python objects.

- Use result-bearing `scf.if` as the shared logical short-circuit representation
  for Qiskit and OpenQASM import. Rationale: It preserves lazy evaluation and
  matches current MLIR structured-control practice.

- Export only canonical single-result Boolean AND and OR shapes. Rationale:
  General ternary and multi-result reconstruction are outside issue #2071 and
  added cloning, normalization, and independent budget machinery.

- Keep forwarded measurement-result recognition. Rationale: QC cleanup can
  replace a classical load with the measurement SSA result; both values denote
  the same validated destination Qiskit `Clbit`.

## Outcome and validation

Implementation and local validation are complete. The current design keeps the
structured export feature while removing speculative Boolean-selection support.
Compared with the previous PR head, the final diff contains 375 fewer net lines.
The release build, all 4,038 configured CTests, all 219 Qiskit translation
tests, stub generation, MLIR documentation, complete Sphinx documentation, and
the full lint session pass. One CTest is skipped by its existing test policy.

## Code and ownership

`bindings/mlir/qiskit/QiskitTranslation.h` defines the frontend-neutral circuit,
control-flow, expression, register, and parameter records shared by the generic
translator and Qiskit adapters. `bindings/mlir/qiskit/QiskitExport.cpp`
validates a `mlir::QCProgram`, recursively collects instructions, and writes the
validated model through `CircuitWriter`. `bindings/mlir/qiskit/QiskitImport.cpp`
converts captured Qiskit expressions and control flow to QC MLIR.

`bindings/mlir/qiskit/Qiskit2_5.cpp` is the only layer that constructs Qiskit
Python control-flow objects. It creates child circuits against the root bit
objects, places temporary barriers, finalizes child writers, and replaces the
barriers with Qiskit operations. `test/python/test_mlir_qiskit_translation.py`
contains the end-to-end contract. `docs/mlir/python_compiler_collection.md`
documents the supported MLIR forms.

A classical snapshot is a `cbit.load` value used by a later condition. Export
rejects the snapshot if an intervening store can make it stale. A returned
classical register initialized as undefined becomes readable only after a
validated unconditional top-level measurement writes the bit. Cleanup-forwarded
measurement results map to that same destination bit.

## Acceptance

Qiskit logical AND and OR import must contain result-bearing `scf.if`, and
round-trip export must produce structurally equivalent Qiskit expressions.
Bitwise AND and OR must remain eager `arith` operations. Nested OpenQASM AND and
OR must export to the equivalent Qiskit expression. General Boolean ternaries
and multi-result `scf.if` must fail with a clear unsupported-shape error.

Existing tests must continue to cover nested control flow, captures, loop
ranges, switches, packed registers, expression and depth bounds, snapshots,
undefined bits, and forwarded measurement conditions. Stub generation must
produce no uncommitted generated changes. The full lint session and final diff
check must pass.

## Interfaces

This change adds no public C++ or Python API. It keeps the internal
`CircuitWriter::addControlFlow` interface and the existing Qiskit 2.5 adapter.
It uses MLIR SCF, Arith, CBit, and QC dialect operations already required by the
translation code. It adds no dependency.
