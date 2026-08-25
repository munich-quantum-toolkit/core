# Export forwarded Qiskit measurement results

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core converts a QC measurement and its following CBit store to one Qiskit
measurement instruction. CBit cleanup can replace a later load from that CBit
with the measurement's SSA result. The result then has a store use and a later
classical-expression use. After this change, `QCProgram.to_qiskit()` exports
that form by using the measurement's destination CBit in the Qiskit expression.

The destination store must remain directly after the measurement, apart from
constants. This rule makes the store precede every other supported use.

## Progress

- [x] (2026-08-19 20:54Z) Reproduce the cleanup-forwarded measurement condition
      from the Benchpress integration.
- [x] (2026-08-25 08:37Z) Merge the current structured-control exporter and
      current `main`, and resolve the old stack in favor of the reviewed base.
- [x] (2026-08-25 08:48Z) Remove delayed-store scheduling support that belongs
      to the mapping pass, retain strict adjacency, and reduce the exporter to
      one result-to-CBit lookup.
- [x] (2026-08-25 08:48Z) Add one focused cleanup regression and update the
      exporter support documentation.
- [x] (2026-08-25 08:50Z) Build the release binding, pass the focused regression
      and all 219 translation tests, regenerate unchanged stubs, pass lint, and
      inspect the final parent-relative diff.
- [x] (2026-08-25 08:52Z) Prepare focused gitmoji commits and a pull request
      description that records the reduced scope and current validation.

## Surprises & Discoveries

- Observation: The failing cleanup program already has an adjacent measurement
  and store. Evidence: its QC IR is `qc.measure`, `cbit.store`, and then an
  `scf.if` whose condition contains the measurement result. The old
  `hasOneUse()` check caused the misleading adjacency error.

- Observation: The earlier sparse-target failure came from the mapping pass's
  generic topological sort. Evidence: the independent mapping change replaces
  that sort with a quantum-wire traversal that places each result producer
  before its earliest classical user. The exporter does not need a second
  scheduler.

- Observation: Successful Qiskit export already forbids every write that could
  replace the recorded destination. Evidence: measurement destinations must be
  unique, dynamic destinations fail, and non-measurement CBit stores fail.
  Separate destination-snapshot bookkeeping is therefore unnecessary.

## Decision Log

- Decision: Remove only the one-use check and keep the existing same-block
  adjacency check. Rationale: strict adjacency already places the store before
  every additional supported use. Date/Author: 2026-08-25 / Codex.

- Decision: Record only the public CBit index for each accepted measurement
  result. Rationale: the existing destination validation prevents overwrites in
  every program that can reach Qiskit construction. Date/Author: 2026-08-25 /
  Codex.

- Decision: Do not accept quantum operations, later measurements, or reversed
  stores between a measurement and its destination. Rationale: the mapping pass
  owns operation ordering, and accepting those forms in the exporter duplicates
  that fix. Date/Author: 2026-08-25 / Codex.

## Outcomes & Retrospective

The old branch mixed two problems: mapper scheduling and cleanup-forwarded SSA
uses. The current change keeps only the exporter problem. The focused cleanup
test fails on the reviewed parent with
`QC measurement destination must follow the measurement in the same block` and
passes with this change. The release binding builds, all 219 translation tests
pass, stub generation produces no diff, and repository lint passes.

## Context and Orientation

`bindings/mlir/qiskit/QiskitExport.cpp` validates a complete `ExportedCircuit`
before it creates a Qiskit object. `collectBlock` finds the unique static CBit
store for each `qc.measure`. `exportExpressionImpl` converts supported MLIR
classical values to Qiskit expressions. `ExportState` carries validated resource
indices between those functions.

`test/python/test_mlir_qiskit_translation.py` contains the end-to-end Qiskit
translation tests. `docs/mlir/python_compiler_collection.md` documents the
measurement destination and result-use rules.

## Plan of Work

In `ExportState`, map each accepted measurement result to its public CBit index.
In `isFusableMeasurementStore`, remove the one-use restriction and keep the
existing constant-only gap rule. In `exportExpressionImpl`, emit a classical-bit
leaf when the input value is a recorded measurement result.

Add one OpenQASM cleanup regression that exports two measurements and a gate
controlled by both results. Update the documentation with the post-store
expression rule.

## Concrete Steps

Run from the repository root:

    clang-format -i bindings/mlir/qiskit/QiskitExport.cpp
    cmake --build build/python/Release --target mqt-core-mlir-bindings --parallel 8
    pytest -q test/python/test_mlir_qiskit_translation.py \
      -k cleanup_forwards_measurement_results
    pytest -q test/python/test_mlir_qiskit_translation.py
    uvx nox -s lint
    git diff --check

The focused command must report one pass. The complete file must have no
failures. The binding change does not alter a public Python signature, so stub
generation must produce no diff.

## Validation and Acceptance

An OpenQASM 2 program that measures two qubits and applies `x` to a third qubit
when both results equal one must still export after `QCOProgram.cleanup()`. The
result must contain two Qiskit measurements and one `if_else` instruction.

## Idempotence and Recovery

Formatting, building, and testing are repeatable. The work remains isolated on
the measurement-result branch. If validation exposes another producer shape,
inspect its final QC IR before changing the store-order rule; do not add an
exporter scheduling policy.

## Artifacts and Notes

The regression on the reviewed parent is:

    RuntimeError: QC measurement destination must follow the measurement in the same block

The final branch changes only the measurement-result preflight, two focused
tests, this plan, and the related documentation relative to its parent.

## Interfaces and Dependencies

No public interface or dependency changes are required. The implementation uses
the existing MLIR `Value`, `Operation`, `cbit::StoreOp`, and Qiskit expression
types. Operation reordering remains outside this exporter change.

Revision note: Rewritten on 2026-08-25 after the mapping stack took ownership of
topological ordering. The plan now covers only cleanup-forwarded measurement
results.
