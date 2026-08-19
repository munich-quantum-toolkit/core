# Export measurement destinations across independent target work

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

Target compilation can leave a measurement result's CBit store after reset,
unitary operations, or later measurements and their destination stores. Qiskit
represents a measurement and its classical destination as one instruction, so
its write is immediate. `QCProgram.to_qiskit()` accepts that delayed MLIR store
when every intervening operation is independent of its destination bit. The
Qiskit instruction order retains the measurement before the intervening work
while attaching the destination to it.

QC cleanup may also forward a load from the just-written CBit to the measurement
SSA result. The result then has both a destination-store use and classical-
expression uses, so a one-use restriction incorrectly rejects common
measurement-controlled programs. The exporter treats such a result as its unique
destination CBit only when the store precedes every other use. Snapshot
validation rejects intervening or nested writes before a consumer.

## Progress

- [x] (2026-08-19 15:54Z) Split this follow-up from the completed structured
      exporter branch and inspect the earlier combined regression.
- [x] (2026-08-19 15:55Z) Relax only the measurement/store adjacency predicate
      and document its classical-state equivalence argument.
- [x] (2026-08-19 15:56Z) Add a sparse target-mapping regression and a direct
      CBit MLIR regression.
- [x] (2026-08-19 16:19Z) Restack onto the audited import and exporter parents,
      strengthen the stale-snapshot negative, and pass the complete translation
      file and repository lint.
- [x] (2026-08-19 20:15Z) Restack onto the post-merge exporter foundation after
      its scalar-input reachability fix and pass all 199 translation tests.
- [x] (2026-08-19 20:54Z) Diagnose the three Benchpress failures as forwarded
      measurement-result uses, add destination provenance and snapshot
      validation, rebuild, and pass 203 active translation tests.
- [x] (2026-08-19 21:16Z) Restack the Qiskit branches onto the latest scalar
      foundation, pass the complete translation file and lint, and validate the
      combined Qiskit/classical tree with all 4,156 native tests, all 219 Qiskit
      translation tests, and all 40 Benchpress integration tests.
- [x] (2026-08-19 21:19Z) Narrow snapshot invalidation from register-wide to
      bit-precise, add direct and nested different-bit regressions, rebuild, and
      pass all six affected tests, all 206 active translation tests, and lint.
- [x] (2026-08-19 22:52Z) Accept reversed stores for measurements with distinct
      static destination bits, add alias regressions, build the binding, pass
      all 209 active translation tests, and pass repository lint before
      independent audit.

## Surprises & Discoveries

- Observation: The sparse-target example retains all three measurements but may
  schedule synthesized quantum gates before a measurement result's CBit store.
  Evidence: the strict exporter rejected the compiled program even though those
  intervening gates cannot access classical state.

- Observation: A Qiskit measurement always owns its destination Clbit and
  therefore writes it immediately. Evidence: the normalized writer API exposes
  `addMeasure(qubit, clbit)` rather than separate measure and store operations.

- Observation: CBit canonicalization forwards a load after a static store to the
  stored SSA value. Evidence: target-independent `QCOProgram.cleanup()` reduces
  `measure; store; load; if` to `measure; store; if %result`.

- Observation: The three post-merge Benchpress failures contain no operation
  between `qc.measure` and `cbit.store`. Evidence: the store is adjacent and a
  later `scf.if` is the result's second use; `hasOneUse()` alone produced the
  misleading adjacency error.

- Observation: A mapped three-qubit W-state can place two complete measurement
  and store pairs between an earlier measurement and its store. Evidence: the
  final QC program orders the destinations as
  `measure a; measure b; store b; measure c; store c; store a`, and all three
  stores target distinct static bits.

## Decision Log

- Decision: Accept `arith.constant`, `qc.measure`, `qc.reset`, and
  `qc::UnitaryOpInterface` operations between a measurement and its CBit store.
  Also accept another measurement's store when both destinations are static and
  distinct. Rationale: these operations cannot observe or overwrite the delayed
  destination, so attaching that destination to the earlier measurement is
  unobservable. Date/Author: 2026-08-20 / Codex.

- Decision: Keep control flow, dynamic or aliasing stores, other CBit
  operations, other arithmetic, memory effects, and unknown operations
  fail-closed in that gap. Rationale: any may observe, overwrite, or condition
  behavior on the classical result. Date/Author: 2026-08-20 / Codex.

- Decision: Record the unique static destination bit and store for each accepted
  measurement result during export preflight. Rationale: later supported
  expressions can use the Clbit that the measurement already writes, without
  mutating the source or allocating a partial output circuit. Date/Author:
  2026-08-19 / Codex.

- Decision: Require the destination store to precede every non-store use and
  apply classical-snapshot overwrite checks from that store to each consumer.
  Rationale: substituting a mutable Clbit for an immutable SSA result is valid
  only while the destination still contains that result. Date/Author: 2026-08-19
  / Codex.

- Decision: Guard the exact target-compiled QASM regression when the independent
  Qiskit branch lacks the sibling classical-control capability/mapping stack.
  Rationale: main's pre-stack mapper aborts on `qco.if`; the always-running
  cleanup regression covers the exporter, and combined Benchpress validation
  supplies the end-to-end evidence. Date/Author: 2026-08-19 / Codex.

## Outcomes & Retrospective

The exporter fuses delayed measurement destinations across independent quantum
work and measurement writes to distinct static bits. It also exports a
cleanup-forwarded measurement result as its unique destination CBit when the
store dominates every other use and no later write makes that replacement stale.
A different static bit may be written safely; a same-bit or dynamic-index write
remains fail-closed. Direct-result, use-before-store, nested-overwrite,
distinct-bit, cleanup-QASM, and guarded target-compilation regressions cover the
boundary.

The release MLIR binding builds with the final source patch. The four focused
measurement-store tests pass. All 209 active tests in
`test/python/test_mlir_qiskit_translation.py` pass, with one target-stack-only
case guarded on this independent branch. Repository lint also passes. Before the
final bit-precision correction, the combined tree passed all 4,156 native tests,
all 219 Qiskit translation tests, and all 40 Benchpress integration tests. The
exact six different-bit Benchpress failures that motivated that correction are
covered by direct and nested regressions. The new reversed-store case covers the
mapped W-state shape. The diff remains uncommitted for independent audit, and
nothing is pushed.

## Context and Orientation

`bindings/mlir/qiskit/QiskitExport.cpp` recursively collects a validated
`ExportedCircuit` before creating a Qiskit object. `isFusableMeasurementStore`
requires one unique static `cbit.store` destination in the same block. It walks
operations between measurement and store, permits only independent quantum work
and measurement writes to distinct static bits, and verifies that the store
precedes every other result use. `ExportState` records the accepted result's
destination bit and store for expression export and snapshot validation.

`test/python/test_mlir_qiskit_translation.py` contains direct MLIR, QASM
cleanup, and target-compilation regressions.
`docs/mlir/python_compiler_collection.md` states the exact measurement-store and
result-use restrictions.

## Plan of Work

Retain the quantum-only intervening-operation allowlist. Extend it to later
measurements and their provably distinct static stores. Keep same-bit and
dynamic stores fail-closed. Require a unique destination store, prove the store
precedes every other use, and record the accepted destination in `ExportState`.

Export a recorded measurement result as a classical-bit expression. Extend
snapshot discovery to start at its destination store and reuse the top-level and
nested same-bit write checks. A dynamic-index write may target the snapshot bit
and therefore remains fail-closed. Keep source IR, writer construction, import
code, and unsupported-use preflight unchanged.

Add regressions for reversed measurement-store order, same-bit and dynamic
intervening stores, a direct measurement-result condition, a consumer before its
store, and top-level and nested overwrites before later consumers. Add an
always-running OpenQASM 2 cleanup case, distinct direct and nested bit writes,
and a guarded exact target-compiled reproducer. Retain the sparse-mapping,
delayed quantum-work, multiple-destination, and stale-load negatives.

Finally, format, rebuild the release binding, run focused and complete Qiskit
translation tests, run repository lint, and inspect the commit-relative diff.
Prepare the uncommitted diff for independent audit. Do not commit or push.

## Concrete Steps

Run from the repository root:

    clang-format --dry-run --Werror bindings/mlir/qiskit/QiskitExport.cpp
    uvx ruff check test/python/test_mlir_qiskit_translation.py
    uvx rumdl check docs/mlir/python_compiler_collection.md \
      .agent/plans/qiskit-measurement-deferral.md
    git diff --check

Build and test against the worktree extension:

    cmake --build build/release --target mqt-core-mlir-bindings --parallel 8
    pytest test/python/test_mlir_qiskit_translation.py \
      -k 'measurement_result or sparse_target_measurement or measurement_store'
    pytest test/python/test_mlir_qiskit_translation.py
    uvx nox -s lint

## Validation and Acceptance

The delayed-store regression must preserve `measure`, `reset`, `x` order. The
reversed-store regression must preserve measurement order and write distinct
destination bits. Same-bit and dynamic intervening stores must fail without
changing the source. A measurement-result condition after its store must become
a Qiskit condition on the destination CBit. A use before the store and a result
whose destination is overwritten before a consumer must also fail without
changing the source. The exact target-compiled QASM case must pass when the
classical stack is assembled.

The release binding, complete translation file, and lint must pass. The final
diff from the structured-export parent must remain limited to measurement
destination/result preflight, focused regressions, documentation, and this plan.
No reader, writer, generic control-flow, scalar-parameter, or CBit
definite-write behavior may change.

## Idempotence and Recovery

Build, format, lint, and test commands are repeatable. If compiled IR changes,
inspect it before adjusting a regression; do not broaden the predicate without a
new equivalence argument. The work is isolated on a child branch, so its
exporter parent remains recoverable.

## Artifacts and Notes

The expected source boundary is:

    allowed between measure and store =
        arith.constant | qc.measure | qc.reset | qc::UnitaryOpInterface |
        statically disjoint measurement cbit.store

    measurement-result expression =
        unique static destination CBit, store before every other use,
        no intervening or nested write that may target its destination bit

All other operations remain disallowed.

## Interfaces and Dependencies

No public interface or dependency changes are required. The implementation adds
internal result-to-destination maps to `ExportState` and uses existing MLIR
operation classes and QC dialect interfaces.

Revision note: Created when measurement-ordering support was split from the
structured-control exporter. Expanded after post-merge integration exposed
cleanup-forwarded result uses and reversed stores for distinct measurements.
