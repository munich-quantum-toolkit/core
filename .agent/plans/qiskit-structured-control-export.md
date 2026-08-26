# Export structured QC control flow to Qiskit

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

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

## Progress

- [x] (2026-08-26 09:00Z) Reconstruct the PR tree on current `main` and resolve
  the changelog conflict from the current unreleased entry.
- [x] (2026-08-26 09:15Z) Retain recursive structured export and the forwarded
  measurement-result fix.
- [x] (2026-08-26 09:30Z) Align Qiskit logical import with OpenQASM by emitting
  canonical short-circuit `scf.if` operations.
- [x] (2026-08-26 09:45Z) Remove general Boolean selection, expression cloning,
  and multi-result sibling budgets from export.
- [x] (2026-08-26 10:00Z) Replace selection tests with canonical round-trip and
  rejection tests.
- [x] (2026-08-26 11:00Z) Build the Python extension and run focused and
  complete local validation.
- [x] (2026-08-26 12:30Z) Record validation evidence and prepare the signed
  commit series for guarded publication.

## Surprises & Discoveries

- Observation: OpenQASM import already emits the required result-bearing
  `scf.if` shapes. Evidence: `emitCondition` in
  `mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` emits lazy AND and
  OR regions.
- Observation: The old Qiskit importer mapped logical and bitwise operations to
  the same eager `arith.andi` and `arith.ori` operations. This evaluated both
  logical operands and disagreed with Qiskit and OpenQASM semantics.
- Observation: MLIR's result-bearing `scf.if` is the native structured form for
  conditional evaluation. `arith.select` selects already-computed SSA values and
  cannot provide short-circuit evaluation.

## Decision Log

- Decision: Keep the recursive normalized circuit model and version-specific
  Qiskit writer. Rationale: Qiskit 2.5 exposes public Python constructors for
  control-flow objects but no equivalent stable C API. The generic translator
  remains independent of Python objects. Date/Author: 2026-08-26 / Codex.
- Decision: Use result-bearing `scf.if` as the shared logical short-circuit
  representation for Qiskit and OpenQASM import. Rationale: It preserves lazy
  evaluation and matches current MLIR structured-control practice. Date/Author:
  2026-08-26 / Codex.
- Decision: Export only canonical single-result Boolean AND and OR shapes.
  Rationale: General ternary and multi-result reconstruction are outside issue
  #2071 and added cloning, normalization, and independent budget machinery.
  Date/Author: 2026-08-26 / Codex.
- Decision: Keep forwarded measurement-result recognition. Rationale: QC cleanup
  can replace a classical load with the measurement SSA result; both values
  denote the same validated destination Qiskit `Clbit`. Date/Author: 2026-08-26
  / Codex.

## Outcomes & Retrospective

Implementation and local validation are complete. The current design keeps the
structured export feature while removing speculative Boolean-selection support.
Compared with the previous PR head, the final diff contains 375 fewer net lines.
The release build, all 4,038 configured CTests, all 219 Qiskit translation
tests, stub generation, MLIR documentation, complete Sphinx documentation, and
the full lint session pass. One CTest is skipped by its existing test policy.

## Context and Orientation

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

## Plan of Work

Keep recursive collection for result-free `scf.if`, constant-range `scf.for`
without loop-carried state, expression-based `scf.while` without carried state,
and result-free `scf.index_switch`. Preserve capture validation, affine loop
parameter projection, packed-register expressions, snapshot checks, definite
CBit initialization, expression limits, and the preflight-before-writer rule.

In Qiskit import, emit the left logical operand before a result-bearing
`scf.if`. For AND, emit the right operand and yield it only in the then region;
yield false in else. For OR, yield true in then and emit the right operand only
in else. Require Boolean operands. Keep bitwise AND and OR as eager arithmetic.

In Qiskit export, require exactly one `i1` result. Recognize AND when the else
yield is false and use the then yield as the right operand. Recognize OR when
the then yield is true and use the else yield as the right operand. Export the
condition and selected right-hand expression recursively, require both regions
to contain only those expression operations or constants, and reject every other
result-bearing shape.

## Concrete Steps

Run commands from the repository root. Configure and build with:

    cmake --preset release
    cmake --build --preset release

Run the focused translation tests first:

    uv run --no-sync pytest test/python/test_mlir_qiskit_translation.py

Then run binding, documentation, and repository checks:

    uvx nox -s stubs
    cmake --build --preset release --target mlir-doc
    uvx nox --non-interactive -s docs
    uvx nox -s lint
    git diff --check

## Validation and Acceptance

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

## Idempotence and Recovery

Builds and tests write only generated files under ignored build or cache
directories and are safe to repeat. If configuration becomes stale, remove only
the named build preset directory after confirming it contains generated output,
then configure again. Never discard unrelated tracked changes.

Before rewriting a published branch, record its exact remote SHA and create a
backup ref. Push with `--force-with-lease=<branch>:<recorded-sha>` so concurrent
remote updates stop the push. Do not rewrite the child PR #2178 in this task.

## Artifacts and Notes

The final history contains four signed commits: implementation, focused tests,
the forwarded-measurement fix, and documentation. Verify each commit with
`git verify-commit` before publication.

Local validation evidence from 2026-08-26:

    cmake --build --preset release
    # passed
    ctest --preset release
    # 100% tests passed, 0 failed out of 4038; 1 skipped
    uv run --no-sync pytest -q test/python/test_mlir_qiskit_translation.py
    # 219 passed
    uvx nox -s stubs
    # passed; no generated tracked changes
    cmake --build --preset release --target mlir-doc
    # passed
    uvx nox --non-interactive -s docs
    # passed
    uvx nox -s lint
    # passed

## Interfaces and Dependencies

This change adds no public C++ or Python API. It keeps the internal
`CircuitWriter::addControlFlow` interface and the existing Qiskit 2.5 adapter.
It uses MLIR SCF, Arith, CBit, and QC dialect operations already required by the
translation code. It adds no dependency.

Revision note: On 2026-08-26, this plan was reduced to the final supported
contract. It now records canonical short-circuit `scf.if` behavior and removes
the abandoned general Boolean-selection design.
