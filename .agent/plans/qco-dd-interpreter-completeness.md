# Complete concrete classical and control-flow interpretation in QCO DD utilities

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

QCO DD execution currently interprets common scalar operations, local classical
memrefs, and basic branches. Real lowered programs can also pass memrefs through
functions, use `cf.switch`, contain multi-block `scf.execute_region`, and emit
additional arithmetic min/max and math operations. This work makes those
concrete programs executable without requiring prior canonicalization.

## Progress

- [x] (2026-08-11 09:30Z) Audited the current interpreter and tests.
- [x] (2026-08-11 09:45Z) Preserved memref identity across function arguments
  and results with shared backing storage and alias-wide deallocation.
- [x] (2026-08-11 09:49Z) Interpreted matching and default `cf.switch`
  successors and multi-block `scf.execute_region` through one concrete CFG
  walker.
- [x] (2026-08-11 10:24Z) Added signed/unsigned integer min/max, floating-point
      min/max variants, and the common unary math and power operations emitted
      by current MLIR flows.
- [x] (2026-08-11 10:27Z) Replaced the old `arith.maxsi` rejection with a
  semantic test that gates a qubit on every newly supported result.
- [x] (2026-08-11 10:30Z) Ran all 144 QCO utility tests successfully.
- [x] (2026-08-11 12:05Z) Passed the final 148-test QCO utility suite and the
      full repository lint session after all DD slices were integrated.
- [x] (2026-08-11 13:15Z) Re-audited the complete branch, removed the obsolete
      QCO-only region argument/yield binders in favor of `bindValuePairs`, and
      made recursive-call tracking a required interpreter invariant.

## Surprises & Discoveries

- Observation: The existing unsupported-operation test intentionally uses
  `arith.maxsi`; it must become a positive semantic test once max operations are
  supported. Evidence:
  `mlir/unittests/Dialect/QCO/Utils/test_dd_functionality.cpp`.
- Observation: Deallocation must invalidate every SSA alias, not only the value
  passed to `memref.dealloc`. Evidence: the new alias use-after-free test fails
  through the original caller value after the returned alias is deallocated.
- Observation: MLIR textual SSA names are scoped to the entire region, including
  block arguments. Evidence: the first multi-block tests were rejected for
  reusing `%arg`; unique block-argument names fixed the test input.
- Observation: Adding Math dialect operation classes requires both loading the
  dialect in parser-based tests and linking `MLIRMathDialect` from the utility.
  Evidence: the focused classical-operation test now constructs and executes
  `math.absf`, transcendental operations, and `math.powf`.

## Decision Log

- Decision: Model memref passing by aliasing the existing backing storage rather
  than copying it. Rationale: MLIR memref arguments have reference semantics, so
  callee stores must be visible to the caller. Date/Author: 2026-08-11, Codex.
- Decision: Reuse one generic CFG-region walker for function bodies and
  multi-block `scf.execute_region`. Rationale: both carry SSA block arguments
  through concrete branch terminators and should share transition limits and
  diagnostics. Date/Author: 2026-08-11, Codex.
- Decision: Use `bindValuePairs` for QCO modifier-region arguments and split
  yield segments as well as standard CFG/call values. Rationale: the general
  binder now covers every linear and classical value category, making the two
  older specialized binders redundant. Date/Author: 2026-08-11, Codex.

## Outcomes & Retrospective

The interpreter now carries memref aliases across calls, traverses concrete CFG
switches and multi-block execute regions, and evaluates the selected common
arith/math operations. The final 148-test QCO utility suite and repository lint
session pass. A final branch audit removed obsolete incremental helpers and an
unreachable nullable recursion-state path without changing behavior.

## Context and Orientation

`mlir/lib/Dialect/QCO/Utils/DDFunctionality.cpp` contains a small concrete
interpreter. `ClassicalEnv` maps SSA values to scalars and memref storage.
`bindValuePairs` carries values across regions and calls. `walkFunctionBody`
interprets `cf.br` and `cf.cond_br`. `applyScfRegion` currently accepts exactly
one block. The corresponding tests live in
`mlir/unittests/Dialect/QCO/Utils/test_dd_functionality.cpp`.

## Plan of Work

Replace memref storage values with shared backing objects or an equivalent alias
map so `bindFrom` can bind a destination memref to the same storage. Extend
function-call result binding consistently and test stores in a callee that are
loaded by its caller.

Generalize concrete CFG traversal to accept `cf.switch` and to execute a region
until its region terminator. Use that machinery for multi-block
`scf.execute_region`; retain the 10000-transition guard. Preserve existing
single-block behavior for `scf.if`, `scf.index_switch`, loops, and QCO regions.

Extend `applyClassicalOp` using MLIR operation classes for signed and unsigned
integer min/max, floating min/max variants, and common unary math operations
that can be evaluated with `<cmath>`. Include only operations present in the
configured LLVM/MLIR version and emitted by supported frontend/lowering flows.
Reject invalid domains or non-finite behavior only where MLIR semantics require
it; otherwise mirror the host operation.

## Concrete Steps

From the repository root, make focused patches and build the utility test
target:

    ./.agent/run.sh cmake --build --preset release --target mqt-core-mlir-unittest-qco-utils

Run:

    ./build/release/mlir/unittests/Dialect/QCO/Utils/mqt-core-mlir-unittest-qco-utils

Then run:

    ./.agent/run.sh uvx nox -s lint

## Validation and Acceptance

A callee receiving a memref must mutate the caller's register. A `cf.switch`
must select its matching or default successor and carry block arguments. A
multi-block `scf.execute_region` must return the selected yielded values. Each
new arithmetic operation must affect a quantum branch or parameter so tests
prove semantic execution rather than mere acceptance. Existing 140 tests must
remain green and new failure diagnostics must be covered.

## Idempotence and Recovery

All commands are repeatable. Keep the storage refactor local to the interpreter;
if a shared-storage approach causes accidental lifetime coupling, revert only
that focused patch and use stable storage IDs instead. No external state is
modified.

## Artifacts and Notes

Before this work, `ClassicalEnv::bindFrom` supports only scalar destinations,
`walkFunctionBody` recognizes only branch and conditional branch, and
`applyScfRegion` rejects regions without exactly one block.

## Interfaces and Dependencies

Use LLVM containers, MLIR `cf`, `arith`, and `math` operation classes already
available to the MLIR component. If adding the Math dialect include, add the
corresponding CMake target dependency. Do not add a third-party evaluator.

Revision note: Initial plan created for memref aliasing, concrete CFG coverage,
and common classical operations.

Revision note (2026-08-11): Recorded completion of all implementation slices and
their focused semantic tests; only aggregate validation remains.

Revision note (2026-08-11): Recorded successful aggregate native and lint
validation.

Revision note (2026-08-11): Recorded the post-integration deletion audit and
consolidation onto the general value binder.
