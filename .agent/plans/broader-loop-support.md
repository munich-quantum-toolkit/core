# Broader loop support and actionable translation errors

Status: complete. Integrated loop construction into the existing builder, added
continue support, simplified diagnostics, and reviewed conversion and test
scope. Replaced the dependency source patch with an upstream fix and validated
the implementation rebased on main.

## Goal and scope

Support break and continue targeting the innermost existing for/while loop,
including nested conditional exits; general SCF while regions; and supported
scalar loop state. Canonical do-while round trips use one `scf.while` without an
exit-only `scf.if`. Keep termination checks, runtime input APIs, and new backend
arithmetic or runtime gate-parameter capabilities out of scope.

## Decisions

- Keep SCF as the shared representation. Build temporary QC CFGs for early
  exits, pass complete scalar state through a common decision block, and reuse
  upstream CFG-to-SCF lifting. Existing ordinary structured lowering stays.
- Loop construction belongs to `QCProgramBuilder`. All jump edges share one
  decision block with complete state. A for-loop counter update is selected
  there only on continuation, avoiding an extra latch and undefined state.
  Continue paths contribute to initialization at eventual for-loop exits.
- Emit general loops as `while (true)` with before statements, a conditional
  break, and after statements. Preserve parallel edge assignments, snapshots,
  exit values, and different before/after tuples.
- Use native local Qiskit variables and nested captures, never extra classical
  bits for compiler state. Keep Python objects in the versioned adapter.
- Preserve definite initialization across reachable exits and account for the
  guaranteed first iteration of do-while loops.
- Keep diagnostic improvements scoped to loops and encountered blockers; retain
  details in Python exceptions. Export remains buffered and non-mutating.
- Use jeff's existing two-region loops, with fixes only for demonstrated
  state/capture conversion defects.
- Use LLVM scoped state restoration and MLIR block inlining. Share range setup,
  condition recognition, and exit-generation merging across their existing
  callers. Retain direct integer selection to avoid unnecessary control flow.

## Validation

`uv run --no-sync pytest -q test/python/test_mlir_loops.py` exercises canonical
do-while shape, zero/one/multiple iterations, both quantum regions, snapshots,
swaps, unequal tuples, i8 overflow, f64, nested exits, range/list
break/continue, initialization, native captures, determinism, and contextual
exceptions. The C++ OpenQASM translation suite also checks buffered output on
failure.

All 31 focused loop tests, 501 MLIR Python tests, and 3,117 MQT MLIR C++ tests
pass. The loop interchange module skips under Qiskit 1.1.0, matching the
existing translation-version contract. Lint, C++ lint, and stub generation pass;
generated stubs have no public API diff. Routine build and check entry points
are in
[AGENTS.md](../../AGENTS.md).

## Implementation findings

- OpenQASM export uses the existing cleanup stages except RemoveDeadValues,
  which introduced poison in an unreachable after region of valid first-exit
  do-while loops. Normal program cleanup retains its previous pipeline.
- jeff's native two-region representation handles these loops. Reverse
  conversion needed independent tuple mapping, scalar conditional results,
  invariant captures, and bounded recursive region rewrites. Forward conversion
  shares conditional lowering between QCO and scalar-only SCF conditionals.
- The jeff dependency pins the decoder correction in
  [upstream PR #52](https://github.com/unitaryfoundation/jeff-mlir/pull/52).
  Region inputs and outputs have independent types; no source patch is applied.
- Equivalent pure index expressions reuse the existing QCO-to-QC qubit cache
  entry, avoiding redundant stores after jeff, including folded constants.
  Equivalence requires the same result number; distinct results of one operation
  remain distinct indices, preserving real qubit permutations.
- Qiskit local variables have fresh native identities per export. Determinism
  checks compare serialized structure, while captures within each circuit tree
  retain one identity.
- CBit initialization follows branch intersections and loop execution: the
  before region executes on every exit path; the after region can execute zero
  times. Existing snapshot validity checks remain in place.
- Scalar declarations use defined placeholders while remaining uninitialized in
  frontend analysis. This keeps unobservable initial loop state representable
  without accepting reads before assignment or rewriting arbitrary poison.
- Direct measurement SSA results are saved after their validated destination
  store, so later overwrites do not change the saved value or add classical
  bits.
- Symbolic gate expressions stay on the existing parameter path when a program
  also uses native scalar loop variables; runtime gate angles remain rejected.
- Wide register comparisons retain direct register expressions, while scalar
  locals remain limited to 64 bits. Snapshot materialization may read at its own
  operation position; deferred reads still reject intervening writes.
- QCO-to-jeff applies conversion patterns before folds, so reset folding cannot
  inspect a source quantum operand after region cloning maps it to a target
  type.
- All Qiskit conditionals use the existing variable-aware branch construction.
  This also lets invalid-capture exceptions reach Python without unwinding
  through the exception-disabled QC builder callback.
