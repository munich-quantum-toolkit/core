# Broader loop support and actionable translation errors

Status: complete. Implementation, simplification review, and local validation
passed.

## Goal and scope

Support breaks targeting the innermost existing for/while loop, including nested
conditional exits; general SCF while regions; and supported scalar loop state.
Canonical do-while round trips use one `scf.while` without an exit-only
`scf.if`. Keep termination checks, continue, runtime input APIs, and new backend
arithmetic or runtime gate-parameter capabilities out of scope.

## Decisions

- Keep SCF as the shared representation. Build temporary QC CFGs for early
  exits, pass complete scalar state through a common decision block, and reuse
  upstream CFG-to-SCF lifting. Existing ordinary structured lowering stays.
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
swaps, unequal tuples, i8 overflow, f64, nested exits, range/list breaks,
initialization, native captures, determinism, and contextual exceptions. The C++
OpenQASM translation suite also checks buffered output on failure.

All 23 focused loop tests, all 493 MLIR Python tests, and all 3,113 MLIR C++
tests pass. `uvx nox -s lint`, `uvx nox -s cpp-lint`, and `uvx nox -s stubs`
pass; stub generation produces no public API diff. C++ lint also uses explicit
changed-file selection for uncommitted edits. Routine build and check entry
points are in [AGENTS.md](../../AGENTS.md).

## Implementation findings

- OpenQASM export uses the existing cleanup stages except RemoveDeadValues,
  which introduced poison in an unreachable after region of valid first-exit
  do-while loops. Normal program cleanup retains its previous pipeline.
- jeff's native two-region representation handles these loops. Reverse
  conversion needed independent tuple mapping, scalar conditional results,
  invariant captures, and bounded recursive region rewrites. Forward conversion
  shares conditional lowering between QCO and scalar-only SCF conditionals.
- The pinned jeff byte decoder assumed matching input/output tuples for while
  and switch. A narrow CMake dependency patch decodes result types
  independently; remove it when updating to an upstream version with this
  correction.
- Equivalent pure index expressions must reuse the existing QCO-to-QC qubit
  cache entry, avoiding redundant stores of register-backed qubits after jeff.
- Qiskit local variables have fresh native identities per export. Determinism
  checks compare serialized structure, while captures within each circuit tree
  retain one identity.
- CBit initialization follows branch intersections and loop execution: the
  before region executes on every exit path; the after region can execute zero
  times. Existing snapshot validity checks remain in place.
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
