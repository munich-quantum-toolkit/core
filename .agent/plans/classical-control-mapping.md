# Preserve classical control during target mapping

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

Target mapping must route quantum operations inside a `qco.if` without moving
the condition before the measurement that produced it. After this change, the
QCO cleanup pipeline exposes the accessed elements of a static qubit tensor as
scalar qubits, the mapper treats structured control as a routing boundary, and
the target preflight admits exactly the tensor conditionals that cleanup can
scalarize. The mapper then restores SSA dominance without changing CBit register
access order. Focused unit tests demonstrate tensor scalarization, target
compilation, sparse conditional routing, and measurement-store-load ordering.

## Progress

- [x] (2026-08-19 14:24Z) Audited the original classical-control implementation
      against the first-class CBit register representation and selected the QCO
      and mapping-only file boundary.
- [x] (2026-08-19 14:34Z) Ported the QCO tensor scalarization and mapping
      boundary changes.
- [x] (2026-08-19 14:36Z) Added a CBit-specific classical-memory ordering
      regression over 32 mapping seeds.
- [x] (2026-08-19 14:45Z) Stacked the slice on the target capability parent and
      admitted only `qco.if` tensor state accepted by
      `hasOnlyScalarizableTensorInputs`.
- [x] (2026-08-19 14:49Z) Built the compiler, QCO IR, QCO utility, and mapping
      targets; ran their focused and complete unit-test binaries.
- [x] (2026-08-19 14:49Z) Recorded the validation evidence and prepared the
      single local mapping commit without pushing it.
- [x] (2026-08-19 19:51Z) Rebased the focused mapping commit onto the finalized
      post-#2158 capability layer, rebuilt the four affected C++ test targets,
      and passed all 147 compiler, 493 QCO IR, 115 QCO utility, and 88 mapping
      tests.
- [x] (2026-08-19 19:51Z) Pass the final post-rebase Markdown and full
      repository lint checks and `git diff --check`.
- [x] (2026-08-20 04:22Z) Reproduced the large `bwt_n37-linear` target compile,
      isolated the mapper's final topological sort as the first scalability
      limit, and replaced its stored adjacency graph with an indexed Kahn sort
      that discovers outgoing SSA edges from use lists.
- [x] (2026-08-20 04:22Z) Added direct regressions for duplicate nested SSA
      captures, stable ready-operation preemption, overlapping SSA and
      classical-memory edges, and cycle detection without partial mutation.
- [x] (2026-08-20 04:31Z) Validated the promoted low-memory implementation: the
      seven affected native binaries passed 1,223/1,223 tests, the complete
      CTest suite passed 4,147/4,147 tests with one expected skip, and the
      repository lint session passed.

## Surprises & Discoveries

- Observation: The earlier mapper preserved order only for values with an MLIR
  memref type. CBit registers use `!cbit.reg<N>`, so the same check does not see
  their loads and stores. Evidence: `cbit.load` and `cbit.store` carry a
  `cbit::RegisterType` operand rather than a `BaseMemRefType` operand.
- Observation: The target capability parent deliberately rejected every qubit
  tensor crossing structured control. Evidence: before stacking this slice,
  `hasUnsupportedQubitTensorState` treated every tensor operand or result of
  `qco.if` as unsupported, including pass-through tensors that cleanup removes.
- Observation: Both generic and explicit-adjacency topological sorts scale
  poorly on a very large straight-line mapping result. Evidence: the stock MLIR
  scan remained in `sortTopologically` after more than 32 minutes 27 seconds at
  4.018 GiB resident memory, while the first replacement stored an outgoing
  vector for every operation and was killed by the operating system.
- Observation: The final post-synthesis `RemoveDeadValues` pass is a separate
  scalability limit even after the mapper sort is bounded. Evidence: an isolated
  run reached that pass and was killed. Removing that already-no-op pass
  together with the low-memory sorter produced a successful experimental compile
  in 232.991 seconds with a maximum resident set of 9,185,083,392 bytes. A
  separate 14-case differential found no IR change when the pass was reapplied.
  The pipeline change therefore belongs to its own child task and commit, not
  this mapping slice.

## Decision Log

- Decision: Preserve the order of operations that access either a CBit register
  or a memref. Rationale: CBit is the current frontend representation, while
  retaining memref support preserves direct callers that map lowered IR.
  Date/Author: 2026-08-19, Codex.
- Decision: Stack this slice directly on the target capability parent and relax
  only the `qco.if` tensor preflight through `hasOnlyScalarizableTensorInputs`.
  Rationale: the predicate and the rewrite share the same branch-access
  analysis, so preflight accepts no form that the cleanup pipeline cannot remove
  before mapping. Date/Author: 2026-08-19, Codex.
- Decision: Keep constant `qco.index_switch` canonicalization outside this task.
  Rationale: static-control selection belongs to the target capability and
  preflight task, not to routing. Date/Author: 2026-08-19, Codex.
- Decision: Use a stable indexed Kahn sort and derive outgoing SSA edges from
  value use lists when each producer is scheduled. Rationale: 32-bit operation
  indices bound the per-operation arrays, a generation vector deduplicates
  nested and repeated uses, and no per-operation outgoing vectors retain the
  potentially large edge set. The minimum-index ready queue preserves the old
  stable order. Date/Author: 2026-08-20, Codex.
- Decision: Compute the complete order before moving operations and fail on a
  cycle. Rationale: a failed sort must not leave a partially reordered block,
  and silently breaking a combined SSA/classical-memory cycle would hide an
  invalid scheduling constraint. Date/Author: 2026-08-20, Codex.

## Outcomes & Retrospective

The mapping slice is complete on top of the target capability parent. Target
preflight accepts only scalarizable `qco.if` tensor inputs, cleanup converts
them to scalar qubits, and mapping preserves CBit and memref access order while
routing structured control. Unsupported dynamic shapes and repeated tensor
accesses retain their preflight diagnostics.

The focused compiler, QCO scalarization, driver, and 32-seed CBit ordering
regressions pass. The seven affected C++ binaries pass 1,223/1,223 tests: 147
compiler, 493 QCO IR, 233 decomposition, 91 mapping, 23 native synthesis, 121
optimization, and 115 QCO utility tests. The complete CTest suite passes all
4,147 registered tests with one expected skip. `git diff --check` and
`uvx nox -s lint` pass. The target-compilation liveness change and OpenQASM
export remain in independent child slices.

The post-#2158 rebase applied without a source conflict. The rebuilt affected
binaries preserved the same passing counts, so the merged CBit implementation
does not require a mapping adaptation.

The large-circuit investigation replaced the mapper's unbounded adjacency
storage with a linear-per-operation representation. The direct sorter tests and
the existing 32-seed CBit regression protect exact edge deduplication, stable
ordering, SSA dominance, classical-memory order, and failure without mutation.
The independent final dead-value cleanup issue is intentionally left to a
separate child commit so this mapping slice remains reviewable.

## Context and Orientation

QCO represents a qubit as a linear SSA value. A `qtensor` value groups several
linear qubits into a tensor. The mapping pass in
`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp` assigns these logical
qubits to target sites and inserts swaps when a two-qubit operation is not
adjacent. Structured operations such as `qco.if` contain regions, so the mapper
must route their nested operations and reconcile the layout at every region
boundary.

`mlir/lib/Dialect/QCO/IR/SCF/IfOp.cpp` owns `qco.if` canonicalization. The new
canonicalization recognizes a statically shaped rank-one qubit tensor whose
branches use complete constant-index extract and insert chains. It extracts the
accessed elements before the conditional, threads those elements as scalar
qubits, and inserts them after the conditional. Unsupported shapes remain
unchanged.

`mlir/lib/Compiler/TargetCompilation.cpp` verifies target capabilities before
cleanup and mapping. It must continue rejecting tensor state in generic
structured control and in unsupported `qco.if` access patterns, while allowing
the exact static tensor forms that the new canonicalization scalarizes.

CBit registers are non-aliasing classical storage values declared by the CBit
dialect. A measurement result reaches a later condition through a `cbit.store`
and `cbit.load`. Those operations share a register operand but do not have an
SSA edge between each other. The mapper's final topological reorder must add an
explicit order edge between consecutive classical-storage accesses.

Kahn's algorithm orders a directed graph by counting each operation's incoming
dependencies and repeatedly scheduling an operation whose count is zero. The
implementation in `Mapping.cpp` stores each top-level operation and its 32-bit
index, one 32-bit incoming count, one 32-bit deduplication generation, and the
result order. It does not store outgoing SSA edges. When an operation is
scheduled, MLIR's value use lists identify its top-level consumers. A separate
successor map represents the linear chain of classical-memory accesses.

## Plan of Work

Add `hasOnlyScalarizableTensorInputs` to
`mlir/include/mlir/Dialect/QCO/QCOUtils.h` and implement it beside the tensor
scalarization pattern in `mlir/lib/Dialect/QCO/IR/SCF/IfOp.cpp`. Register the
pattern with the existing `qco.if` canonicalization. Add positive and negative
tests to `mlir/unittests/Dialect/QCO/IR/test_qco_ir.cpp`.

Use the same predicate in `mlir/lib/Compiler/TargetCompilation.cpp` to permit
only scalarizable `qco.if` tensor state through target preflight. Keep the
existing conservative rejection for every other structured-control operation.
Add a compiler-pipeline regression that compiles a runtime conditional with an
untouched static tensor and confirms cleanup removes the tensor before mapping.

Update `walkProgramGraph` in `mlir/include/mlir/Dialect/QCO/Utils/Drivers.h` so
a unary structured operation is released as an explicit graph boundary. Update
its existing layer-order test in
`mlir/unittests/Dialect/QCO/Utils/test_drivers.cpp`.

Update `Mapping.cpp` so it keeps sparse structured operations when no routing
workspace is required, selects the qubit value that crosses each control-flow
boundary, and restores SSA order with a stable indexed Kahn sort. The sort must
preserve accesses to both `cbit::RegisterType` and `BaseMemRefType`, derive
outgoing SSA edges from use lists, deduplicate them with a generation vector,
and leave the block unchanged if the combined graph contains a cycle. Add the
direct CBit dialect link to the QCO transform library.

Add routing regressions to
`mlir/unittests/Dialect/QCO/Transforms/Mapping/test_mapping.cpp`. The CBit
regression must build a measurement, store, load, and conditional in one block,
run mapping on a sparse target, and verify that the store remains before the
load and conditional. Direct sorter regressions must cover duplicate and nested
SSA uses, stable priority after a newly ready earlier operation, an edge that is
both SSA and classical-memory ordering, and a cycle that returns failure without
moving any operation. Add direct test dependencies only where the test uses the
CBit API.

## Concrete Steps

Run these commands from the repository root:

    cmake --preset release
    cmake --build --preset release --parallel 8 --target \
      mqt-core-mlir-unittest-qco-ir \
      mqt-core-mlir-unittest-qco-utils \
      mqt-core-mlir-unittest-mapping \
      mqt-core-mlir-unittests-compiler
    ./build/release/mlir/unittests/Dialect/QCO/IR/mqt-core-mlir-unittest-qco-ir
    ./build/release/mlir/unittests/Dialect/QCO/Utils/mqt-core-mlir-unittest-qco-utils
    ./build/release/mlir/unittests/Dialect/QCO/Transforms/Mapping/mqt-core-mlir-unittest-mapping
    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler
    ctest --test-dir build/release --output-on-failure
    git diff --check
    uvx nox -s lint

If the utility binary has a different generated target name, inspect
`mlir/unittests/Dialect/QCO/Utils/CMakeLists.txt` and run the listed binary
directly. Record the exact command in this plan.

## Validation and Acceptance

The QCO IR tests must show that supported tensor conditionals contain scalar
qubit operands after canonicalization and that dynamic or incomplete tensor
access remains unchanged. Driver tests must show unary `qco.if` and
`qco.index_switch` operations in their own graph layers. Mapping tests must show
that sparse conditionals remain sparse when no swap is required, acquire routing
workspace when swaps are required, and make progress when a tensor carrier is
inactive.

The CBit regression is the semantic acceptance test. After mapping, the
measurement must remain before its `cbit.store`, the store must remain before
the corresponding `cbit.load`, and the load must remain before the `qco.if`
whose condition uses it. All focused binaries, `git diff --check`, and the
repository lint session must pass.

The direct sorter regressions must show that multiple nested uses create one
dependency, a newly ready operation with an earlier original index preempts a
later ready operation, a classical edge already represented by SSA is released
once, and a cycle returns false without changing block order. The complete CTest
suite must report no failures.

The target-compilation regression must accept an untouched static tensor around
a runtime `qco.if`, remove all tensor types before mapping, and retain both the
conditional and statically placed qubits. Dynamic tensors and unsupported
extract/insert chains must retain their stable preflight diagnostics.

## Idempotence and Recovery

The build and test commands are repeatable. The implementation changes only the
files named in this plan. If a focused test fails, rerun that binary with a
GoogleTest filter before changing code. Do not reset or modify another worktree.
No command in this plan changes a remote repository.

## Artifacts and Notes

The original QCO, driver, and mapping production files match the first-class
CBit base except for this task's changes. This permits a narrow port. The CBit
storage predicate is the only semantic adaptation required in the mapper.

The `bwt_n37-linear` investigation produced these checkpoints:

    stock MLIR topological scan: >32:27 elapsed, 4.018 GiB resident, unfinished
    explicit outgoing-adjacency sort: operating-system kill
    low-memory sort plus final RemoveDeadValues: reached RemoveDeadValues, killed
    low-memory sort without redundant final cleanup: 232.991 s elapsed,
      9,185,083,392 bytes maximum resident, success
    explicit RemoveDeadValues differential after target compile: 14/14 no-ops

Only the low-memory sort belongs to this plan. The final cleanup deletion and
its 14-case fixed-point evidence are maintained by a separate child ExecPlan.

Final validation from the promoted mapping worktree:

    affected native binaries: 1,223/1,223 passed
    ctest --test-dir build/release --output-on-failure:
      4,147/4,147 passed, one expected skip
    uvx nox -s lint: passed
    git diff --check: passed

## Interfaces and Dependencies

`mlir::qco::hasOnlyScalarizableTensorInputs(IfOp)` is a public QCO utility that
returns true only when every tensor input can be converted to scalar threading
by the `qco.if` canonicalizer. The target capability verifier calls this
function before cleanup so that its accepted language matches the rewrite.

`MLIRQCOTransforms` must link `MLIRCBitDialect` privately because `Mapping.cpp`
directly tests `cbit::RegisterType`. The mapping unit test must also link that
dialect when it constructs CBit operations directly.

Plan revision note: Created the focused implementation plan after separating
target capabilities, OpenQASM export, and Qiskit export into independent tasks.
Revised it after the target capability parent became available and this slice
could include the narrow scalarization-aware preflight relaxation. Recorded the
clean post-#2158 rebase and its complete affected-binary validation before
stacking the OpenQASM follow-up. Revised it again after the large BWT benchmark
exposed the topological sort's resource scaling; promoted the audited
use-list-based Kahn implementation and kept the independent target-pipeline
cleanup optimization in a separate child task. The promoted implementation
passed the full native CTest and lint sessions before the mapping commit was
amended.
