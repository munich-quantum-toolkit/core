# Preserve classical control during target mapping

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

Mapping a program to a constrained quantum target must preserve the program's
measurement-fed classical control. Before this repair, routing could create a
real cyclic dependency through a later conditional, and the subsequent
topological reorder could move a classical-register comparison before the
measurement result was stored. Cleanup then legally folded the now-stale
comparison and silently removed the conditional quantum work. Routing could also
insert a SWAP between a measurement and its classical destination, which made
native Qiskit export reject an otherwise valid mapped program.

After this change, mapped programs terminate, verify, remain executable on the
target, preserve their measurement-to-store-to-comparison-to-conditional
semantics, and retain the measurement/store adjacency required by native Qiskit
export. Three focused Core regressions demonstrate the independent routing,
memory-order, and measurement-destination invariants. The exact 21 affected
Benchpress mapping cases provide end-to-end evidence.

## Progress

- [x] (2026-09-03 14:40+02:00) Reproduced silent control loss with `cc_n12` on a
      line target: all 25 register comparisons and conditionals disappeared
      after target compilation.
- [x] (2026-09-03 14:48+02:00) Traced the loss to the custom topological
  reorder, which modeled SSA dependencies but not CBit memory effects.
- [x] (2026-09-03 14:47+02:00) Added a focused CBit regression that failed
  against the unmodified sorter because cleanup removed `cbit.cmp` and the
  conditional branch.
- [x] (2026-09-03 15:02+02:00) Extracted the constrained `cc_n12` routing cycle
  and proved that a SWAP consumed a later `qco.if` result while feeding an
  earlier syndrome gate.
- [x] (2026-09-03 15:20+02:00) Ordered recursively reported effects on the same
  concrete SSA resource and made the ready-node selection stable by original
  block position.
- [x] (2026-09-03 15:27+02:00) Added a four-qubit star regression and adjusted
      hot routing so a SWAP endpoint that crossed structured control is rewound
      to the earliest unresolved frontier. Disabling this adjustment makes the
      regression abort at the sorter's acyclicity assertion.
- [x] (2026-09-03 15:42+02:00) Reproduced a separate native-export failure in
  which a routed SWAP split a measurement from its direct memory-write
  destination.
- [x] (2026-09-03 15:54+02:00) Added a third regression and anchored the first
      routed SWAP after the measured bit's first direct write, without moving
      the normal insertion point backward or crossing structured control.
- [x] (2026-09-03 15:58+02:00) Built the focused target; passed all three new
  regressions for 25 repeats and all 97 tests in the mapping test binary.
- [x] (2026-09-03 15:59+02:00) Built a fresh wheel and confirmed that the
  minimized measurement-destination program exports natively with adjacent
  measurement/store operations and one routing SWAP.
- [x] (2026-09-03 15:59+02:00) Ran all 21 requested Benchpress mapping cases.
      Every case compiled with the complete conditional count; all 15 cases at
      or below 64 bits exported and validated, while the six wide cases reached
      only the separate, known Qiskit register-width limit.
- [x] (2026-09-03 16:18+02:00) Passed Clang 22 formatting, the complete
      non-unity C++ lint build, direct Clang 22 analysis of the three changed
      C++ translation units, ExecPlan Markdown lint, and `git diff --check`.
- [x] (2026-09-03 16:30+02:00) Passed `uvx nox -s lint`; all repository-wide
      pre-commit hooks completed successfully.

## Surprises & Discoveries

- Observation: Target compilation initially succeeded while silently dropping
  control. Evidence: constrained `cc_n12` changed from 25 comparisons and 25
  conditionals to none after cleanup.
- Observation: CBit register operations already report standard MLIR memory
  effects in `mlir/include/mlir/Dialect/CBit/IR/CBitOps.td`. No new CBit API is
  required to preserve their observable order.
- Observation: A FIFO Kahn worklist can place older independent ready operations
  between `qco.measure` and `cbit.store`. Choosing the earliest original
  operation among ready nodes keeps the reordered block as close to source order
  as dependencies permit.
- Observation: Conservatively chaining every effectful operation introduces a
  cycle in constrained `cc_n12`. Even the necessary CBit store-to-comparison
  edge exposes an already-existing routed SSA cycle:
  `swap -> swap -> swap -> ctrl -> measure -> store -> cmp`, followed by
  `cmp -> if -> swap -> swap -> swap`. The mapper, rather than the sorter, must
  prevent that backward quantum dependency.
- Observation: Rewinding every SWAP operand to the earliest frontier fixes the
  adaptive programs but invalidates three pure-quantum mapping tests because it
  crosses routing state already reflected in the layout. Retaining the normal
  one-step insertion point unless the wire crossed structured control preserves
  existing routing behavior.
- Observation: Stable scheduling is insufficient when routing itself inserts a
  SWAP immediately after a measured qubit. A four-qubit line program produced
  `measure -> swap -> sink -> store`, and native Qiskit export reported
  `measurement destination must follow the measurement in the same block`.
- Observation: The direct store need not be the measurement's next operation.
  Dynamic tensor placement can leave a `qco.sink`, and the builder can
  materialize an index constant before `cbit.store`. The robust local invariant
  is the first subsequent operation that directly consumes the measured bit and
  reports a memory write, provided no structured control is crossed.
- Observation: Pull request #2240 merged into `main` as `557375a2a` during this
  work. The completed diff was transplanted onto current `main` at `815dcd336`,
  so this repair is a focused follow-up rather than another commit in the former
  pull-request stack.

## Decision Log

- Decision: Fix ordering in the shared QCO topological sorter rather than in
  Benchpress or the CBit canonicalizer. Rationale: the sorter creates the
  invalid observable order, and both top-level and nested mapping paths use it.
  Date/Author: 2026-09-03 / Codex.
- Decision: Preserve original order between recursively reported accesses that
  carry the same concrete effect SSA value. Ignore value-less or unknown effects
  in this repair. Rationale: a CBit register SSA value identifies the
  non-aliasing mutable resource that needs ordering, while broader barriers
  conflict with temporary SSA reorder requirements created by routing.
  Date/Author: 2026-09-03 / Codex.
- Decision: Choose the earliest original block-position operation from the
  sorter's ready set. Rationale: a stable topological order minimizes movement
  and retains source adjacency whenever dependencies allow it. Date/Author:
  2026-09-03 / Codex.
- Decision: Preserve the mapper's original decrement-all, insert, and
  increment-all protocol. On the first use of each SWAP endpoint in a routing
  sequence, scan backward to the earliest unresolved frontier and change the
  insertion iterator only if the scan crosses `qco.if`, `qco.index_switch`,
  `scf.for`, or `scf.while`. Rationale: a routed SWAP must not consume a later
  structured-control result, while ordinary routing must keep its established
  layout and iterator bookkeeping. Date/Author: 2026-09-03 / Codex.
- Decision: If a SWAP input is a measurement result, place the first SWAP after
  the first later direct consumer of the measured bit that reports a memory
  write, but stop searching at structured control and never move earlier than
  the first operand's normal definition anchor. Rationale: this restores the
  native exporter's measurement/destination contract without treating an
  arbitrary delayed write across control as the destination or expanding the
  temporary use-before-definition window. Date/Author: 2026-09-03 / Codex.
- Decision: Keep this follow-up scoped to mapping and ordering. Rationale: the
  151- and 301-bit programs now map with complete control, and their remaining
  Qiskit export error is the independent greater-than-64-bit comparison work
  assigned to the second pull request. Date/Author: 2026-09-03 / Codex.

## Outcomes & Retrospective

The implementation addresses three coupled causes rather than masking their
symptom. Routing no longer creates a backward dependency through later
structured control. The sorter preserves effects on the same concrete mutable
resource and selects ready operations stably. A routed SWAP no longer splits a
measurement from its direct classical destination.

All three focused regressions pass for 25 consecutive repeats, and the complete
mapping binary passes 97 of 97 tests. A wheel built from the final source maps
all 21 requested programs with full conditional counts. The four Shor layouts,
nine constrained `cc_n12`, `cc_n32`, and `cc_n64` layouts, and both teleport
programs export and validate. The six constrained `cc_n151` and `cc_n301`
layouts preserve all 303 or 603 conditionals and stop only at the separate
64-bit Qiskit bridge limit. The Feynman QEC program, used as extra routing-cycle
coverage, also exports and validates with all three conditionals.

The main lesson is that a sorter cannot repair a dependency cycle created by
routing. Observable memory ordering made the latent mapper bug visible, and the
correct repair needed both a narrowly defined routing frontier and a stable,
side-effect-aware reorder.

## Context and Orientation

This change targets current MQT Core `main`, which already contains pull request
2240 and its `reorderTopologically` implementation. A qubit wire in this
repository is a linear SSA def-use chain: each quantum operation consumes a
qubit value and usually produces its continuation. Target mapping in
`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp` assigns those logical
wires to hardware sites and inserts SWAP operations when a two-qubit gate is not
adjacent on the target.

Independent wires can advance to different positions. One wire can pass through
a later classically controlled operation while an earlier two-qubit gate on
another wire remains unresolved. If a routing SWAP combines those positions, its
result can feed the earlier gate while depending on the later control, which is
a real cycle. Hot routing therefore records the earliest unresolved operation
and selectively rewinds only endpoints that crossed a supported
structured-control operation.

Mapping calls `qco::reorderTopologically`, implemented in
`mlir/lib/Dialect/QCO/Utils/Sorting.cpp`, after it has repaired physical
adjacency. The sorter uses Kahn's algorithm: operations with no unresolved
predecessors are placed into a ready set, emitted, and removed from their
successors' predecessor counts. SSA edges include ordinary operands and values
captured by nested regions, but they do not order accesses through a mutable
CBit handle.

A CBit register is an SSA handle to mutable classical storage. `cbit.store`
writes that storage, while `cbit.cmp` reads the full register. There is no SSA
result-to-operand edge from the store to the comparison, so their reported
memory effects supply the required order. If the comparison moves before the
store, canonicalization sees the zero-initialized register as untouched, folds
the condition, and removes the conditional quantum branch.

Native Qiskit translation also requires a measurement's direct classical
destination to follow it in block order. When the measured qubit itself needs an
inserted SWAP, the mapper must anchor that SWAP after the bit's direct
memory-write destination. The search is intentionally stopped at structured
control, and the selected anchor can only delay the mapper's normal insertion
point, never move it earlier.

The focused regressions live in
`mlir/unittests/Dialect/QCO/Transforms/Mapping/test_mapping.cpp`. Their target
links are declared in the adjacent `CMakeLists.txt`. This plan and its
implementation must preserve unrelated changes, modify no other worktree, and
perform no GitHub action. Any eventual pull request needs normal human review
and the AI-use disclosure required by `docs/ai_usage.md`.

## Milestones

### Milestone 1: Preserve observable classical-register order

Add a CBit-backed mapping regression containing a zero-initialized register, a
measurement, a store, a full-register comparison, and a conditional X gate. Run
mapping alone first and inspect measurement/store adjacency, then run the
canonicalizer separately and inspect the surviving store, comparison,
conditional, and controlled gate. Update `reorderTopologically` to add
original-order dependencies for recursively reported effects on the same
concrete SSA resource and to choose the earliest original ready operation. This
milestone is complete when the regression fails without these sorter changes and
passes with the complete semantic chain intact.

### Milestone 2: Prevent backward routing through classical control

Add a four-qubit star-topology regression with three CNOTs, an ancilla
measurement, and a conditional correction. Record the earliest unresolved
frontier before hot SWAP insertion. Preserve the existing one-step iterator
protocol, but rewind the first use of an endpoint when its backward segment
crosses supported structured control. This milestone is complete when mapping
succeeds, the module verifies, the result is target-executable, a SWAP is
present, and the conditional X remains. Disabling only the rewind must reproduce
the sorter acyclicity failure.

### Milestone 3: Keep the measurement destination adjacent

Add a four-qubit line regression in which routing after a mid-circuit
measurement needs a SWAP. Find the measured bit's first direct memory-write
consumer before any structured control and use it only to delay the first SWAP
past that destination. Preserve the original insertion point if it is already
later. This milestone is complete when the mapped store directly follows its
defining measurement, at least one SWAP remains, the module verifies and is
target-executable, and native Qiskit conversion succeeds.

### Milestone 4: Validate the focused repair and exact integrations

Build and repeat the three regressions, run the complete mapping binary, build a
wheel, and exercise the exact affected Benchpress programs. Finish with C++
formatting, C++ lint, repository lint, and a whitespace-error diff check. This
milestone is complete when the 97 mapping tests pass, all 21 cases retain their
expected conditional counts, the 15 at-or-below-64-bit cases export and
validate, and the six wide cases fail only with the documented width-limit
message.

## Plan of Work

In `mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp`, generalize the
existing boundary lookup to return a `WireIterator`. Add a helper that scans
backward from a hot SWAP insertion point and selects the position before the
earliest crossed supported structured-control operation. Record the earliest
active routing frontier before the existing decrement-all step, pass that
boundary into hot SWAP insertion, and adjust each physical endpoint on its first
use. Keep the existing increment-all step so untouched wires and inserted SWAP
chains resume normally.

In the same file, recognize a qubit value defined by `qco.measure`. Scan forward
for the first operation that directly consumes the measurement's bit result and
reports `MemoryEffects::Write`, returning no destination if structured control
is crossed first. When either SWAP input has such a destination, delay insertion
to the latest applicable destination only if it is later than the normal
first-input definition anchor.

In `mlir/lib/Dialect/QCO/Utils/Sorting.cpp`, centralize dependency insertion so
duplicate SSA and effect edges do not inflate predecessor counts. Record each
operation's original block position. For every recursively reported memory
effect with a concrete SSA value, add an original-order edge from the previous
operation that affected the same value. Use a priority queue ordered by the
recorded position for the ready set.

In `mlir/unittests/Dialect/QCO/Transforms/Mapping/test_mapping.cpp`, add the
three semantic regressions described above. Register and link the CBit dialect
through the test and its `CMakeLists.txt`. Assert transformed operations,
ordering, verification, target executability, and SWAP/control survival rather
than comparing a fragile textual snapshot.

## Concrete Steps

Run every command from the repository root. Configure against the project's
supported LLVM/MLIR 23.1 package and build the focused test:

    cmake --preset release \
      -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir \
      -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
    cmake --build build/release \
      --target mqt-core-mlir-unittest-mapping -j 4

Run the three regressions repeatedly, then the full binary:

    build/release/mlir/unittests/Dialect/QCO/Transforms/Mapping/\
      mqt-core-mlir-unittest-mapping \
      --gtest_filter='MappingPassFixture.RouteBeforeLaterClassicalControl:\
MappingPassFixture.PreserveStoredRegisterControlDuringRouting:\
MappingPassFixture.KeepMeasurementDestinationAdjacentDuringRouting' \
--gtest_repeat=25 --gtest_brief=1
build/release/mlir/unittests/Dialect/QCO/Transforms/Mapping/\
      mqt-core-mlir-unittest-mapping --gtest_brief=1

Build a wheel with the same LLVM/MLIR package. Install it into an isolated
Python target directory rather than replacing the developer environment:

    CMAKE_ARGS='-DMLIR_DIR=/path/to/llvm/lib/cmake/mlir \
      -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm' \
      uv build --wheel --out-dir dist
    python -m pip install --no-deps --target wheel-site dist/mqt_core-*.whl

With that isolated package first on `PYTHONPATH`, compile the four-qubit line
program described in Milestone 3, convert it with
`QCProgram.to_qc(copy=True).to_qiskit(target=target)`, and inspect the mapped
IR. The `cbit.store` line must immediately follow `qco.measure`, a `qco.swap`
must remain, and conversion must succeed.

From a Benchpress checkout, run the MQT gym's load, target preparation, compile,
native-Qiskit conversion, and `circuit_validator` path for these
repository-relative inputs: `qasm/qasmbench-small/shor_n5/shor_n5.qasm` on all
four synthetic layouts; `cc_n12`, `cc_n32`, `cc_n64`, `cc_n151`, and `cc_n301`
on square, heavy-hex, and line layouts; and Feynman `teleport.qasm` plus
`teleportv2.qasm` on FakeTorino. Count `qco.if` before export so the wide cases
remain verifiable even though their native export is a separate task.

Finish with the repository checks:

    clang-format -i \
      mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp \
      mlir/lib/Dialect/QCO/Utils/Sorting.cpp \
      mlir/unittests/Dialect/QCO/Transforms/Mapping/test_mapping.cpp
    uvx nox -s cpp-lint
    uvx nox -s lint
    git diff --check

## Validation and Acceptance

`RouteBeforeLaterClassicalControl` must fail without the crossed-control
adjustment because the mapper creates a real cycle. With the repair, it must
map, verify, be target-executable, contain a routing SWAP, and retain its
conditional X.

`PreserveStoredRegisterControlDuringRouting` must fail without effect ordering
because cleanup removes the comparison/control chain. Immediately after mapping,
its store must directly follow its defining measurement. After canonicalization,
the store must precede the comparison, the comparison must feed the conditional,
and the branch X must remain.

`KeepMeasurementDestinationAdjacentDuringRouting` must fail without the
destination anchor by producing a measurement-to-SWAP-to-store sequence. With
the repair, its module must verify, be target-executable, contain a routing
SWAP, and place the direct store immediately after the measurement. The same
minimal program must convert to native Qiskit without the former destination
error.

The complete mapping binary must report 97 passing tests. Shor must retain four
conditionals on each of four layouts. The constrained `cc_n12`, `cc_n32`, and
`cc_n64` cases must retain and export 25, 65, and 129 conditionals respectively
on each of three layouts. Teleport and teleport-v2 must retain and export two
and three conditionals. Every resulting Qiskit circuit must pass Benchpress
target validation. The constrained 151- and 301-qubit cases must compile and
retain 303 and 603 conditionals; their only accepted remaining failure is
`Qiskit register comparisons support at most 64 bits`.

## Idempotence and Recovery

Configuration, compilation, tests, wheel construction, formatting, and lint
commands are repeatable. Use a fresh isolated wheel target when validating a new
source revision so an older extension cannot mask the result. CMake can resume a
partially populated build directory after dependency or network failure. If
compiler-cache temporary files are not writable, set `CCACHE_TEMPDIR` to a
writable temporary directory and rerun the same build.

Do not reset the branch, delete another checkout, or edit another task's
worktree. Use `git status`, file-scoped `git diff`, and `git diff --check` to
inspect only this task's changes. This plan authorizes no push, pull-request
creation, or other GitHub mutation.

## Artifacts and Notes

Focused and full test evidence from the final source:

    [==========] 3 tests from 1 test suite ran.
    [  PASSED  ] 3 tests.
    ... repeated 25 times ...
    [==========] 97 tests from 4 test suites ran.
    [  PASSED  ] 97 tests.

The minimized native-export case reports adjacent operations and a retained
SWAP:

    {'adjacent': True, 'swaps': 1,
     'qiskit_ops': {'cx': 3, 'measure': 1, 'swap': 1}}

Representative exact integration evidence is:

    shor_n5: 4 qco.if / 4 if_else, all 4 layouts validate
    cc_n12: 25 qco.if / 25 if_else, all 3 constrained layouts validate
    cc_n32: 65 qco.if / 65 if_else, all 3 constrained layouts validate
    cc_n64: 129 qco.if / 129 if_else, all 3 constrained layouts validate
    teleport / teleportv2: 2 / 3 conditionals, FakeTorino validates
    cc_n151 / cc_n301: 303 / 603 qco.if on all 3 constrained layouts;
      export reaches only the documented 64-bit bridge limit

The final commit-aware `cpp-lint` session completed its non-unity build and
selected exactly `Mapping.cpp`, `Sorting.cpp`, and `test_mapping.cpp`.
Clang-format and Clang-tidy 22 reported zero failed checks.

## Interfaces and Dependencies

The public mapping and sorting interfaces remain unchanged. The implementation
uses the existing `mlir::qco::WireIterator`, `IRRewriter`, and QCO structured
control operations in `Mapping.cpp`. It uses MLIR's standard
`MemoryEffectOpInterface`, `getEffectsRecursively`, and `MemoryEffects::Write`
rather than adding a CBit-specific dependency API. `Sorting.cpp` uses LLVM's
`PriorityQueue`, `DenseMap`, and small dense sets for stable scheduling and
deduplicated concrete-resource dependencies.

The test target gains only the existing `MLIRCBitDialect` library. No new
third-party dependency, public option, benchmark workaround, or generated file
is introduced.

Revision note (2026-09-03): The final revision adds the independently reproduced
measurement-destination failure and third regression, records the narrow
destination anchor and monotonic insertion decision, updates validation from 96
to 97 tests, identifies this work as a follow-up to #2240 on current `main`, and
records the final build and lint evidence.
