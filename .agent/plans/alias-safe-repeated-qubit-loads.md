# Implement alias-safe repeated qubit loads

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core's QC builder currently rejects repeated loads of a qubit-register
element, and the typed OpenQASM frontend expands every runtime qubit index into
an `scf.index_switch` with one case per register element. After this change,
ordinary `memref.load` operations may name the same register element repeatedly,
including from the entry block, and QC-to-QCO conversion remains correct even
when two sequential runtime indices happen to alias. OpenQASM programs with
runtime qubit indices lower to a fixed-size load/use sequence instead of code
whose size grows with the register width.

The observable proof is twofold. A raw QC module containing repeated qubit loads
converts to verified QCO in which each register operand is extracted immediately
before its quantum operation and inserted immediately afterward. An OpenQASM
program containing `x q[i];` emits a checked `memref.load` and no
width-dependent `scf.index_switch`.

## Progress

- [x] (2026-08-01 11:00Z) Investigated issue #1893 against the typed OpenQASM
  frontend and LLVM/MLIR 22 behavior.
- [x] (2026-08-01 11:00Z) Allocated a clean task worktree at the selected
  `e772dba5ce1c51cc7b8931b8a5031a826040f3d5` revision and read repository
  policy.
- [x] (2026-08-01 12:25Z) Implement Stage 1 builder, QC-to-QCO, and QTensor
  behavior.
- [x] (2026-08-01 12:40Z) Add focused Stage 1 regression tests and verify the
      raw conversion without relying on cleanup. The QC IR (311 tests), QTensor
      IR (29 tests), and QC-to-QCO (141 tests) suites pass.
- [ ] Commit Stage 1 as one signed, AI-attributed commit.
- [ ] Implement Stage 2 direct OpenQASM register loads.
- [ ] Add focused Stage 2 translation and end-to-end tests.
- [ ] Commit Stage 2 as one signed, AI-attributed commit.
- [ ] Integrate the refreshed `origin/main` tip and rerun affected validation.
- [ ] Run repository lint and an independent exact-revision review.

## Surprises & Discoveries

- Observation: MLIR's LLVM 22 CSE cannot be the correctness mechanism. QC
  quantum operations declare memory writes, so an intervening gate prevents CSE
  from reusing a prior `memref.load`.
- Observation: The current OpenQASM frontend no longer emits dynamic loads. It
  selects eagerly loaded references through nested `scf.index_switch`
  operations, making emitted operation count depend on register widths.
- Observation: Existing QTensor canonicalizers scan only constant-index tensor
  chains. Dynamic indices can alias unless they are the exact same SSA value, so
  only adjacent exact-index folds are safe without an alias analysis.
- Observation: The old tensor-chain search can cross a structured QCO operation.
  Once structured regions carry complete tensors, folding an insert against an
  extract on the far side of that operation bypasses region-local quantum
  updates. The replacement canonicalizers therefore use direct SSA
  producer/consumer relationships only.
- Observation: `ValueRange{value}` does not own its initializer-list storage.
  Keeping such a range in a local variable produced a dangling view during the
  first conversion prototype. Single-qubit helper calls now use owned
  `SmallVector<Value, 1>` storage.

## Decision Log

- Decision: Implement two dependent commits, with the core conversion preceding
  the OpenQASM migration. Rationale: The raw conversion becomes independently
  useful and Stage 2 never lands on an unsafe converter. Date/Author:
  2026-08-01, Codex.
- Decision: Treat qubit `memref.load` as reference provenance and materialize
  QCO values around each consuming quantum operation. Rationale: This supports
  arbitrary sequential runtime aliasing without adding an analysis, dialect
  type, or operation. Date/Author: 2026-08-01, Codex.
- Decision: Keep simultaneous operands subject to the existing distinct-qubit
  contract and preserve OpenQASM runtime assertions. Rationale: A multi-qubit
  QCO operation cannot extract the same linear qubit twice. Date/Author:
  2026-08-01, Codex.
- Decision: Preserve the public eager `allocQubitRegister` API and add a
  storage-only primitive for the frontend. Rationale: Existing builders and
  fixtures remain source-compatible. Date/Author: 2026-08-01, Codex.
- Decision: Start from the user-selected revision, then integrate the one newer
  live-base commit before final verification. Rationale: This preserves exact
  implementation scope and satisfies the repository's live-base handoff rule.
  Date/Author: 2026-08-01, Codex.
- Decision: Preserve exact QCO reference comparison where representation is
  unchanged, and assert operation-local extract/insert linearity plus complete
  tensor state for structured-register cases. Rationale: Eagerly extracted QCO
  references encode the representation Stage 1 intentionally removes and are no
  longer valid structural references for those cases. Date/Author: 2026-08-01,
  Codex.

## Outcomes & Retrospective

Implementation is in progress. This section will record final behavior, commit
identifiers, validation results, remaining limitations, and lessons learned.

## Context and Orientation

The QC dialect models `!qc.qubit` as a reference. A QC gate mutates that
reference in place. The QCO dialect models `!qco.qubit` linearly: an operation
consumes an input SSA value and produces the next value. A QTensor is a
one-dimensional `tensor<...x!qco.qubit>` whose `qtensor.extract` removes one
qubit and whose `qtensor.insert` returns it.

`mlir/include/mlir/Dialect/QC/Builder/QCProgramBuilder.h` and
`mlir/lib/Dialect/QC/Builder/QCProgramBuilder.cpp` own the public builder.
`allocQubitRegister` currently allocates a qubit memref and eagerly loads every
constant element. `loadQubit` rejects entry-block and repeated loads using
per-region maps.

`mlir/lib/Conversion/QCToQCO/QCToQCO.cpp` converts QC reference semantics to QCO
value semantics. Its current load pattern extracts a qubit when the load is
converted and keeps it live until a structured boundary or register
deallocation. That is unsafe when another load may name the same element.

`mlir/lib/Dialect/QTensor/IR/Operations/ExtractOp.cpp` and `InsertOp.cpp` own
local tensor-chain canonicalization. They use constant-index equality when
searching through a chain.

`mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` emits QC from the
typed OpenQASM frontend. Non-scalar declarations currently retain vectors of
eagerly loaded values. Dynamic references recursively construct
`scf.index_switch` operations over those vectors.

Tests belong under `mlir/unittests/Conversion/QCToQCO/`,
`mlir/unittests/Dialect/QC/IR/`, `mlir/unittests/Dialect/QTensor/IR/`, and
`mlir/unittests/Dialect/QC/Translation/`. Production tools are not test
locations.

## Plan of Work

First, simplify the builder. Add
`Value QCProgramBuilder::allocQubitRegisterStorage(int64_t size)`, which
validates the size, selects dynamic allocation mode, allocates
`memref<sizex!qc.qubit>`, and registers it for automatic deallocation. Implement
the existing eager allocator in terms of this function. Make `loadQubit` emit an
ordinary `memref.load` at any valid insertion point and remove all region
tracking.

Second, refactor QC-to-QCO state. Before conversion, collect every qubit
`memref.load` into owned register-access records containing the original memref
and index. Reject uses that escape through unsupported calls, returns, stores,
or general block arguments before changing IR. Replace the current persistent
extraction maps with central materialize and commit helpers. Materialization
extracts register-backed operands in operand order from the latest mapped
QTensor; standalone references resolve through the existing qubit map. Commit
updates standalone mappings or reinserts register outputs in reverse order.
Measurement, reset, gates, barriers, and modifiers all use these helpers.

Qubit load conversion then erases only the reference marker. Register
deallocation consumes the already-complete mapped tensor. Structured control
flow carries QTensor values for used registers and QCO values only for
standalone qubits. The existing terminator phase yields those values, but no
insert-before or re-extract-after bookkeeping remains. Transient provenance must
be discarded before later conversion phases; no erased `Operation*` may survive
in state.

Third, add adjacent QTensor folds that use LLVM 22 `isEqualConstantIntOrValue`.
An extract immediately consuming an insert result at the same exact dynamic or
constant index forwards the inserted scalar and original destination. An insert
immediately consuming an extract result at the same index forwards the original
tensor. Existing non-adjacent patterns remain constant-only and must not cross a
potentially aliasing dynamic index.

After Stage 1 tests pass, commit it. Then migrate the OpenQASM emitter.
Represent every non-scalar qubit declaration with storage allocated through the
new builder method. Resolve static register indices with a constant index load.
Evaluate and bounds-check dynamic indices once, cast them to MLIR `index`, and
load directly. Scalar qubits, gate arguments, and hardware qubits retain their
existing representations.

Remove only quantum dynamic-dispatch construction and its projected width
accounting. Measurement, reset, barrier, and modifier emission use the directly
resolved values. Keep classical dynamic indexing and general construction budget
checks. Preserve signed negative-index wrapping, bounds assertions, and pairwise
runtime assertions for possibly identical gate operands.

Finally, add behavioral tests, integrate current `origin/main`, run focused and
broad validation, update this plan, and obtain an independent read-only review.
Do not push, open a pull request, post GitHub text, or modify issue state.

## Concrete Steps

Run all commands from the repository root.

Inspect changes continuously with:

    git status --short
    git diff --check
    git diff --stat

Configure and build the MLIR-enabled release tree with the repository wrapper:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release

During Stage 1, run the focused conversion and dialect binaries discovered in
the generated build tree. At minimum, run the QC-to-QCO, QC IR, and QTensor IR
test suites with GoogleTest filters for the new cases.

During Stage 2, run the OpenQASM QC translation tests and compiler pipeline
tests. Exercise `mqt-cc` on a small OpenQASM input containing a dynamic qubit
index and inspect emitted QC/QCO for successful verification and absence of
quantum width-expanded switches.

After integrating the live base, rerun the affected tests, then:

    ./.agent/run.sh uvx nox -s lint
    git diff --check
    git status --short

## Validation and Acceptance

Stage 1 is accepted when builder tests can emit repeated identical loads in the
entry block and nested SCF, and raw `createQCToQCO()` conversion verifies
without first running canonicalization or CSE. Tests must cover constant and
dynamic indices, repeated sequential use, an eager constant reference followed
by a possibly aliasing dynamic reference, structured control flow, measurement,
reset, barriers, and modifiers. The resulting QCO must contain complete tensors
at structured boundaries and register deallocation.

QTensor canonicalization is accepted when adjacent operations at the same
constant or exact dynamic SSA index fold, while adjacent operations at distinct
dynamic SSA indices remain unchanged.

Stage 2 is accepted when OpenQASM dynamic quantum operations emit direct
register loads and no quantum-selection `scf.index_switch`. A large register
must emit approximately the same number of operations as a small register for
the same source operation. Existing index bounds, negative-index, and
same-qubit-operand failures must remain observable.

The complete change is accepted when focused tests, affected compiler tests, the
release build, repository lint, `git diff --check`, and independent review pass.
Any environment-limited check must be recorded with exact evidence rather than
weakened.

## Idempotence and Recovery

All build and test commands are repeatable and keep caches inside this worktree.
Edits are confined to this task worktree. If a conversion experiment fails,
preserve the failing test and adjust the implementation; do not reset, clean, or
modify another worktree. Integrating `origin/main` is postponed until the two
selected commits exist, so conflicts can be resolved against clear atomic
history.

## Artifacts and Notes

The selected implementation base is:

    e772dba5ce1c51cc7b8931b8a5031a826040f3d5

At worktree allocation time the live base was one unrelated commit newer:

    a4293f1473f4a716aec81707d3cbe01cd1a1b83a
    ✨ Expose DD serialization in Python (#1983)

Issue #1893 is an enhancement/MLIR issue and is not labeled `good first issue`.
No external GitHub mutation is authorized by this plan.

## Interfaces and Dependencies

The only new public C++ interface is:

    Value QCProgramBuilder::allocQubitRegisterStorage(int64_t size);

`QCProgramBuilder::allocQubitRegister(int64_t)` remains source-compatible.
`QCProgramBuilder::loadQubit(Value memref, Value index)` retains its signature
but permits repeated and entry-region loads.

The implementation uses existing MLIR 22 APIs: `memref::LoadOp`,
`qtensor::ExtractOp`, `qtensor::InsertOp`, `isEqualConstantIntOrValue`, dialect
conversion patterns, SCF iter arguments, and the repository's existing LLVM
containers. It adds no external dependency, dialect operation, dialect type,
pass, or command-line option.

Revision note (2026-08-01, Codex): Created the living plan from the approved
two-stage design and recorded the refreshed-base boundary before implementation.
