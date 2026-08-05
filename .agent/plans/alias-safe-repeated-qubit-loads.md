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
- [x] (2026-08-01 12:48Z) Commit Stage 1 as the signed, AI-attributed
      `🐛 Make repeated qubit loads alias-safe` commit.
- [x] (2026-08-01 14:05Z) Implement Stage 2 direct OpenQASM register loads.
- [x] (2026-08-01 14:10Z) Add focused Stage 2 translation and end-to-end tests.
      The OpenQASM target (71 tests), QC IR (312 tests), and selected OpenQASM
      compiler pipeline (14 tests) suites pass. `mqt-cc` also emits verified QC,
      operation-local QCO extract/use/insert sequences, and Adaptive QIR for a
      combined dynamic gate/measurement/reset/barrier program.
- [x] (2026-08-01 14:15Z) Commit Stage 2 as the signed, AI-attributed
      `✨ Emit direct OpenQASM qubit register loads` commit.
- [x] (2026-08-01 14:20Z) Rebase the two commits onto refreshed `origin/main`,
      initially at `a5757fe95af7d5f7ba85471c27d8d294afc0b894`.
- [x] (2026-08-01 15:10Z) Run the release build, all 4,401 CTest tests, focused
      MLIR suites, compiler and `mqt-cc` smoke tests, and repository lint.
- [x] (2026-08-01) Address independent-review finding MF-01 by applying the same
      runtime distinctness checks to dynamic barrier operands, rejecting exact
      duplicate barrier references during semantic analysis, and adding focused
      frontend/emitter regressions.
- [x] (2026-08-01) Rebuild all affected release targets and rerun the complete
      post-review CTest matrix: all 4,402 tests pass, with the same two
      configured QDMI skips.
- [x] (2026-08-01) Rebase both signed commits again after `origin/main` advanced
      to `618c37fa4d5d15be282117d2c12f26a3b6e3dd75`, then complete a clean
      434-step release rebuild and pass all 4,404 exact-base CTest cases with
      the same two configured QDMI skips.
- [x] (2026-08-01) Complete a fresh independent review of exact source revision
      `7a84fdf22de840efa423164cb967d5da10ecebf7`; no actionable findings remain.
- [x] (2026-08-02) Address the subsequent exhaustive-review findings: replace
      custom recursive SCF capture discovery with MLIR `RegionUtils`, remove
      dead modifier register state and void casts, diagnose provably duplicate
      simultaneous register operands, and distinguish intact tensor/storage
      allocation from eager element extraction in the builder documentation.
- [x] (2026-08-02) Replace the translation test's custom mutating IR normalizer
      with the production QC-to-QCO conversion and strengthen structured
      conversion checks to prove operation-local extract/use/insert linearity
      and complete QTensor state.
- [x] (2026-08-02) Make OpenQASM alias accounting and assertion emission sparse
      by register, replace ordered barrier deduplication with LLVM dense
      containers, and reject unavoidable whole-register/dynamic barrier aliases
      during semantic analysis.
- [x] (2026-08-02) Validate the remediation with LLVM 22.1.8 clang-tidy, every
      repository hook, a complete 434-step release build, all 4,409 configured
      CTest cases, all 419 Python tests (416 passed, three configured skips),
      and QC/QCO/QIR `mqt-cc` smoke tests on a 99,999-qubit dynamic program.
- [x] (2026-08-02) Integrate current `origin/main`
      `ae7aff283c0c06da5e01e53349a3eaec8b322c1e` with a signed merge, then rerun
      all repository hooks and the full Python 3.13 test session.
- [x] (2026-08-02) Integrate the subsequently advanced `origin/main`
      `1f9fd7ebfa085bd57e7224caac8ab3a73df36981` with signed merge `d8cb2b315`,
      including the compiler-wide global-phase normalization.
- [x] (2026-08-02) Address independent-review finding MF-07 by rejecting
      implicit standalone and register-backed qubit captures in every QC
      modifier verifier and in the verifier-independent QC-to-QCO preflight. Add
      direct/nested coverage for all three modifiers and positive classical
      capture coverage.
- [x] (2026-08-02) Address the final quality findings by adding the Unreleased
      changelog entry, adopting the live `moduleOp` naming policy, and
      documenting the current review and validation state.
- [x] (2026-08-02) Complete the exact-base release build and all 4,467 CTest
      cases, every repository hook, LLVM 22.1.8 changed-line clang-tidy, the
      affected MLIR suites, Python 3.10 (417 passed, three configured skips),
      and compact QC/QCO/QIR `mqt-cc` smokes. The remaining Python-version
      repetitions were intentionally omitted at the user's request after the
      broad native validation was green.
- [x] (2026-08-02) Address final independent-review robustness findings by
      validating quantum SSA sources before mutation. Ranked and unranked qubit
      memrefs, non-one-dimensional or derived register storage, unsupported
      quantum block arguments/results, and quantum captures through unsupported
      region-bearing operations now receive diagnostics instead of reaching an
      assertion or invalid dominance state.
- [x] (2026-08-02) Pass all 165 QC-to-QCO tests, including the six
      verifier-disabled `PreflightRejects*` regressions, and rerun changed-line
      clang-tidy and formatting for the final remediation.
- [x] (2026-08-02) Obtain a fresh independent read-only review of exact source
      revision `252010af9d58bea0371aede7a86880b9d74eba49`; no actionable
      correctness, conversion-legality, C++20/LLVM 22 idiom, performance, or
      test-coverage findings remain.
- [x] (2026-08-05) Rebase the 12 signed feature and remediation commits linearly
      from boundary `1f9fd7ebfa085bd57e7224caac8ab3a73df36981` onto refreshed
      `origin/main` `ecd32f734212b6f622927626a215f2ea31335759`; discard the
      three obsolete merge-main commits and preserve the concurrent
      compiler-target, QDMI, OpenQASM, Jeff, mapping, and Python changes.
- [x] (2026-08-05) Resolve the rebase semantically by retaining upstream
      `compileForTarget`, moving the existing QCO cleanup pipeline to the start
      of target compilation while retaining post-mapping cleanup, preserving
      both sets of `Utils.h` dependencies, and merging every changelog entry.
- [x] (2026-08-05) Validate the rebased series with `git range-diff`, signed
      commit verification, 1,390 focused MLIR tests, four target compiler
      checks, 29 Python MLIR tests, direct-load QC/QCO/QIR `mqt-cc` smokes,
      changed-file hooks, LLVM 22.1.8 changed-line clang-tidy, and
      `git diff --check`.
- [x] (2026-08-05) Address independent-review finding Q-01 by replacing the
      skipped nested `scf.for`/`scf.while` reference comparison with an
      independently built complete-QTensor reference. Extend the existing
      QTensor iterator and IR-equivalence grouping to model `scf.condition` and
      both `scf.while` regions; all 165 QC-to-QCO and three QTensor iterator
      tests pass.
- [x] (2026-08-05) Repair the replacement CI debug-build failure by preserving
      the established partially extracted `nestedForLoopWhileOp` fixture for
      QCO-to-QC tests and adding the complete-QTensor fixture separately. The
      focused nested-loop checks and the complete 137-test QCO-to-QC and
      165-test QC-to-QCO suites pass.
- [x] (2026-08-05) Rebase the 14 signed commits again after `origin/main`
      advanced during replacement CI to
      `077b73f802c408b036453fce9d597e52db70e8c7`. Preserve the newly merged
      PennyLane/QDMI implementation and place #1987 above #2005 in the
      merge-time-ordered changelog.

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
- Observation: Once quantum dispatch is removed, the existing emission budget
  correctly accepts registers that were previously rejected solely because their
  widths were multiplied into projected operation counts. Large and small
  registers now produce identical operation counts for the same dynamic source
  access.
- Observation: The QCO mapping pass intentionally expects a canonical
  all-extracts-before-all-inserts tensor shape. The public place-and-route API
  now composes the existing QCO cleanup pipeline before mapping, which restores
  that supported shape for statically addressable programs without weakening the
  operation-local conversion invariant.
- Observation: Several direct-QCO test fixtures encoded the old partially
  extracted structured state. Rewriting those fixtures to allocate complete
  QTensors directly exposed and verified the new region-boundary invariant while
  preserving direct QC-to-QIR behavior.
- Observation: The aggregate lint cache contained incomplete hook environments
  from an interrupted provisioning run. Moving only the generated cache aside
  and allowing `prek` to recreate it produced a clean all-files lint run.
- Observation: Gates already rejected exact duplicate qubits during semantic
  analysis and asserted potentially aliasing dynamic operands at runtime, but
  barriers originally did neither. Reusing the runtime assertion path for
  barriers preserves the linear QCO contract. Grouping accesses by register and
  using LLVM dense containers avoids quadratic scans over expanded whole
  registers when all indices are statically distinct.
- Observation: MLIR already provides `getUsedValuesDefinedAbove` for region
  capture discovery. Using it for each supported SCF operation avoids
  recursively interpreting block locality and correctly distinguishes values
  captured from enclosing regions from qubits allocated inside a structured
  region.
- Observation: The QC modifier verifiers already reject register loads inside
  modifier bodies. Register-specific modifier maps were therefore dead state;
  modifier operands need only the existing standalone SSA alias mapping.
- Observation: The legacy OpenQASM translation reference test repaired eager
  register fixtures with a custom, order-sensitive IR mutation. Lowering both
  modules through the production QC-to-QCO pass yields a stronger semantic
  comparison and deletes that bespoke normalizer.
- Observation: `QCOProgramBuilder::qtensorAlloc(1)` and
  `QCOProgramBuilder::allocQubitRegister(1)` have intentionally different
  linear-state results. The first returns one intact tensor, while the second
  eagerly extracts its element and returns the residual tensor plus a standalone
  qubit. The corresponding QC APIs now document the same storage-only versus
  eager-reference distinction.
- Observation: QC modifier verification previously allowed a qubit from an
  enclosing region to be used without appearing among the modifier operands.
  QC-to-QCO maps only the aliased modifier block arguments, so such a capture
  reached an assertion when pass verification was disabled. MLIR
  `getUsedValuesDefinedAbove` provides the exact capture query needed by both
  the verifier and conversion preflight while leaving classical captures valid.
- Observation: Collecting load provenance must not assume that the source memref
  is rank one. Rank-zero `memref<!qc.qubit>` is valid MLIR, and calling
  `front()` on its empty index range asserted. The preflight now validates
  storage shape and value origin before recording provenance.
- Observation: `BaseMemRefType`, rather than `MemRefType`, is required when
  classifying unsupported quantum storage. Otherwise unranked
  `memref<*x!qc.qubit>` block arguments and operations can remain dynamically
  legal and escape conversion unchanged.
- Observation: Only `scf.for`, `scf.while`, `scf.if`, and `scf.index_switch`
  participate in QC-to-QCO's explicit quantum-state threading. Capturing quantum
  values through another region-bearing operation, such as `scf.execute_region`,
  must be diagnosed before rewriting; purely classical uses of those operations
  remain legal.
- Observation: QIR Base intentionally requires single-block straight-line QC
  input and does not lower SCF. Dynamic loop programs therefore target the
  Adaptive profile; the Base-profile dynamic-loop failure is not caused by this
  change, and the QC-to-QIR implementations remain untouched.
- Observation: The nested `scf.for`/`scf.while` conversion test originally
  skipped reference equivalence because the QTensor iterator did not recognize
  `scf.condition` as a linear-chain terminator and IR equivalence did not assign
  the loop's before- and after-region tensor values to the allocation's
  equivalence group. Modeling those standard SCF edges removes the test
  exception and lets the existing permutation-aware oracle compare the complete
  structured state.
- Observation: QCO-to-QC intentionally exercises the older partially extracted
  QTensor fixture, whereas Q-01 needs a complete-QTensor QC-to-QCO reference.
  These are distinct conversion inputs rather than interchangeable names, so
  both test builders must remain available.

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
- Decision: Treat barriers as simultaneous multi-qubit operations for alias
  validation. Rationale: A QCO barrier consumes its operands linearly just like
  a multi-qubit gate; exact duplicates should fail in semantic analysis, while
  potentially equal runtime indices need `cf.assert`. Date/Author: 2026-08-01,
  Codex.
- Decision: Use MLIR `RegionUtils` to discover structured captures and keep the
  conversion's custom logic limited to classifying captured quantum values as
  standalone qubits or register provenance. Rationale: Region ownership and
  nested block locality are standard MLIR concerns and should not be
  reimplemented recursively. Date/Author: 2026-08-02, Codex.
- Decision: Reject only simultaneous register operands whose equality is
  statically proven by identical SSA values or equal integer constants.
  Rationale: QCO linearity makes those operations invalid, while distinct
  dynamic values may alias only at runtime and remain the source frontend's
  responsibility to guard. Date/Author: 2026-08-02, Codex.
- Decision: Keep `qtensorAlloc`/storage-only allocation distinct from
  `allocQubitRegister` eager extraction, but make the distinction explicit in
  public documentation and use storage-only APIs whenever complete tensor state
  is required. Rationale: Collapsing the APIs would either hide linear
  extraction or break source compatibility. Date/Author: 2026-08-02, Codex.
- Decision: Reject implicit qubit captures in QC modifiers at both operation
  verification and conversion preflight. Rationale: Modifier bodies may capture
  classical parameters, but every quantum value must enter through an aliased
  modifier block argument so the QCO linear mapping is complete. Date/Author:
  2026-08-02, Codex.
- Decision: Centralize the supported QC quantum-value-source contract in a
  read-only preflight using MLIR `BaseMemRefType` and
  `getUsedValuesDefinedAbove`. Rationale: The lowering state can map only
  `qc.alloc`, `qc.static`, direct rank-one register loads, QC modifier qubit
  arguments, and captures through the four explicitly converted SCF operations;
  diagnosing every other source before dialect conversion prevents assertion and
  dominance failures without adding alias analysis. Date/Author: 2026-08-02,
  Codex.
- Decision: After the complete release build, CTest matrix, hooks, and one
  Python session passed, limit the final iterations to the affected conversion
  binary, changed-line clang-tidy, and compiler smokes. Rationale: The user
  explicitly requested shorter iterations once the broad validation was working.
  Date/Author: 2026-08-02, Codex.
- Decision: Keep both pre- and post-mapping QCO cleanup in target compilation.
  Rationale: The existing cleanup canonicalizes operation-local QTensor access
  into the mapper's supported form without adding a bespoke transformation,
  while the later cleanup preserves established post-mapping normalization.
  Date/Author: 2026-08-05, Codex.
- Decision: Close Q-01 through the existing permutation-aware equivalence
  infrastructure rather than adding a one-off comparison path. Rationale:
  `scf.condition` and the two `scf.while` regions are standard parts of the
  QTensor def-use graph, and supporting them centrally strengthens every
  equivalence user while keeping the production conversion unchanged.
  Date/Author: 2026-08-05, Codex.

## Outcomes & Retrospective

The core implementation retains the intended two-stage shape. Stage 1 makes
`memref.load` a stable QC register-access marker, lowers every quantum use to an
operation-local QTensor extract/use/reverse-insert sequence, carries complete
QTensor state through structured control flow, rejects unsupported escaping
references before mutation, and restricts QTensor rewrites to local, alias-safe
cases. Stage 2 adds storage-only register allocation and makes the typed
OpenQASM emitter produce checked point-of-use loads without quantum
`scf.index_switch` expansion.

Independent review first found a barrier-specific alias gap in Stage 2. The
initial remediation centralized runtime distinctness assertions across gates and
barriers and rejected exact duplicate barrier operands. Two later scrutiny
rounds then identified custom SCF capture discovery, redundant modifier state,
provably invalid simultaneous register operands, shallow structured test
oracles, a custom translation-test IR rewriter, ambiguous eager builder usage,
and quadratic barrier bookkeeping.

The completed remediation uses MLIR `RegionUtils`, removes the dead state and
all new void casts, rejects statically proven simultaneous aliases before
mutation, compares translation fixtures after the production QC-to-QCO pass, and
verifies that every register access is an operation-local extract/use/insert
triple. Structured tests require complete tensor operands, region arguments, and
results. OpenQASM emission counts and emits only pairs that may alias; a
10,000-element static barrier emits no runtime alias checks and lowers in linear
time.

The final scrutiny rounds also closed the entire preflight contract. Modifier
verifiers and conversion preflight reject implicit quantum captures while
allowing classical captures. A centralized source validator uses
`BaseMemRefType` and MLIR region-capture discovery to accept only the quantum
SSA sources and region operations the lowering maps. Rank-zero, unranked,
derived, block-argument, and unsupported-region inputs now fail with diagnostics
before any delayed conversion state is created.

After integrating the current base, the release build completed and all 4,467
configured CTest cases passed, with the two existing QDMI job-ID skips. Every
repository hook passed. Python 3.10 passed 417 tests with three expected Qiskit
compatibility skips; the user requested that the remaining interpreter
repetitions be skipped. LLVM 22.1.8 changed-line clang-tidy reported no
diagnostics, the final QC-to-QCO suite passed all 165 tests, and compact
`mqt-cc` smokes emitted direct-load QC, operation-local QCO, Adaptive QIR for
dynamic SCF, and Base QIR for straight-line input.

No dialect operation, type, pass, command-line option, or external dependency
was added. Direct QC-to-QIR conversion remains unchanged. The current base is
`077b73f802c408b036453fce9d597e52db70e8c7`. The rebased series retained 12
signed feature/remediation commits; Q-01 adds one separately reviewable test
oracle remediation commit and its CI correction adds one signed follow-up.

## Context and Orientation

The QC dialect models `!qc.qubit` as a reference. A QC gate mutates that
reference in place. The QCO dialect models `!qco.qubit` linearly: an operation
consumes an input SSA value and produces the next value. A QTensor is a
one-dimensional `tensor<...x!qco.qubit>` whose `qtensor.extract` removes one
qubit and whose `qtensor.insert` returns it.

`mlir/include/mlir/Dialect/QC/Builder/QCProgramBuilder.h` and
`mlir/lib/Dialect/QC/Builder/QCProgramBuilder.cpp` own the public builder.
Before this work, `allocQubitRegister` allocated a qubit memref and eagerly
loaded every constant element, while `loadQubit` rejected entry-block and
repeated loads using per-region maps.

`mlir/lib/Conversion/QCToQCO/QCToQCO.cpp` converts QC reference semantics to QCO
value semantics. Its previous load pattern extracted a qubit when the load was
converted and kept it live until a structured boundary or register deallocation.
That was unsafe when another load could name the same element.

`mlir/lib/Dialect/QTensor/IR/Operations/ExtractOp.cpp` and `InsertOp.cpp` own
local tensor-chain canonicalization. The former implementations searched
constant-index chains; the replacements fold only direct producer/consumer pairs
whose indices are equal by MLIR's standard value-or-constant utility.

`mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` emits QC from the
typed OpenQASM frontend. Before Stage 2, non-scalar declarations retained
vectors of eagerly loaded values and dynamic references recursively constructed
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

The final validated base also includes:

    a4293f1473f4a716aec81707d3cbe01cd1a1b83a
    ✨ Expose DD serialization in Python (#1983)

    a5757fe95af7d5f7ba85471c27d8d294afc0b894
    🚀 Improve ZX MCX decomposition complexity (#1984)

    618c37fa4d5d15be282117d2c12f26a3b6e3dd75
    ♻️ Reuse MLIR modifier body helpers (#1985)

    907e30c1dc11d47ef0c2fe14659e8c7ac2c297ae
    ⚡ Speed up documentation builds (#1988)

    ae7aff283c0c06da5e01e53349a3eaec8b322c1e
    🔧 Maintenance round (#1989)

    1f9fd7ebfa085bd57e7224caac8ab3a73df36981
    ✨ Normalize compiler-wide global phases (#1986)

    ecd32f734212b6f622927626a215f2ea31335759
    Refreshed main used for the linear PR #1987 rebase

    077b73f802c408b036453fce9d597e52db70e8c7
    ✨ Add PennyLane support for QDMI devices (#2005)

Issue #1893 is an enhancement/MLIR issue and is not labeled `good first issue`.
The user authorized updating PR #1987 with an explicit force-with-lease after
final exact-head validation.

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

Revision note (2026-08-01, Codex): Recorded the completed two-stage
implementation, live-base integration, full validation, fixture migrations, and
lint-cache recovery before independent review.

Revision note (2026-08-01, Codex): Refreshed the final base to include the
subsequent ZX functionality commit and recorded the exact-head 4,401-test
validation.

Revision note (2026-08-01, Codex): Recorded the independent-review barrier
finding, its semantic and runtime remediation, and focused validation.

Revision note (2026-08-01, Codex): Integrated the final modifier-helper base
commit and recorded the exact-base release rebuild and 4,404-test result.

Revision note (2026-08-01, Codex): Closed the plan after a fresh independent
exact-revision review reported no actionable findings.

Revision note (2026-08-02, Codex): Reopened and updated the plan for the two
subsequent exhaustive remediation rounds, current-base integration, and final
validation. A fresh exact-revision review remains pending.

Revision note (2026-08-02, Codex): Closed the plan after integrating `1f9fd7eb`,
fixing implicit modifier captures and the complete quantum-value preflight
contract, recording the shortened final validation requested by the user, and
receiving a clean independent review of exact source revision
`252010af9d58bea0371aede7a86880b9d74eba49`.

Revision note (2026-08-05, Codex): Reopened the plan for the signed linear
rebase onto `ecd32f734`, recorded the semantic conflict resolutions and focused
validation, and closed independent-review finding Q-01 with a complete-QTensor
`scf.while` reference oracle.

Revision note (2026-08-05, Codex): Rebased again onto `077b73f80` after main
advanced during replacement CI, preserved the PennyLane/QDMI additions, and
recorded the QCO-to-QC fixture compatibility correction.
