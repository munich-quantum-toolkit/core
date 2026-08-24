# Export structured Qiskit control flow with CBit state

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core can import Qiskit 2.5 structured control flow, but it currently rejects
the same operations during export. After this change, `QCProgram.to_qiskit()`
can preserve supported nested `scf.if`, `scf.for`, `scf.while`, and
`scf.index_switch` operations. Conditions and switch targets can read captured
first-class CBit registers and use supported Boolean, unsigned-integer, and
floating-point expression trees. A user can observe the result by exporting an
MLIR program, inspecting the Qiskit control-flow operations, and importing the
result again.

The Qiskit 2.5 C API cannot construct control-flow operations or classical
expressions. The generic exporter therefore validates and normalizes the whole
circuit before it allocates a Qiskit writer. The version-specific writer emits
ordinary operations through the C API and emits a zero-operand barrier as a
temporary placeholder for each control-flow operation. After conversion to
Python, it finalizes nested blocks and replaces each placeholder with the
already validated public Qiskit control-flow operation.

This plan covers only structured-control export. Relaxing measurement-result
store adjacency across quantum-only operations is an independently reviewable
follow-up with its own ExecPlan and branch.

## Progress

- [x] (2026-08-19 15:07Z) Read the repository instructions, inspect the CBit,
      scalar-parameter, and expression-capture base, and compare it with the
      earlier combined control-flow implementation.
- [x] (2026-08-19 15:15Z) Add the version-neutral writer interface and the
  Qiskit 2.5 deferred Python control-flow writer without changing the import
  reader or scalar parameter identity model.
- [x] (2026-08-19 15:28Z) Replace flat export collection with recursive
      preflight and emission that uses CBit loads, stores, register/index
      access, snapshots, and definite writes.
- [x] (2026-08-19 15:43Z) Add focused CBit structured-control exporter tests and
      update the public support documentation.
- [x] (2026-08-19 15:52Z) Build the binding, run all 190 Qiskit translation
      tests and repository lint, and review the semantic diff. Creating the
      signed local commit is the final handoff step.
- [x] (2026-08-19 16:04Z) Close the final audit gaps for repeated-bit Uint
      expressions and non-CBit function results, add five focused cases, and
      rerun all 195 translation tests before restacking.
- [x] (2026-08-19 16:22Z) Restack onto the finalized captured-expression import
      parent, rebuild the exact structured branch, and pass all 196 translation
      tests.
- [x] (2026-08-19 19:55Z) Restack again after #2158 merged, rebuild the release
      bindings, pass all 196 translation tests, and pass the complete repository
      lint session and focused diff checks.
- [x] (2026-08-19 20:13Z) Restack onto the audited scalar/capture foundation,
      preserve named-input reachability validation recursively through nested
      structured blocks, rebuild the binding, and pass all 197 translation
      tests.
- [x] (2026-08-21 15:55Z) Restack the export-only commit onto the updated #2175
      head, port it to the closed name-keyed `Parameter` API and current MQT
      metadata, fix the two include-cleanliness findings, rebuild the binding,
      and pass all 204 translation tests.
- [x] (2026-08-21 16:08Z) Preserve low-bit semantics when exporting integer
      truncation, accept direct constant-index switch selectors, add three
      focused round-trip regressions, and pass all 207 translation tests.
- [x] (2026-08-21 16:20Z) Complete the final scope and semantic reviews, pass
      stub generation and repository lint, explicitly defer the complete
      documentation build for this handoff, and prepare the two focused
      implementation and documentation commits.
- [x] (2026-08-21 16:28Z) Restack both commits onto #2175's final
      classical-expression node-bound fix, confirm the export patches remain
      equivalent, rebuild the binding, and pass all 208 translation tests.
- [x] (2026-08-22 06:46Z) Merge the updated `main` after #2175 landed as a
      squash commit, retain its finalized importer and minimized tests, remove
      the duplicated pre-squash parent coverage, rebuild the binding, and pass
      repository lint.
- [x] (2026-08-22 06:59Z) Bound speculative packed-register matching, replace
      recursive classical-snapshot discovery with a bounded worklist, and omit
      loop parameter metadata when the projected value reaches no emitted
      parameter expression; pass the three focused regressions.
- [x] (2026-08-22 07:01Z) Rebuild the release binding, pass all 211 Qiskit
      translation tests against that exact build, and pass focused format and
      static checks; repository lint reformatted the plan and is ready for its
      final clean rerun.
- [x] (2026-08-22 07:04Z) Pass the final clean repository lint run and an
      independent review with no remaining actionable findings; the audit fix is
      ready to commit and push.
- [x] (2026-08-22 07:50Z) Apply the final complexity pass by sharing exporter
      parameter state and native Qiskit symbols, replacing deferred insertion
      bookkeeping with in-place placeholders, reducing repeated round trips and
      source-preservation checks, and shortening internal user-documentation
      detail. Rebuild the binding, pass all 208 minimized translation tests on
      Qiskit 2.5.0, 2.5.1, and 2.5.2, regenerate unchanged stubs, and pass the
      complete repository lint session.

## Surprises & Discoveries

- Observation: The current base already has three independent foundations that
  the old combined implementation did not preserve: first-class CBit registers,
  closed scalar `Parameter` trees keyed by unique names, and
  Python-authoritative import of captured classical expressions. Evidence: the
  base contains `cbit.load`/`cbit.store` export discovery, the variant-backed
  `Parameter` API, and `NativeControlFlowReader::rootClbitIndex`.

- Observation: Qiskit 2.5 exposes structured-control inspection in its C API but
  no matching constructors. Evidence: the current writer can append gates,
  measurements, resets, barriers, and unitaries natively, while the previous
  implementation had to finalize Python block circuits and construct `IfElseOp`,
  `ForLoopOp`, `WhileLoopOp`, and `SwitchCaseOp` through public Python classes.

- Observation: The compiler may move qubit-register `memref.load` operations
  into nested SCF regions. Evidence: the first compiled structured-control probe
  failed qubit resolution until resource discovery walked all loads in the
  function rather than only the entry block.

- Observation: CBit initialization makes the old synthetic false-store logic
  both unnecessary and incorrect for this branch. Evidence: zero-initialized
  allocations round-trip without stores, while undefined returned registers are
  accepted only after validated top-level measurement writes.

- Observation: A syntactic packed-Uint tree can place the same resolved CBit at
  multiple output positions. Evidence: treating `(c[0] | (c[0] << 1))` as a
  register creates invalid repeated-register metadata, while the general
  expression tree represents it exactly.

- Observation: Core represents a circuit without classical outputs with one
  constant-zero `i64` exit-code result. Evidence: `QCProgramBuilder::finalize()`
  and Qiskit import use this sentinel, while every other non-CBit result carries
  semantics that Qiskit circuit export cannot preserve.

- Observation: MLIR truncation to `i1` selects the low bit, while Qiskit's
  Uint-to-Bool cast tests whether the complete integer is nonzero. Evidence: an
  imported Qiskit index expression lowers to `arith.shrui` plus `arith.trunci`;
  exporting that truncation as a cast reverses the result for values such as
  binary `010`.

- Observation: Merging a stacked branch after its parent landed as a squash can
  retain both the old and finalized parent tests without a textual conflict.
  Evidence: the first merged Python diff contained 1,123 changed lines instead
  of the export commit's 803; rebuilding it from `main` plus the export-only
  patch restored the expected delta and kept the finalized #2175 cases.

- Observation: The packed-register recognizer and snapshot validator ran before
  the bounded classical-expression exporter. Evidence: a shared zero-valued
  `arith.ori` DAG caused exponential speculative matching, while a long SSA
  chain entered recursive snapshot discovery before the documented 4,096-node
  and 64-level checks.

- Observation: A loop projection can have SSA uses without contributing a
  parameter to the emitted Qiskit body. Evidence: a projected value used only by
  a dead `math.sin` expression caused finalization to report that the loop
  parameter was absent from its body.

- Observation: Parent and child native writers previously created separate
  Qiskit parameter objects and repaired their identity after Python conversion.
  Evidence: all generated parameter names are unique, so one symbol table shared
  by the writer tree creates the required identity directly and removes the
  `assign_parameters` pass.

- Observation: Every exported control-flow block uses all root qubits and
  classical bits in identity order. Evidence: each child writer is created with
  the root bit counts and is rebased by constructing a circuit from the parent's
  exact bit lists and composing once.

- Observation: Re-importing most exported test circuits repeated #2175's import
  coverage without checking additional exporter behavior. Evidence: direct
  assertions already inspect conditions, case labels, captures, parameter
  identities, and block contents. One broad round trip and focused semantic
  round trips retain the end-to-end contract.

## Decision Log

- Decision: Change only `CircuitWriter`'s output interface and leave all reader
  interfaces untouched. Rationale: the import capture slice is already reviewed
  and does not need exporter construction code. Date/Author: 2026-08-19 / Codex.

- Decision: Represent the collected output as a recursive `ExportedCircuit`
  whose instructions may own one `ExportedControlFlow`. Rationale: validation,
  supported-gate checks, and writer emission must recurse through every block
  before any top-level Qiskit circuit is exposed. Date/Author: 2026-08-19 /
  Codex.

- Decision: Keep the closed scalar parameter tree unchanged. Give each live
  `scf.for` induction parameter a collision-free generated name through
  `Parameter::symbol(...)`, shared by `Loop::parameter` and its lexical body.
  Rationale: Qiskit's `ForLoopOp` must use the same Python `Parameter` object
  that appears in body gates, while the current scalar model intentionally uses
  unique names instead of a second identity field. Date/Author: 2026-08-21 /
  Codex.

- Decision: Treat a returned undefined CBit register as initialized only by
  validated, unconditional measurement stores in the entry block. Reject a load
  before such a write and reject writes that occur only in conditional or loop
  blocks as initialization. Rationale: Qiskit classical bits start at zero, but
  an MLIR register with undefined initialization has no value until every
  observed bit is definitely written. Date/Author: 2026-08-19 / Codex.

- Decision: Build packed-register expressions as one
  `ExpressionKind::ClassicalRegister` leaf. Rationale: this preserves register
  bit order and lets the Qiskit adapter reuse an actual registered
  `ClassicalRegister` when possible instead of reconstructing each shift and
  bitwise-or operation. Date/Author: 2026-08-19 / Codex.

- Decision: If two packed output positions resolve to the same CBit, reject the
  packed-register match and use the general classical expression tree.
  Rationale: a Qiskit `ClassicalRegister` cannot contain the same bit twice, but
  repeated expression leaves are valid. Date/Author: 2026-08-19 / Codex.

- Decision: Accept a non-CBit return only when it is the sole constant-zero
  `i64` no-output sentinel. Reject floating, nonzero, computed, multiple, or
  mixed non-CBit results. Rationale: this preserves Core's established circuit
  convention without silently discarding observable SSA results. Date/Author:
  2026-08-19 / Codex.

- Decision: Keep delayed measurement stores strict and leave their quantum-only
  relaxation to a separate branch and ExecPlan. Rationale: structured-control
  construction and measurement-order equivalence have independent correctness
  arguments and should be reviewed separately. Date/Author: 2026-08-19 / Codex.

- Decision: Export an integer truncation to `i1` as a Qiskit bit-index
  expression, recovering the original shifted index when present and otherwise
  indexing bit zero. Lift a direct constant `index` switch selector to a 64-bit
  Uint expression. Rationale: these mappings preserve MLIR semantics and cover
  valid structured selectors without treating truncation as truthiness.
  Date/Author: 2026-08-21 / Codex.

- Decision: Treat the squash-merged `main` tree as authoritative for #2175 and
  replay only the two structured-export commits while resolving the merge.
  Rationale: this preserves the reviewed importer refactors and streamlined
  parent coverage without changing #2176's scope. Date/Author: 2026-08-22 /
  Codex.

- Decision: Give speculative packed-register matching the same depth and node
  budgets as expression export, use an iterative bounded snapshot walk, and add
  loop metadata only when the generated symbol appears in an emitted body
  parameter. Rationale: preflight must have predictable cost and must not expose
  a Qiskit loop parameter that its body does not contain. Date/Author:
  2026-08-22 / Codex.

- Decision: Append a zero-operand native barrier for each deferred control-flow
  operation and replace that exact instruction after converting the circuit to
  Python. Rationale: an in-place placeholder preserves instruction order without
  merging insertion offsets with controlled-unitary replacements. A focused
  zero-qubit, CBit-only regression verifies that a zero-operand placeholder can
  become a control-flow instruction with classical operands. Date/Author:
  2026-08-22 / Codex.

- Decision: Share one native parameter-symbol table among the root writer and
  all child writers, and share the exporter's SSA-to-parameter state across
  lexical block collection. Rationale: MLIR values remain unique across nested
  regions and generated loop names avoid collisions, so copied scopes and
  post-construction Python parameter replacement add no semantic protection.
  Date/Author: 2026-08-22 / Codex.

- Decision: Make the all-root-bit invariant explicit and remove per-operation
  identity capture maps. Rationale: the exporter never produced sparse or
  permuted captures, and retaining vectors implied unsupported generality while
  duplicating allocation and validation work. Date/Author: 2026-08-22 / Codex.

## Outcomes & Retrospective

Structured Qiskit control flow now exports recursively through a normalized,
frontend-neutral plan and a Qiskit 2.5 deferred Python writer. Captured CBits,
packed registers, Boolean/Uint/Float expressions, nested blocks, static loops,
switches, and loop parameter identity round-trip. Preflight rejects stale
snapshots, unsupported expression/result forms, invalid labels, and undefined
CBit reads or returns before allocating the Qiskit writer.

After the final complexity pass, the release MLIR binding builds successfully
and all 208 tests in `test/python/test_mlir_qiskit_translation.py` pass against
the exact worktree-built extension with Qiskit 2.5.0, 2.5.1, and 2.5.2. Stub
generation produces no tracked changes, and the complete repository lint session
passes. The documentation build remains explicitly deferred for this handoff.
The semantic diff leaves the refreshed import reader and current name-keyed
scalar parameter normalizer unchanged, while recursively checking that every
named scalar input remains reachable from the emitted top-level or nested Qiskit
parameter trees. Speculative expression recognition and snapshot validation are
bounded before recursion can consume unbounded resources, and dead loop
parameter expressions no longer expose invalid Qiskit metadata. The
measurement-store relaxation remains out of scope for this completed plan and
will receive its own branch and ExecPlan.

## Context and Orientation

`bindings/mlir/qiskit/QiskitTranslation.h` defines normalized data shared by the
generic MLIR translator and each supported Qiskit version. `CircuitWriter`
accepts flat operations and `addControlFlow`, which owns normalized metadata and
one writer for each nested block.

`bindings/mlir/qiskit/QiskitExport.cpp` converts one `mlir::QCProgram` into that
normalized writer stream. `ExportState` discovers qubit resources, returned
`!cbit.reg<N>` values, scalar parameters, and recursively collected
instructions. A CBit register is a first-class SSA value. `cbit.load` reads one
element, `cbit.store` writes one element, and `cbit.get_reg` plus
`cbit.get_index` describe a measurement destination. Each SCF region becomes a
nested circuit block over all root qubits and classical bits.

An SCF operation is MLIR's structured-control representation. `scf.if` has one
or two regions, `scf.for` has a constant iteration range, `scf.while` has a
condition region and a body region, and `scf.index_switch` has labeled case
regions plus a default region. Supported exported forms have no general SSA
results. The only accepted result-bearing `scf.if` form is a pure Boolean select
that reconstructs a Qiskit classical expression.

A classical snapshot is a `cbit.load` result used later in a condition or
expression. Export is valid only if no intervening store can make that loaded
value stale before the control-flow operation consumes it. A definite write is
an unconditional validated top-level measurement store. Definite-write tracking
prevents an undefined returned CBit from being read before it gains a
Qiskit-representable value.

`bindings/mlir/qiskit/Qiskit2_5.cpp` implements the version-specific reader and
writer. The reader and its public-Python expression capture logic stay
unchanged. The root writer and its children share scalar symbols by their
validated unique names. `PythonClassicalBuilder` reconstructs normalized
expression trees. The writer emits one temporary native barrier at each
control-flow position, finalizes child writers against the parent's exact bit
objects, creates Python control-flow operations, and replaces the barriers in
place.

`test/python/test_mlir_qiskit_translation.py` contains the end-to-end import and
export contract. `docs/mlir/python_compiler_collection.md` contains the public
support table and its exact restrictions.

## Plan of Work

The implementation adds `CircuitWriter::addControlFlow` in
`bindings/mlir/qiskit/QiskitTranslation.h`. The method accepts a
`ControlFlowKind`, one classical target, loop and switch metadata, owned block
writers. Each block writer has the full root qubit and classical-bit counts.

The `PythonClassicalBuilder` in `bindings/mlir/qiskit/Qiskit2_5.cpp` turns
constants, captured Clbits, captured ClassicalRegisters, casts, indexing, unary
operations, and binary operations into Qiskit's public expression objects.
`NativeCircuitWriter` appends a zero-operand native barrier as a placeholder for
each control-flow operation. During `finish`, it converts native circuits to
Python, rebases each nested block onto the parent's exact Qubit and Clbit
objects, uses the symbol table shared by the writer tree, builds the public
control-flow operations, and replaces the placeholders in place. It validates
native writer compatibility and bit counts before construction; generic
preflight has already validated block shape, loop metadata, and switch labels.

`bindings/mlir/qiskit/QiskitExport.cpp` preserves the existing scalar parameter
normalizer and resource discovery while adding recursive circuit and
control-flow records, expression reconstruction, packed-register recognition,
snapshot validation, loop projection, recursive collection, recursive
constructible-gate validation, and recursive writer emission. It uses only CBit
operations for classical state and preflights all unsupported results, dynamic
indices or bounds, signed or over-wide expressions, non-finite values, repeated
labels, stale snapshots, repeated measurement destinations, and unsupported loop
forms before writer allocation.

For undefined returned CBit registers, scan validated stores in the entry block
in program order. Only an unconditional measurement store makes its destination
definitely written. Reject any exported load of an undefined bit before its
first definite write. A store inside nested control flow may be exported as a
measurement destination but cannot establish top-level initialization.
Zero-initialized CBit allocations need no synthetic stores because Qiskit starts
classical bits at zero.

Focused tests cover nested if/while/switch captures, register conditions,
Boolean select expressions, loop ranges and identity, empty branches, rejection
without source mutation, undefined CBit definite writes, stale snapshots,
malformed labels, and unsupported expression forms. The MLIR test functions
return all public classical registers, the existing import capture tests remain
unchanged, and `docs/mlir/python_compiler_collection.md` records the support
table and exact structured-export restrictions.

## Concrete Steps

Run all commands from the repository root. Inspect formatting throughout:

    git diff --check
    clang-format --dry-run --Werror bindings/mlir/qiskit/Qiskit2_5.cpp \
      bindings/mlir/qiskit/QiskitExport.cpp \
      bindings/mlir/qiskit/QiskitTranslation.h
    uvx ruff check test/python/test_mlir_qiskit_translation.py

Configure and build the release MLIR binding if this isolated worktree does not
already have a compatible build:

    cmake --build build/python/Release --target mqt-core-mlir-bindings --parallel 8

Run focused tests while iterating, then the complete translation file against
the worktree-built extension:

    uv run --no-sync pytest test/python/test_mlir_qiskit_translation.py \
      -k 'control_flow or expression or measurement_store'
    uv run --no-sync pytest test/python/test_mlir_qiskit_translation.py

Regenerate Python stubs after the binding changes and build the complete
documentation after updating the support table:

    uvx nox -s stubs
    uvx nox --non-interactive -s docs

The complete documentation build was explicitly deferred for the final handoff;
run it before merging if the pull-request checks do not cover it.

Run the repository lint session last after each completed commit-sized batch:

    uvx nox -s lint

## Validation and Acceptance

An exported result-free `scf.if`, constant-range `scf.for`, expression-based
`scf.while`, or result-free `scf.index_switch` must produce the matching Qiskit
operation. Importing that Qiskit circuit again must succeed. Captured bits must
refer to the same root Clbit objects, and packed registers must retain
little-endian bit order. A live loop induction value must use one Python
Parameter identity in the loop metadata and every nested gate expression.

An undefined returned CBit may be measured unconditionally and then read. A load
before that write, a conditional-only initializing write, a duplicate
destination, a dynamic destination, or a stale snapshot must fail before Qiskit
writer allocation. Zero-initialized returned CBits need no emitted initializer.
All rejected exports must leave the source MLIR text unchanged.

The commit is accepted when the release binding builds, all Qiskit translation
tests pass, lint passes, and the diff changes only the writer interface, generic
exporter, Qiskit 2.5 writer, tests, documentation, and this plan.

## Idempotence and Recovery

All build, format, lint, and test commands are repeatable. Source edits stay in
this dedicated worktree and do not modify other task worktrees. The generic
exporter finishes validation before it calls `selectTranslation` or allocates a
writer, so failures cannot expose a partial Qiskit circuit. If Python
post-processing fails, `finish` owns and discards its incomplete local objects.
Do not cherry-pick the earlier combined implementation because it would restore
obsolete MemRef classical state and overwrite the reviewed scalar and import
models.

## Artifacts and Notes

The starting commit already passes captured-expression import tests and uses
unique symbol names in closed `Parameter` trees. The old combined implementation
is a design reference only. The final commit boundary is:

    structured export: interface + recursive collector + deferred writer +
    tests + support documentation + this plan

## Interfaces and Dependencies

At completion, `CircuitWriter` has this additional virtual operation:

    void addControlFlow(
        ControlFlowKind kind, ClassicalTarget target, Loop loop,
        std::vector<SwitchCase> switchCases,
        std::vector<std::unique_ptr<CircuitWriter>> blocks);

`QiskitExport.cpp` owns `ExportedCircuit` and `ExportedControlFlow` records and
recursively calls this operation only after complete preflight. `Qiskit2_5.cpp`
implements the operation with deferred public-Python construction while keeping
native gate and scalar parameter creation. No new dependency is introduced. The
implementation uses LLVM and MLIR utilities already linked by the binding,
nanobind for public Python objects, and Qiskit 2.5's existing C API.

Revision note: Created the self-contained plan after comparing the reviewed
CBit/scalar/import base with the earlier combined implementation, then closed it
after the release build, complete translation tests, lint, and semantic review.
Updated it for the final audit fixes and restack onto the amended import parent.
Updated it again after #2175 changed the scalar representation and metadata
contract; only the export-specific delta was replayed. Recorded the final
low-bit and constant-index corrections together with the required stubs, lint,
deferred documentation build, and 208-test validation after the last parent
update. Recorded the squash-merge resolution and the bounded-preflight and dead
loop-parameter fixes found during the post-merge audit. Recorded the final
complexity pass that shares parameter state, replaces deferred insertion with
native placeholders, makes the all-root-bit invariant explicit, and reduces
repeated test and documentation work; recorded the successful build, unchanged
stubs, clean lint, and 208-test Qiskit 2.5.0-2.5.2 matrix. Merged `main` through
`84ace8ef2`, preserved its QCO switch, deterministic finalization,
loop-unrolling, and nanobind 3 changes, and made the vendored Qiskit C API
initialization safe for the new free-threaded binding mode. The split-mode
binding build and all 209 Qiskit 2.5.2 translation tests passed.
