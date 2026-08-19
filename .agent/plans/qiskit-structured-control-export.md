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
ordinary operations through the C API, finalizes nested blocks, and then uses
Qiskit's public Python classes to insert the already validated control-flow
operations at their recorded positions.

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

## Surprises & Discoveries

- Observation: The current base already has three independent foundations that
  the old combined implementation did not preserve: first-class CBit registers,
  scalar `Parameter` trees keyed by stable identity, and hybrid native/Python
  import of captured classical expression variables. Evidence: the base commit
  contains `cbit.load`/`cbit.store` export discovery, shared `Parameter` nodes
  with an `identity` field, and `NativeControlFlowReader::rootClbitIndex`.

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

## Decision Log

- Decision: Change only `CircuitWriter`'s output interface and leave all reader
  interfaces untouched. Rationale: the import capture slice is already reviewed
  and does not need exporter construction code. Date/Author: 2026-08-19 / Codex.

- Decision: Represent the collected output as a recursive `ExportedCircuit`
  whose instructions may own one `ExportedControlFlow`. Rationale: validation,
  supported-gate checks, and writer emission must recurse through every block
  before any top-level Qiskit circuit is exposed. Date/Author: 2026-08-19 /
  Codex.

- Decision: Keep scalar parameter trees and stable identities unchanged. Give
  each live `scf.for` induction parameter a generated `ParameterKind::Symbol`
  with one identity shared by `Loop::parameter` and its lexical body. Rationale:
  Qiskit's `ForLoopOp` must use the same Python `Parameter` object that appears
  in body gates, and generated names must not collide with free program inputs.
  Date/Author: 2026-08-19 / Codex.

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

## Outcomes & Retrospective

Structured Qiskit control flow now exports recursively through a normalized,
frontend-neutral plan and a Qiskit 2.5 deferred Python writer. Captured CBits,
packed registers, Boolean/Uint/Float expressions, nested blocks, static loops,
switches, and loop parameter identity round-trip. Preflight rejects stale
snapshots, unsupported expression/result forms, invalid labels, and undefined
CBit reads or returns before allocating the Qiskit writer.

The release MLIR binding builds successfully after the final post-merge restack.
All 197 tests in `test/python/test_mlir_qiskit_translation.py` pass against the
worktree-built extension, and `uvx nox -s lint` passes. The semantic diff leaves
the import reader and existing identity-keyed scalar parameter normalizer
unchanged, while recursively checking that every named scalar input remains
reachable from the emitted top-level or nested Qiskit parameter trees. The
measurement-store relaxation remains out of scope for this completed plan and
will receive its own branch and ExecPlan. Nothing is pushed.

## Context and Orientation

`bindings/mlir/qiskit/QiskitTranslation.h` defines normalized data shared by the
generic MLIR translator and each supported Qiskit version. `CircuitWriter`
currently accepts only flat operations. It will gain `addControlFlow`, which
owns normalized metadata and one writer for each nested block.

`bindings/mlir/qiskit/QiskitExport.cpp` converts one `mlir::QCProgram` into that
normalized writer stream. The current `ExportState` discovers qubit resources,
returned `!cbit.reg<N>` values, scalar parameters, and flat instructions. A CBit
register is a first-class SSA value. `cbit.load` reads one element, `cbit.store`
writes one element, and `cbit.get_reg` plus `cbit.get_index` describe a
measurement destination. Structured export needs a recursive collector because
each SCF region is a nested circuit block with captured root qubits and
classical bits.

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
unchanged. The writer already preserves scalar symbols by `Parameter.identity`.
The new `PythonClassicalBuilder` reconstructs normalized expression trees. The
writer records control-flow insertion points, finalizes child writers against
the parent's exact bit objects, creates Python control-flow operations, and
inserts them in top-down order.

`test/python/test_mlir_qiskit_translation.py` contains the end-to-end import and
export contract. `docs/mlir/python_compiler_collection.md` contains the public
support table and its exact restrictions.

## Plan of Work

First, add `CircuitWriter::addControlFlow` in
`bindings/mlir/qiskit/QiskitTranslation.h`. The method accepts a
`ControlFlowKind`, one classical target, loop and switch metadata, owned block
writers, and the captured root qubit and classical-bit indices.

Next, extend `bindings/mlir/qiskit/Qiskit2_5.cpp`. Add a
`PythonClassicalBuilder` that turns constants, captured Clbits, captured
ClassicalRegisters, casts, indexing, unary operations, and binary operations
into Qiskit's public expression objects. Extend `NativeCircuitWriter` to record
control flow without adding a C placeholder. During `finish`, convert native
circuits to Python, rebase each nested block onto the parent's exact Qubit and
Clbit objects, preserve canonical scalar parameter objects across blocks, build
the public control-flow operations, and insert them at stable instruction
positions. Validate block shape, captures, loop metadata, switch labels, and bit
counts before construction.

Then refactor `bindings/mlir/qiskit/QiskitExport.cpp`. Preserve the existing
scalar parameter normalizer and resource discovery. Add recursive circuit and
control-flow records, expression reconstruction, packed-register recognition,
snapshot validation, loop projection, recursive collection, recursive
constructible-gate validation, and recursive writer emission. Use only CBit
operations for classical state. Preflight all unsupported results, dynamic
indices or bounds, signed or over-wide expressions, non-finite values, repeated
captures or labels, stale snapshots, repeated measurement destinations, and
unsupported loop forms before writer allocation.

For undefined returned CBit registers, scan validated stores in the entry block
in program order. Only an unconditional measurement store makes its destination
definitely written. Reject any exported load of an undefined bit before its
first definite write. A store inside nested control flow may be exported as a
measurement destination but cannot establish top-level initialization.
Zero-initialized CBit allocations need no synthetic stores because Qiskit starts
classical bits at zero.

Add focused tests that cover nested if/while/switch captures, register
conditions, Boolean select expressions, loop ranges and identity, empty
branches, rejection without source mutation, undefined CBit definite writes,
stale snapshots, malformed labels, and unsupported expression forms. Return all
public classical registers from MLIR test functions. Keep the existing import
capture tests unchanged. Update the support table and structured-export
restrictions in `docs/mlir/python_compiler_collection.md`.

## Concrete Steps

Run all commands from the repository root. Inspect formatting throughout:

    git diff --check
    clang-format --dry-run --Werror bindings/mlir/qiskit/Qiskit2_5.cpp \
      bindings/mlir/qiskit/QiskitExport.cpp \
      bindings/mlir/qiskit/QiskitTranslation.h
    uvx ruff check test/python/test_mlir_qiskit_translation.py

Configure and build the release MLIR binding if this isolated worktree does not
already have a compatible build:

    cmake --preset release
    cmake --build build/release --target mqt-core-mlir-bindings --parallel 8

Run focused tests while iterating, then the complete translation file against
the worktree-built extension:

    uv run --no-sync pytest test/python/test_mlir_qiskit_translation.py \
      -k 'control_flow or expression or measurement_store'
    uv run --no-sync pytest test/python/test_mlir_qiskit_translation.py

Run the repository lint session after each completed commit-sized batch:

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
models. Do not push the local commit.

## Artifacts and Notes

The starting commit already passes captured-expression import tests and uses
`Parameter.identity` for scalar symbols. The old combined implementation is a
design reference only. The final commit boundary is:

    structured export: interface + recursive collector + deferred writer +
    tests + support documentation + this plan

## Interfaces and Dependencies

At completion, `CircuitWriter` has this additional virtual operation:

    void addControlFlow(
        ControlFlowKind kind, ClassicalTarget target, Loop loop,
        std::vector<SwitchCase> switchCases,
        std::vector<std::unique_ptr<CircuitWriter>> blocks,
        const std::vector<uint32_t>& qubits,
        const std::vector<uint32_t>& clbits);

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
