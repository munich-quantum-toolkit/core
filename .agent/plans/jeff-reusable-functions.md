# Preserve reusable functions in jeff programs

This ExecPlan follows `.agent/PLANS.md`. Its progress, discoveries, decisions,
and results must stay current. Run commands from the repository root.

## Purpose

Preserve function definitions and calls when exchanging QCO programs with jeff.
Repeated applications of a parameterized circuit should share one helper body,
including after binary serialization. Expand a call only when a quantum modifier
requires it: jeff supports ordinary calls, but its call instruction has no
control, inverse, or power modifier.

## Progress

- [x] (2026-09-04) Rebase onto current `main` and inspect the updated compiler
  structure and pinned jeff reader.
- [x] (2026-09-04) Preserve native function definitions and calls in both
  directions and correct entry-point indexing.
- [x] (2026-09-04) Expand calls only under quantum modifiers and normalize
  exposed global phases.
- [x] (2026-09-04) Remove call-graph ordering and recursion restrictions now
  that the pinned reader registers all function signatures before bodies.
- [x] (2026-09-04) Add binary-roundtrip, metadata, mutable-register, and
  controlled-phase regressions.
- [x] (2026-09-04) Build and pass all 151 jeff conversion tests and all 158
  compiler tests against the dependency revision pinned by current `main`.
- [ ] Run repository lint on the final stack and record the result here.

## Surprises & Discoveries

The jeff schema represents complete programs as a function table and a
designated entry function. It supports function calls directly. The pinned
jeff-mlir dependency serializes and reads `func.func` and `func.call`; no new
schema or private metadata is needed.

The entry-point attribute indexes the function table, not the string table. The
old conversions confused those tables. This can select a wrong function when a
custom gate adds a string or when multiple functions are present.

The pinned reader now registers every function signature before reading bodies.
Forward references and recursion therefore need no exporter-specific ordering or
analysis.

## Decision Log

Use MLIR's function, call, and return type-conversion patterns. Add only the
conversion from `qco.call` to `func.call`. Strip `mqt.unitary` on export because
the verifier describes QC/QCO bodies, not jeff bodies. Do not infer this marker
on import. A complete imported program has one public entry point; make its
other functions private so downstream inlining can remove unused helpers.

Use `inlineCall` only under QC/QCO modifiers, through the existing
`unroll-modifiers` pass. Its native inliner callback collects newly exposed
calls without rescanning the whole program or removing unrelated symbols.
Register promised inliner interfaces in the pass's dependent-dialect registry.
Normalize newly exposed global phases before distributing modifiers.

Preserve live function signatures in QC/QCO cleanup. Omit native dead-value
removal there because it erases unused private-function arguments, including
required borrowed qubits. Keep canonicalization, CSE, and register shrinking.

Keep source function order. Reject declarations and multiple outer blocks
because the serializer requires defined single-block functions. Reject mutable
classical-register helper arguments: their reference semantics cannot be
represented by passing a jeff array value without an explicit ABI.

## Scope and implementation

`mlir/lib/Conversion/QCOToJeff/QCOToJeff.cpp` owns helper signature conversion,
native calls, and serialized module metadata.
`mlir/lib/Conversion/JeffToQCO/JeffToQCO.cpp` owns the inverse type conversion,
entry-point lookup, and private helper visibility. Keep these changes in the
existing conversion libraries.

`mlir/lib/Dialect/MQT/Transforms/UnrollModifiers.cpp` owns call expansion under
modifiers. QC and QCO dialect inliner interfaces supply the legality rules;
verified unitary functions are private, defined, nonrecursive, and single-block.
Do not add a full-program flattening pass or change caller-owned contexts in the
compiler API. Keep ordinary calls and helper definitions intact.

The compiler regression in `mlir/unittests/Compiler/test_compiler_pipeline.cpp`
exports a parameterized helper used twice, serializes it, imports it, and
converts it to QC. The test checks function count, call count, gate body,
visibility, and entry-point identity. Include a custom gate so function and
string indexes differ.
`mlir/unittests/Conversion/JeffRoundTrip/test_jeff_round_trip.cpp` owns the
controlled-helper phase regression and metadata diagnostics, including a forward
reference that exercises the corrected reader.

## Milestones and validation

First build the conversion and compiler tests:

    cmake --preset release
    cmake --build --preset release --target mqt-core-mlir-unittest-jeff-round-trip mqt-core-mlir-unittests-compiler -j4

Run focused tests, then their complete binaries:

    build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler --gtest_filter='CompilerPipelineTest.*Jeff*'
    build/release/mlir/unittests/Conversion/JeffRoundTrip/mqt-core-mlir-unittest-jeff-round-trip
    build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler

Both native calls must survive the binary roundtrip with one shared helper body.
The controlled helper must retain its relative phase and controlled gate. Input
and successful output must verify. QC conversion must remove positional
pass-through quantum results while retaining helper calls.

Finish with:

    uvx nox -s cpp-lint
    uvx nox -s lint

Keep target and QIR integration tests in the subsequent stack layers. Those
changes own flattening required by their output formats and must exercise
imported private jeff helpers.

## Recovery and coordination

Builds and tests are repeatable. Preserve unrelated work and do not modify
another task's worktree. Do not change the external jeff-mlir dependency without
coordination. This plan does not authorize remote actions. Inspect the complete
commit message, sign commits, and verify signatures before publication.

## Outcomes & Retrospective

The focused build and tests pass. The rebased design relies on the format's
native function table and corrected dependency reader, so no call graph,
function reordering, recursion policy, or private metadata is needed. Ordinary
calls remain reusable; only calls inside unsupported quantum modifiers are
expanded. An independent jeff/MLIR review removed one unnecessary whole-module
phase-normalization run when no candidate call can be inlined.
