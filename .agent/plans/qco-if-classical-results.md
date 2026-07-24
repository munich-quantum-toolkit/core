# Add SSA classical results to QCO conditionals

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

After this change, a QC program can use an `scf.if` to compute ordinary
classical values while also updating quantum state, convert to QCO, and convert
back to QC without allocating scratch memory or losing either kind of result.
The QCO dialect will represent that behavior directly: a `qco.if` returns an
ordinary SSA-value prefix followed by the existing linear quantum-value suffix.
Focused dialect, conversion, mapping, and round-trip tests will demonstrate that
the classical value remains connected to its consumer while QCO's explicit
single-use quantum flow remains intact.

## Progress

- [x] (2026-07-24 19:05Z) Create this work on top of the reduced structured
  `for`/`while` conversion fix.
- [x] (2026-07-24 19:05Z) Compare SSA-based `qco.if` results, retained `scf.if`,
      and scratch-memory alternatives with an independent design review.
- [x] (2026-07-24 19:38Z) Extend `qco.if`, `qco.yield`, their verifiers,
      parser/printer, builders, tied-value helpers, and region-branch interface
      for mixed results.
- [x] (2026-07-24 19:38Z) Update QCO consumers, canonicalization, dead-gate
      handling, and mapping so only the linear suffix participates in quantum
      dataflow and routing.
- [x] (2026-07-24 19:38Z) Add QC-to-QCO and QCO-to-QC lowering for classical
      conditional results.
- [x] (2026-07-24 19:38Z) Add a feature-specific QCO-to-Jeff diagnostic and
      explicitly reject result-bearing `scf.index_switch` in QC-to-QCO.
- [x] (2026-07-24 19:38Z) Add dialect, conversion, round-trip, mapping, and
      diagnostic regressions.
- [x] (2026-07-24 19:42Z) Build the affected targets, run focused and complete
      suites, run `uvx nox -s lint`, and record the evidence here.
- [ ] Keep the branch local until the prerequisite structured-loop PR is merged;
      rebase then prepare it for human review without opening a PR
      automatically.

## Surprises & Discoveries

- Observation: `scf.if` can return arbitrary types but has no explicit quantum
  input-to-block-argument ties. Reusing it with QCO linear values would weaken
  the representation relied on by `WireIterator`, mapping, and dead-gate
  analysis. Evidence: `qco.if` exposes custom tied-value helpers in
  `mlir/include/mlir/Dialect/QCO/IR/QCOOps.h`, while `scf.if` captures values
  from above its regions.
- Observation: `qco.yield` is shared by conditionals, index switches, and gate
  modifiers. Relaxing its generated operand constraint requires a parent-aware
  verifier so modifier and index-switch invariants do not regress. Evidence: all
  five parent operations terminate their regions with the same `qco::YieldOp`.
- Observation: mapping previously treated every SCF loop result as a qubit. The
  prerequisite branch now filters mixed loop state and preserves classical
  terminator operands, which provides the routing pattern needed for mixed
  `qco.if` yields.
- Observation: MLIR's generic region-branch canonicalization rebuilds this
  result-segmented custom operation without preserving its segment property.
  Registering that pattern changed a mixed `qco.if` into an invalid operation.
  The existing QCO-specific constant-condition and condition-propagation
  patterns cover the useful folds, so the generic pattern remains unregistered.
- Observation: the MLIR parser verifies operations before returning a module.
  Invalid mixed yields are therefore rejected during `parseSourceString`,
  allowing tests to assert the precise `qco.yield` diagnostic directly.

## Decision Log

- Decision: Give `qco.if` two result segments ordered as classical values first
  and linear values second. Rationale: this keeps classical computation in
  ordinary SSA, preserves the existing explicit QCO quantum-flow contract, and
  matches QC-to-QCO's convention of appending quantum state to existing SCF
  state. Date/Author: 2026-07-24, Codex after independent design review.
- Decision: Keep operands and branch block arguments linear-only. Rationale:
  classical inputs can be captured normally, while quantum inputs need explicit
  ties to enforce single-use flow. Date/Author: 2026-07-24, Codex.
- Decision: Generalize `qco.yield` syntactically but recover strict typing with
  a parent-aware verifier. Rationale: one terminator can then express mixed
  conditional results without weakening modifier or index-switch semantics.
  Date/Author: 2026-07-24, Codex.
- Decision: Do not add classical results to `qco.index_switch` in this change.
  Rationale: its parser, result layout, mapping, and both conversions form a
  separate reviewable surface. QC-to-QCO must diagnose that case explicitly
  rather than use scratch memory. Date/Author: 2026-07-24, Codex.
- Decision: Jeff conversion may reject classical-result `qco.if` with a precise
  capability diagnostic. Rationale: QC to QCO to QC to QIR is the primary
  pipeline; Jeff limitations must be explicit but must not constrain valid QCO
  representation. Date/Author: 2026-07-24, Codex.
- Decision: Do not register MLIR's generic region-branch canonicalization for
  `qco.if` until it can preserve `resultSegmentSizes`. Rationale: retaining the
  QCO-specific patterns is both smaller and correct for mixed results.
  Date/Author: 2026-07-24, Codex.

## Outcomes & Retrospective

The implementation avoids runtime memory operations and limits the dialect
change to one conditional operation plus its shared terminator. Focused tests
demonstrate textual round-tripping, verifier diagnostics, constant folding,
QC-to-QCO and QCO-to-QC conversion, a composed round trip, mapping, and the
intentional Jeff and index-switch capability diagnostics.

All affected suites pass: 447 QCO IR tests, 134 QC-to-QCO tests, 131 QCO-to-QC
tests, 14 mapping tests, 82 QCO utility tests, and 117 Jeff round-trip tests.
The changed C++ files produce no local clang-tidy 22 warnings. Targeted
repository hooks, `uvx nox -s lint`, and `git diff --check` pass. The branch
remains local on top of the reduced structured-loop PR and awaits that PR's
merge before rebasing and publication.

## Context and Orientation

QC is the reference-style quantum dialect: quantum operations mutate logical
references and SCF operations carry classical SSA values. QCO is the explicit
linear-value form used for analyses and transformations: each quantum operation
consumes a quantum value and produces its successor. The conversion in
`mlir/lib/Conversion/QCToQCO/QCToQCO.cpp` discovers which quantum values cross
an SCF region and appends explicit state. The reverse conversion lives in
`mlir/lib/Conversion/QCOToQC/QCOToQC.cpp`.

`qco.if` is declared in `mlir/include/mlir/Dialect/QCO/IR/QCOOps.td`, with
hand-written parsing, printing, verification, tied-value helpers, and
replacement helpers in `mlir/lib/Dialect/QCO/IR/QCOOps.cpp`. Its regions receive
only linear quantum block arguments and end in `qco.yield`. `qco.yield` is also
used by `qco.index_switch`, `qco.ctrl`, `qco.inv`, and `qco.pow`, so changes to
its ODS type constraint must be paired with parent-specific verification.

QCO mapping in `mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp` extends
structured operations with all physical qubits and may reorder yielded quantum
values during routing. Utilities under `mlir/lib/Dialect/QCO/Utils/` follow tied
quantum values and remove dead gates. These consumers must use the linear result
segment, never the complete mixed result range.

Generated TableGen files and generated dialect documentation are build outputs.
Do not edit or commit them. Follow `AGENTS.md` and `docs/ai_usage.md`; this plan
does not authorize pushing the branch or opening a pull request.

## Plan of Work

First change `IfOp` in `QCOOps.td` so it has an `AnyType` classical-result
segment followed by its existing `LinearType` result segment, guarded by
`AttrSizedResultSegments`. Keep condition and quantum operands unchanged. Change
`YieldOp` to accept arbitrary operands and constrain its valid parents with
`ParentOneOf`. In `QCOOps.cpp`, teach the custom parser to infer the split: the
number of trailing linear results equals the number of `args(...)` assignments,
and all earlier result types are classical. Emit parse errors for inconsistent
counts or non-linear trailing types. Keep the existing textual shape for
quantum-only programs.

Add verification that the classical segment contains no QCO linear type, the
linear segment matches the quantum operands, region arguments match those
operands, and both yields match the complete result signature. Add a
parent-aware `YieldOp::verify()` that compares its operands with the relevant
parent result types; for `qco.ctrl`, compare only the target outputs. Update all
`IfOp` tied-value methods so classical results are never treated as quantum
results. Complete `RegionBranchOpInterface` operand-to-region and
region-to-parent mappings while this contract is explicit.

Audit builders, canonicalization, `ReplaceClassicalControls`, mapping,
`WireIterator`, and QCO dead-gate elimination. Replace ambiguous whole-result
uses with the generated linear-result accessor. When mapping rewrites a
`qco.yield`, retain the classical prefix and permute only its linear suffix.
When a dead conditional has classical results, erase it only if those results
are unused; replace only linear result uses with their tied inputs.

In QC-to-QCO, create `qco.if` with the original `scf.if` result types as its
classical segment and discovered quantum targets as its linear segment. Move the
regions, let the existing second terminator-conversion phase yield original
classical values followed by resolved quantum values, replace the original SCF
results with `getClassicalResults()`, and update quantum mappings from
`getLinearResults()`. Leave an `scf.if` with no quantum state unchanged. Reject
a result-bearing `scf.index_switch` with a precise unsupported-feature
diagnostic.

In QCO-to-QC, create an `scf.if` that returns only the converted classical
types. Replace its linear block arguments with QC references when inlining the
regions, lower each `qco.yield` to an `scf.yield` containing only its classical
prefix, and replace the QCO operation with the new classical SCF results plus
the corresponding QC references for its linear results. Preserve an else region
whenever classical results exist. In QCO-to-Jeff, reject a classical-result
`qco.if` with a feature-specific message before generic conversion fails.

Add tests close to each contract. Dialect tests must cover textual round-trip,
invalid segment and yield signatures, tied-value offsets, region-branch
relationships, and constant-condition canonicalization. Conversion tests must
cover QC-to-QCO, QCO-to-QC, nested conditionals in loops, and a full
QC-to-QCO-to-QC round trip while asserting that no `memref.alloca`, store, or
load exists. Mapping tests must force quantum permutation and prove the
classical yield remains connected. Jeff and index-switch tests must check their
specific diagnostics. Existing quantum-only suites must continue to pass.

## Concrete Steps

From the repository root, inspect definitions and consumers before editing:

    rg -n "IfOp|YieldOp|getResults\\(\\)|getTied" \
      mlir/include/mlir/Dialect/QCO mlir/lib mlir/unittests

Configure and build with the repository's debug preset if needed:

    .agent/run.sh cmake --preset debug
    .agent/run.sh cmake --build build/debug -j 4

Run the focused dialect and conversion binaries discovered under
`build/debug/mlir/unittests`, including the QCO IR, QC-to-QCO, QCO-to-QC,
round-trip, mapping, and Jeff conversion targets. Each must report all tests
passed. Then run:

    .agent/run.sh uvx nox -s lint
    git diff --check

Update `Progress`, `Surprises & Discoveries`, and `Outcomes & Retrospective`
after each milestone, recording exact test counts and any intentionally
unsupported behavior.

## Validation and Acceptance

A parsed and verified `qco.if` with one `i1` result and one `!qco.qubit` result
must print and parse again with the same two result segments. Each branch must
yield exactly `i1, !qco.qubit`; placing a qubit in the classical segment or
yielding a classical value from a modifier must fail verification with a
specific diagnostic.

A QC module containing a result-bearing `scf.if` and quantum operations in both
branches must convert to a verified QCO module containing a mixed-result
`qco.if`. Its classical result must feed the original consumer, and the module
must contain no scratch allocation, store, or load. Converting that module back
to QC must produce a verified result-bearing `scf.if`; an end-to-end round-trip
test must preserve the observable classical return value.

The QCO mapping pass must succeed on a mixed-result `qco.if`, leave the module
verified, preserve the classical yield values, and produce executable two-qubit
operations for the test device. A classical-result `qco.if` sent to Jeff and a
result-bearing `scf.index_switch` sent to QC-to-QCO must fail with their named
capability diagnostics rather than crash or silently lower incorrectly.

All existing quantum-only QCO IR, builder, utility, mapping, QC-to-QCO,
QCO-to-QC, and Jeff tests must pass. The final repository-wide lint session and
`git diff --check` must pass without modifying files.

## Idempotence and Recovery

Build, test, format, and lint commands are repeatable. TableGen regeneration is
performed by CMake into the build directory and must never be copied into the
source tree. If a verifier or parser change breaks existing quantum-only text,
restore compatibility in the custom parser rather than updating unrelated test
fixtures. If a consumer cannot safely handle the classical segment, add an
explicit capability diagnostic before proceeding; do not introduce scratch
memory as a fallback.

Keep all work confined to this task worktree. Preserve unrelated changes and do
not alter the prerequisite worktree. This branch must remain local until its
base PR is merged; rebasing and publication require a separate human decision.

## Artifacts and Notes

The target textual shape is:

    %flag, %q1 = qco.if %cond args(%arg = %q0)
        -> (i1, !qco.qubit) {
      qco.yield %true, %arg : i1, !qco.qubit
    } else args(%arg = %q0) {
      qco.yield %false, %arg : i1, !qco.qubit
    }

The result ordering is a stable internal contract: original classical SCF
results remain first, and conversion-generated QCO linear state is appended.

## Interfaces and Dependencies

`qco::IfOp` must expose generated `getClassicalResults()` and
`getLinearResults()` accessors. Its condition and quantum operand accessors
remain source compatible. Its tied-value helpers accept only linear results and
map them to the corresponding quantum operand and branch argument after
subtracting the classical prefix.

`qco::YieldOp::verify()` must derive its expected operand types from one of
`qco::IfOp`, `qco::IndexSwitchOp`, `qco::CtrlOp`, `qco::InvOp`, or `qco::PowOp`.
No new runtime library or third-party dependency is required. Builtin MLIR
`scf.if` remains the classical control-flow representation on the QC side.

Revision note (2026-07-24): Initial plan created from the independent design
review and the reduced structured-loop prerequisite. It records the SSA result
segmentation, consumer audit, explicit target diagnostics, and local-only
publication boundary.
