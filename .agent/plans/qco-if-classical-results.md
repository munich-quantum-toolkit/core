# Add SSA classical results to QCO conditionals

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

After this change, a QC program can use an `scf.if` or `scf.index_switch` to
compute ordinary classical values while also updating quantum state, convert to
QCO, and convert back to QC without allocating scratch memory or losing either
kind of result. The QCO dialect will represent that behavior directly: `qco.if`
and `qco.index_switch` return an ordinary SSA-value prefix followed by the
existing linear quantum-value suffix. Focused dialect, conversion, mapping, and
round-trip tests will demonstrate that classical values remain connected to
their consumers while QCO's explicit single-use quantum flow remains intact.

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
- [x] (2026-07-24 21:08Z) Rebase the two local commits onto the merge of the
      prerequisite structured-loop PR #1935 without replaying its historical
      commits.
- [x] (2026-07-24 21:34Z) Extend `qco.index_switch`, its parser, printer,
      verifier, tied-value helpers, target extension, region-branch interface,
      builders, and equivalence support for mixed classical and linear results.
- [x] (2026-07-24 21:34Z) Preserve classical `scf.index_switch` results through
      QC-to-QCO and QCO-to-QC without scratch storage and add focused round-trip
      regressions.
- [x] (2026-07-24 22:07Z) Extend mapping, tensor traversal, and module
      equivalence to distinguish each conditional's classical prefix from its
      tied linear suffix.
- [x] (2026-07-24 22:21Z) Rebuild all affected targets, pass 932 focused tests,
      pass repository lint and changed-source clang-tidy, and run
      `git diff --check`.
- [x] (2026-07-24 23:31Z) Complete an independent read-only review, fix its
      positional-classical-yield equivalence finding, rerun validation, and keep
      the branch local for human inspection.
- [x] (2026-07-24 22:47Z) Add QCO-specific `qco.if` canonicalization patterns
      that forward equal classical yields, coalesce duplicate classical result
      columns, and remove unused classical results without changing the tied
      linear suffix.
- [x] (2026-07-24 22:47Z) Move the QC-to-QCO-to-QC composition regressions into
      a dedicated round-trip test target so each directional conversion test
      links only its own conversion library.
- [x] (2026-07-24 22:53Z) Rebuild the affected targets, pass 718 affected tests,
      pass changed-source clang-tidy and repository lint, run
      `git diff --check`, and complete an independent read-only verification of
      the follow-up.

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
- Observation: GitHub merged PR #1935 as one commit, so a plain rebase tried to
  replay its historical commits. Rebasing only the two commits after `cc5b3ed46`
  moved the local plan and feature cleanly onto merge commit `745149687`.
- Observation: `qco.index_switch` used the full result list to infer the types
  of its region arguments. Once classical results are added, the assignment list
  itself must define the number of trailing linear result types. The parser now
  uses that count consistently for every case and the default.
- Observation: QCO's module-equivalence and tensor-iterator utilities assumed
  all conditional results were linear. Mixed results exposed incorrect result
  indexing for both `qco.if` and `qco.index_switch`; the utilities now map the
  classical prefix positionally and traverse only the tied linear suffix.
- Observation: mapping had recursive handling for `qco.if` but no equivalent
  handling for the already-existing `qco.index_switch`. Supporting mixed index
  switches therefore also required making every case and the default region a
  routing child, extending their explicit targets, and realigning only their
  yielded linear suffixes. The focused mapping regression exercises two cases, a
  default, one classical result, and two quantum wires.
- Observation: the independent review found that module equivalence still
  compared every `qco.yield` operand as a permutation, even though the new
  classical result prefixes are positional. The comparison now checks the
  classical prefix through `IRMapping` in order and permits permutation only for
  the tied linear suffix. Regressions cover swapped and duplicate classical
  values for both conditional operation forms.
- Observation: `PatternRewriter::eraseOpResults` safely rebuilds a
  result-bearing operation and transfers its regions, but it copies QCO's
  result-segment property unchanged. The dedicated dead-classical-result pattern
  therefore removes the matching yield operands and immediately updates the
  replacement `qco.if` property. Evidence: the canonicalization regression
  reduces four classical results to one, retains the linear result and both
  linear yields, and verifies the resulting module.
- Observation: the former QCO-to-QC round-trip tests conflated two contracts:
  direct reverse conversion and composition of both conversions. The direct
  suite now starts from QCO text for both conditional forms, while
  `mlir/unittests/Conversion/QCQCORoundTrip/` explicitly owns the two-pass tests
  and their intentional two-library dependency.

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
- Decision: Add classical results to `qco.index_switch` in the same change,
  superseding the earlier decision to defer them. Rationale: the user requested
  one coherent conditional-result contract, and index switches can use the exact
  same classical-prefix, linear-suffix representation without scratch memory or
  a separate abstraction. Date/Author: 2026-07-24, Codex.
- Decision: Jeff conversion may reject classical-result `qco.if` with a precise
  capability diagnostic. Rationale: QC to QCO to QC to QIR is the primary
  pipeline; Jeff limitations must be explicit but must not constrain valid QCO
  representation. Date/Author: 2026-07-24, Codex.
- Decision: Do not register MLIR's generic region-branch canonicalization for
  `qco.if` until it can preserve `resultSegmentSizes`. Rationale: retaining the
  QCO-specific patterns is both smaller and correct for mixed results.
  Date/Author: 2026-07-24, Codex.
- Decision: Reproduce the useful classical-result subset of MLIR's `scf.if`
  canonicalization patterns in QCO-specific patterns while leaving the linear
  suffix untouched. Rationale: equal and duplicate classical yields and dead
  classical results can be simplified safely, whereas the generic region-branch
  patterns cannot express QCO's atomic quantum input/argument/yield/result
  bundle. Date/Author: 2026-07-24, Codex.
- Decision: Test bidirectional QC/QCO composition in a dedicated round-trip
  target. Rationale: the QCO-to-QC unit-test target should not depend on the
  reverse conversion merely to construct its inputs; direct tests cover each
  conversion independently, while the separate target names and owns the
  intentional two-way dependency. Date/Author: 2026-07-24, Codex.
- Decision: Restore every mapped `qco.index_switch` region to the parent layout
  before joining instead of attempting pairwise branch convergence. Rationale:
  an index switch has an arbitrary number of regions, and a single explicit
  parent-layout invariant is simpler, deterministic, and reuses the existing
  `qco.if` yield-realignment machinery. Date/Author: 2026-07-24, Codex.

## Outcomes & Retrospective

The implementation avoids runtime memory operations and gives both QCO
conditional operations the same explicit contract: ordinary classical SSA
results followed by tied linear quantum results. The index-switch extension
removes its former conversion rejection and covers parser/printer behavior,
verification, region-branch and tied-value interfaces, builders, module
equivalence, tensor traversal, mapping, and both conversion directions.

After rebuilding the affected targets, the complete focused suites passed: 450
QCO IR tests, 134 QC-to-QCO tests, 132 QCO-to-QC tests, 82 QCO utility tests, 3
QTensor utility tests, 15 mapping tests, and 117 Jeff round-trip tests, for 933
tests in total. Repository lint and changed-source clang-tidy also passed; the
only direct clang-tidy diagnostics outside the filtered source lines were known
generated-MLIR warnings and an analyzer warning in MLIR's
`OperationState::getOrAddProperties`.

The earlier independent read-only review passed the IR design, conversions,
mapping, tensor traversal, performance, compatibility, tests, and scope after
identifying one module-equivalence correctness issue and one documentation
mismatch. Both were fixed and revalidated. The feature is under review in a
draft pull request.

The follow-up adds classical-only `qco.if` simplification without registering
MLIR's unsafe generic region-branch pattern bundle. Equal branch yields are
forwarded, duplicate classical result columns are coalesced, and dead classical
results are removed together with the matching yield operands and segment
metadata. The QCO-to-QC target no longer links QC-to-QCO; two direct QCO-to-QC
regressions and a dedicated two-test QC/QCO round-trip target preserve both
layers of coverage. The affected debug suites pass 451 QCO IR, 131 QCO-to-QC,
134 QC-to-QCO, and 2 round-trip tests. LLVM 22.1.8 clang-tidy reports no
diagnostics in the changed source files, repository lint passes, and
`git diff --check` is clean. Independent read-only verification confirmed that
the patterns preserve the quantum bundle and result-segment metadata, that no
generic region-branch patterns are registered, and that each directional test
target now owns only its conversion dependency.

## Context and Orientation

QC is the reference-style quantum dialect: quantum operations mutate logical
references and SCF operations carry classical SSA values. QCO is the explicit
linear-value form used for analyses and transformations: each quantum operation
consumes a quantum value and produces its successor. The conversion in
`mlir/lib/Conversion/QCToQCO/QCToQCO.cpp` discovers which quantum values cross
an SCF region and appends explicit state. The reverse conversion lives in
`mlir/lib/Conversion/QCOToQC/QCOToQC.cpp`.

`qco.if` and `qco.index_switch` are declared in
`mlir/include/mlir/Dialect/QCO/IR/QCOOps.td`, with hand-written parsing,
printing, verification, tied-value helpers, and replacement helpers in
`mlir/lib/Dialect/QCO/IR/QCOOps.cpp` and `mlir/lib/Dialect/QCO/IR/SCF/`. Their
regions receive only linear quantum block arguments and end in `qco.yield`.
`qco.yield` is also used by `qco.ctrl`, `qco.inv`, and `qco.pow`, so its
generalized ODS type constraint must remain paired with parent-specific
verification.

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

Add QCO-specific `IfOp` canonicalization patterns in
`mlir/lib/Dialect/QCO/IR/SCF/IfOp.cpp`. One pattern forwards an ordinary
classical result when both branches yield the same value and coalesces two
classical result positions when both corresponding branch yields are identical.
A second pattern removes unused classical result positions from the operation
and both `qco.yield` terminators, updating `resultSegmentSizes` at the same
time. These patterns must never inspect or erase linear results, qubit operands,
branch arguments, or the linear yield suffix.

In QC-to-QCO, create `qco.if` with the original `scf.if` result types as its
classical segment and discovered quantum targets as its linear segment. Move the
regions, let the existing second terminator-conversion phase yield original
classical values followed by resolved quantum values, replace the original SCF
results with `getClassicalResults()`, and update quantum mappings from
`getLinearResults()`. Leave an `scf.if` with no quantum state unchanged. Apply
the same segmentation to `scf.index_switch`: preserve its original results as
the classical prefix, append discovered quantum state as the linear suffix, and
give each new QCO region only linear block arguments.

In QCO-to-QC, create an `scf.if` that returns only the converted classical
types. Replace its linear block arguments with QC references when inlining the
regions, lower each `qco.yield` to an `scf.yield` containing only its classical
prefix, and replace the QCO operation with the new classical SCF results plus
the corresponding QC references for its linear results. Preserve an else region
whenever classical results exist. Lower `qco.index_switch` identically to a
result-bearing `scf.index_switch`, retaining the classical yield prefix and
replacing only its linear results with QC references. In QCO-to-Jeff, reject a
classical-result `qco.if` with a feature-specific message before generic
conversion fails.

Add tests close to each contract. Dialect tests must cover textual round-trip,
invalid segment and yield signatures, tied-value offsets, region-branch
relationships, constant-condition canonicalization, equal classical-yield
forwarding, duplicate classical-result coalescing, dead classical-result
removal, and preservation of the complete linear suffix. Conversion tests must
cover QC-to-QCO and QCO-to-QC directly for both conditional operation forms and
nested conditionals in loops. A dedicated QC/QCO round-trip target must own the
full two-pass composition regressions and assert that no `memref.alloca`, store,
or load exists. Mapping tests must force quantum permutation and prove the
classical yield remains connected. Jeff tests must check the target-specific
diagnostic. Existing quantum-only suites must continue to pass.

## Concrete Steps

From the repository root, inspect definitions and consumers before editing:

    rg -n "IfOp|YieldOp|getResults\\(\\)|getTied" \
      mlir/include/mlir/Dialect/QCO mlir/lib mlir/unittests

Configure and build with the repository's debug preset if needed:

    .agent/run.sh cmake --preset debug
    .agent/run.sh cmake --build build/debug -j 4

Run the focused dialect and conversion binaries discovered under
`build/debug/mlir/unittests`, including the QCO IR, QC-to-QCO, QCO-to-QC,
dedicated QC/QCO round-trip, mapping, and Jeff conversion targets. Each must
report all tests passed. Then run:

    .agent/run.sh uvx nox -s lint
    git diff --check

Update `Progress`, `Surprises & Discoveries`, and `Outcomes & Retrospective`
after each milestone, recording exact test counts and any intentionally
unsupported behavior.

## Validation and Acceptance

A parsed and verified `qco.if` or `qco.index_switch` with one `i1` result and
one `!qco.qubit` result must print and parse again with the same two result
segments. Each region must yield exactly `i1, !qco.qubit`; placing a qubit in
the classical segment or yielding a classical value from a modifier must fail
verification with a specific diagnostic.

A QC module containing a result-bearing `scf.if` or `scf.index_switch` and
quantum operations in every region must convert to a verified QCO module
containing the corresponding mixed-result operation. Its classical result must
feed the original consumer, and the module must contain no scratch allocation,
store, or load. Converting that module back to QC must reproduce the
result-bearing SCF operation; end-to-end round-trip tests must preserve the
observable classical return value.

The QCO mapping pass must succeed on mixed-result `qco.if` and
`qco.index_switch` operations, leave the module verified, preserve the classical
yield values, and produce executable two-qubit operations for the test device. A
classical-result `qco.if` sent to Jeff must fail with its named capability
diagnostic rather than crash or silently lower incorrectly.

All existing quantum-only QCO IR, builder, utility, mapping, QC-to-QCO,
QCO-to-QC, QC/QCO round-trip, and Jeff tests must pass. The QCO-to-QC test
target must not link `MLIRQCToQCO`; only the dedicated round-trip target may
link both directional conversion libraries. The final repository-wide lint
session and `git diff --check` must pass without modifying files.

## Idempotence and Recovery

Build, test, format, and lint commands are repeatable. TableGen regeneration is
performed by CMake into the build directory and must never be copied into the
source tree. If a verifier or parser change breaks existing quantum-only text,
restore compatibility in the custom parser rather than updating unrelated test
fixtures. If a consumer cannot safely handle the classical segment, add an
explicit capability diagnostic before proceeding; do not introduce scratch
memory as a fallback.

Keep all work confined to this task worktree. Preserve unrelated changes and do
not alter any other task worktree. The prerequisite PR is merged and this branch
is rebased; publication remains a separate human decision.

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

`qco::IndexSwitchOp` must expose the same two result accessors and retain its
existing index, cases, targets, and region APIs. Its parser derives the number
of linear results from the `args(...)` assignments, and all cases plus the
default must use the same explicit target list. Its tied-value helpers and
target-extension helper operate only on the linear suffix.

`qco::YieldOp::verify()` must derive its expected operand types from one of
`qco::IfOp`, `qco::IndexSwitchOp`, `qco::CtrlOp`, `qco::InvOp`, or `qco::PowOp`.
No new runtime library or third-party dependency is required. Builtin MLIR
`scf.if` remains the classical control-flow representation on the QC side.

Revision note (2026-07-24): Initial plan created from the independent design
review and the reduced structured-loop prerequisite. It records the SSA result
segmentation, consumer audit, explicit target diagnostics, and local-only
publication boundary.

Revision note (2026-07-24): Rebased onto merged PR #1935 and expanded the single
conditional-result contract to `qco.index_switch` at the user's request. The
earlier deferral decision is explicitly superseded, and the plan now covers the
parser, verifier, tied-value, conversion, equivalence, and test surfaces for
both conditional operation forms.

Revision note (2026-07-24): Recorded the completed independent review and its
remediation. Classical yield values are now compared positionally while the
linear suffix remains permutation-aware, with swapped-value and duplicate-value
regressions for both conditional operation forms.

Revision note (2026-07-24): Added the follow-up milestone for QCO-specific
classical `qco.if` canonicalization patterns and separated bidirectional
conversion regressions from the directional QCO-to-QC test target.
