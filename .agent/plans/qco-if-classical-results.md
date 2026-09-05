# Add SSA classical results to QCO conditionals

Status: historical implementation record.

## Goal and scope

After this change, a QC program can use an `scf.if` or `scf.index_switch` to
compute ordinary classical values while also updating quantum state, convert to
QCO, and convert back to QC without allocating scratch memory or losing either
kind of result. The QCO dialect will represent that behavior directly: `qco.if`
and `qco.index_switch` return an ordinary SSA-value prefix followed by the
existing linear quantum-value suffix. Focused dialect, conversion, mapping, and
round-trip tests will demonstrate that classical values remain connected to
their consumers while QCO's explicit single-use quantum flow remains intact.

## Constraints

- `scf.if` can return arbitrary types but has no explicit quantum
  input-to-block-argument ties. Reusing it with QCO linear values would weaken
  the representation relied on by `WireIterator`, mapping, and dead-gate
  analysis. Evidence: `qco.if` exposes custom tied-value helpers in
  `mlir/include/mlir/Dialect/QCO/IR/QCOOps.h`, while `scf.if` captures values
  from above its regions.

- `qco.yield` is shared by conditionals, index switches, and gate modifiers.
  Relaxing its generated operand constraint requires a parent-aware verifier so
  modifier and index-switch invariants do not regress. Evidence: all five parent
  operations terminate their regions with the same `qco::YieldOp`.

- mapping previously treated every SCF loop result as a qubit. The prerequisite
  branch now filters mixed loop state and preserves classical terminator
  operands, which provides the routing pattern needed for mixed `qco.if` yields.

- MLIR's generic region-branch canonicalization rebuilds this result-segmented
  custom operation without preserving its segment property. Registering that
  pattern changed a mixed `qco.if` into an invalid operation. The existing
  QCO-specific constant-condition and condition-propagation patterns cover the
  useful folds, so the generic pattern remains unregistered.

- the MLIR parser verifies operations before returning a module. Invalid mixed
  yields are therefore rejected during `parseSourceString`, allowing tests to
  assert the precise `qco.yield` diagnostic directly.

- `qco.index_switch` used the full result list to infer the types of its region
  arguments. Once classical results are added, the assignment list itself must
  define the number of trailing linear result types. The parser now uses that
  count consistently for every case and the default.

- QCO's module-equivalence and tensor-iterator utilities assumed all conditional
  results were linear. Mixed results exposed incorrect result indexing for both
  `qco.if` and `qco.index_switch`; the utilities now map the classical prefix
  positionally and traverse only the tied linear suffix.

- mapping had recursive handling for `qco.if` but no equivalent handling for the
  already-existing `qco.index_switch`. Supporting mixed index switches therefore
  also required making every case and the default region a routing child,
  extending their explicit targets, and realigning only their yielded linear
  suffixes. The focused mapping regression exercises two cases, a default, one
  classical result, and two quantum wires.

- the independent review found that module equivalence still compared every
  `qco.yield` operand as a permutation, even though the new classical result
  prefixes are positional. The comparison now checks the classical prefix
  through `IRMapping` in order and permits permutation only for the tied linear
  suffix. Regressions cover swapped and duplicate classical values for both
  conditional operation forms.

- `PatternRewriter::eraseOpResults` safely rebuilds a result-bearing operation
  and transfers its regions, but it copies QCO's result-segment property
  unchanged. The dedicated dead-classical-result pattern therefore removes the
  matching yield operands and immediately updates the replacement `qco.if`
  property. Evidence: the canonicalization regression reduces four classical
  results to one, retains the linear result and both linear yields, and verifies
  the resulting module.

- the former QCO-to-QC round-trip tests conflated two contracts: direct reverse
  conversion and composition of both conversions. The direct suite now starts
  from QCO text for both conditional forms, while
  `mlir/unittests/Conversion/QCQCORoundTrip/` explicitly owns the two-pass tests
  and their intentional two-library dependency.

- PR #1939 touched the same Jeff round-trip test file but a different contract.
  Git merged its entry-point return-placement regression cleanly alongside this
  change's precise rejection test for classical conditional results, and the
  complete 118-test Jeff round-trip suite passes.

## Decisions

- Give `qco.if` two result segments ordered as classical values first and linear
  values second. Rationale: this keeps classical computation in ordinary SSA,
  preserves the existing explicit QCO quantum-flow contract, and matches
  QC-to-QCO's convention of appending quantum state to existing SCF state.

- Keep operands and branch block arguments linear-only. Rationale: classical
  inputs can be captured normally, while quantum inputs need explicit ties to
  enforce single-use flow.

- Generalize `qco.yield` syntactically but recover strict typing with a
  parent-aware verifier. Rationale: one terminator can then express mixed
  conditional results without weakening modifier or index-switch semantics.

- Add classical results to `qco.index_switch` in the same change, superseding
  the earlier decision to defer them. Rationale: the user requested one coherent
  conditional-result contract, and index switches can use the exact same
  classical-prefix, linear-suffix representation without scratch memory or a
  separate abstraction.

- Jeff conversion may reject classical-result `qco.if` with a precise capability
  diagnostic. Rationale: QC to QCO to QC to QIR is the primary pipeline; Jeff
  limitations must be explicit but must not constrain valid QCO representation.

- Do not register MLIR's generic region-branch canonicalization for `qco.if`
  until it can preserve `resultSegmentSizes`. Rationale: retaining the
  QCO-specific patterns is both smaller and correct for mixed results.

- Reproduce the useful classical-result subset of MLIR's `scf.if`
  canonicalization patterns in QCO-specific patterns while leaving the linear
  suffix untouched. Rationale: equal and duplicate classical yields and dead
  classical results can be simplified safely, whereas the generic region-branch
  patterns cannot express QCO's atomic quantum input/argument/yield/result
  bundle.

- Test bidirectional QC/QCO composition in a dedicated round-trip target.
  Rationale: the QCO-to-QC unit-test target should not depend on the reverse
  conversion merely to construct its inputs; direct tests cover each conversion
  independently, while the separate target names and owns the intentional
  two-way dependency.

- Restore every mapped `qco.index_switch` region to the parent layout before
  joining instead of attempting pairwise branch convergence. Rationale: an index
  switch has an arbitrary number of regions, and a single explicit parent-layout
  invariant is simpler, deterministic, and reuses the existing `qco.if`
  yield-realignment machinery.

## Outcome and validation

QCO conditionals use ordinary classical SSA results followed by tied linear
quantum results. Conversions, builders, interfaces, module equivalence, tensor
traversal, and mapping support this shared contract. Classical-only `qco.if`
folds update yields and result segments together while preserving the quantum
bundle; generic region-branch patterns are not registered.

Directional conversion tests depend only on their own conversion; a separate
round-trip target covers composition. Final affected QCO IR, both conversions,
round-trip, and jeff suites passed, together with changed-source clang-tidy and
repository lint.

## Code and ownership

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

## Acceptance

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

## Interfaces

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
