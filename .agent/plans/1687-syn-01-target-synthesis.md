# Split target-independent optimization from target-native synthesis

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

Two-qubit optimization and hardware lowering must answer different questions.
Before routing, the compiler should rewrite a constant unitary window only when
doing so strictly reduces its two-qubit gate count, without choosing a hardware
basis. After routing, the compiler should lower operations that the real
`mlir::CompilerTarget` does not support at their mapped provider sites and
remove routing SWAPs on targets that do not declare SWAP native. A final,
independently runnable pass must reject any operation that is not legal for its
actual type, arity, parameter count, provider site IDs, and ordered locus.

After this change, C++ pipeline code can construct these three stages
independently. Focused tests demonstrate a profitable CX cancellation, preserve
isolated and runtime-parameterized gates before routing, lower SWAP to a
target-selected U/CX basis after routing, and reject direction, operation,
arity, parameter, and site mismatches.

## Progress

- [x] (2026-08-03 13:09Z) Verified the assigned isolated worktree, clean status,
      initial CT-01 development base `b6eb95521cb76224c137496f113f38b9ce295854`,
      and repository policy; read `AGENTS.md`, `docs/ai_usage.md`,
      `.agent/PLANS.md`, and the MQT PR remediation instructions.
- [x] (2026-08-03 13:42Z) Inspected `CompilerTarget`, the native-menu parser,
      Euler and Weyl decomposition, two-qubit run scanning, mapping output,
      compiler and Python APIs, focused tests, the changelog, and Simon
      Hofmann's useful #1969 commits.
- [x] (2026-08-03 15:35Z) Removed the duplicate gate enum/parser and obsolete
      textual menu, CLI, C++ `QCOProgram`, Python binding/stub, and
      menu-specific tests. Added only a minimal decomposition adapter from
      `CompilerTarget::SynthesisBasis`.
- [x] (2026-08-03 16:18Z) Implemented independently constructible pre-routing
  optimization, target-native synthesis, and target-conformance passes.
- [x] (2026-08-03 17:02Z) Replaced the broad menu suite with sixteen focused
      stage-contract tests and built and ran the target-synthesis,
      decomposition, and compiler unit-test binaries successfully.
- [x] (2026-08-03 18:33Z) Updated the existing Unreleased changelog entry and
      completed source formatting, changed-file checks, repository lint, and
      initial diff checks.
- [x] (2026-08-03 19:01Z) Resolved all three blockers from independent read-only
      review: reverse-only symmetric-entangler loci, native SWAP authority, and
      native `qco.pow` body handling. The reviewer approved the remediated code
      and the focused suite passes all sixteen tests.
- [x] (2026-08-03 19:20Z) Rebuilt and reran all three affected suites, reran
      changed-file hooks and full repository lint, completed focused
      `clang-tidy` checks and the final diff audit, and prepared one signed
      atomic commit for local handoff without publication.
- [x] (2026-08-03 19:30Z) Restacked the sole SYN commit without content
      conflicts onto the final CT-01 squash
      `f775395a25fddba0a3b54996416d1311bc6ebe71`. The old and restacked commits
      have the same stable patch ID.
- [x] (2026-08-03 19:45Z) Repeated the requested builds, 16/199/215 tests,
      interface and stub checks, focused `clang-tidy`, hooks, lint,
      stale-surface search, and diff audits on the final CT-01 base, then
      amended this evidence into the single signed commit.

## Milestones

### Milestone 1: Make `CompilerTarget` the only capability model

The completed first milestone removes
`mlir/Dialect/QCO/Transforms/Decomposition/NativeGateset.h` and its source. Gate
identity, provider aliases, arity, parameter counts, global capability checks,
and basis selection now remain in `mlir/include/mlir/Compiler/Target.h` and
`mlir/lib/Compiler/Target.cpp`. The new decomposition-only
`NativeSynthesisBasis` adapts the target-selected single-qubit basis to
`EulerBasis` and caches a Weyl decomposer for the target-selected entangler. It
contains no parser, menu, aliases, target classification, or independent basis
selection.

The generated `fuse-two-qubit-unitary-runs` pass and all of its string-menu
surfaces are removed rather than retained as a synthetic target. The later PIPE
slice owns high-level target pipeline composition, so this slice exposes only
the three typed C++ pass factories.

### Milestone 2: Separate optimization, lowering, and verification

The completed pre-routing stage scans maximal constant windows on one pair of
linear QCO wires. It evaluates a canonical U/CX Weyl decomposition and rewrites
only when the synthesized entangler count is strictly smaller than the window's
original two-qubit operation count. It does not accept a target and does not
rewrite isolated or runtime-parameterized gates.

The completed post-routing stage first asks the supplied `CompilerTarget`
whether each one- or two-qubit unitary is supported at its ordered provider
locus. Supported operations, including supported runtime-parameterized gates,
remain untouched. Unsupported constant operations and all ordinary `qco.swap`
operations that the target does not support are lowered through the target's
globally usable synthesis basis. A target-native SWAP remains untouched. For a
symmetric basis entangler available only at the reverse ordered locus, emission
reverses the entangler operands while preserving logical wire identity. The pass
asks for the basis only after finding an actual lowering need, so an incomplete
target can still accept an already-conforming program. An unsupported runtime
gate receives an operation-local matrix diagnostic.

The completed conformance stage traces each qubit operand back to `qco.static`,
including values passing through QCO `if`/`index_switch` and SCF `for`/`while`
regions. It then calls `CompilerTarget::supports` on the real operation and
ordered provider site IDs. It checks unitary, measurement, and reset operations
and reports the actual operation spelling, arity, parameter count, and locus.

### Milestone 3: Prove the contracts and hand off one atomic change

The focused target-synthesis test binary now proves the three stage boundaries,
including unitary equivalence for profitable optimization, SWAP lowering, and a
reverse-only symmetric entangler. It also proves that native SWAP and `qco.pow`
shells remain untouched. The decomposition and compiler suites prove that typed
bases still cover every supported entangler and that removing the old high-level
menu API does not break the remaining compiler pipeline. Changelog curation,
formatting and lint, and independent review are complete. After CT-01 was
squash-merged, the same patch restacked without content conflicts onto the final
target implementation. Post-restack validation and the final signed amend
complete this milestone.

This slice must not add `compileForTarget`, alter default pipeline composition,
or remove the coupling-only `placeAndRoute` overload; those are PIPE
responsibilities. It must not push, open or modify a pull request, or mutate
GitHub without separate authorization.

## Surprises & Discoveries

- Observation: The old two-qubit pass performed four jobs: single-qubit
  target-basis lowering, two-qubit optimization, isolated two-qubit lowering,
  and residual menu checking. Evidence: its `hasNonNativeGate` condition could
  rewrite a pre-routing window without reducing the entangler count.
- Observation: CT-01 already recognizes all fifteen gates understood by the
  deleted menu, including provider aliases and the same entangler preference.
  The old `NativeGateset` duplicated the enum, parser switch, basis resolution,
  and operation classifier.
- Observation: Mapping emits `qco.static` with hardware identifiers, while
  `CompilerTarget` permits sparse provider IDs. Evidence: the conformance tests
  use sites 10 and 20 and distinguish loci `[10, 20]` and `[20, 10]`.
- Observation: Applying an MLIR greedy rewrite driver can reorder an unrelated
  constant even when the quantum pattern does not match. The preservation test
  therefore asserts that the isolated SWAP remains and no entangler is
  introduced, rather than requiring byte-identical generic IR.
- Observation: Initial configuration required network access to fetch pinned
  repository dependencies. Once fetched into the worktree-local build tree,
  focused compilation and tests were repeatable without source workarounds.
- Observation: The first post-restack stub check selected the Python-packaged
  CMake launcher and an unpinned MLIR, then failed during configuration.
  Pointing the check at the real CMake binary and local LLVM/MLIR 22.1.3
  installation built the bindings, regenerated every stub, and left the tracked
  stubs unchanged.
- Observation: Pull request #1969 established the useful progressive ordering of
  decomposition, optional routing, and late native synthesis, with a test that
  routed SWAPs disappear. Its `targetNative` Python duck typing and coupling CLI
  are intentionally excluded.
- Observation: `CompilerTarget` deliberately accepts an operand-symmetric
  entangler as globally usable when either ordered orientation is available,
  while exact operation support remains order-sensitive. Target-native emission
  must therefore choose the supported orientation per mapped locus.
- Observation: `qco.pow`, like `qco.ctrl` and `qco.inv`, is a target-visible
  unitary shell with a region body. Synthesis and conformance must classify the
  shell and skip its implementation body.
- Observation: Independent review also identified non-blocking follow-up
  opportunities: failure-atomic preflight, sink-only source validation,
  dedicated `index_switch`/`while` regressions, all single-qubit adapter bases,
  and more legacy scanner regressions. These are classified as future hardening
  rather than expanding this slice after its requested contracts pass.

## Decision Log

- Decision: Remove the native-menu pass and all high-level menu APIs in this
  slice instead of retaining a compatibility adapter. Rationale: the series
  intentionally moves callers to a typed `CompilerTarget`; a synthetic menu
  target would preserve the parallel configuration model this change removes.
  Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Keep `NativeSynthesisBasis` as a minimal decomposition-only adapter
  directly constructed from `CompilerTarget::SynthesisBasis`. Rationale: Euler
  emission and cached Weyl decomposition need transform-specific values, but
  target capability semantics and basis selection must remain centralized.
  Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Measure pre-routing profitability by a strict reduction in two-qubit
  basis uses and materialize a canonical U/CX sequence only after the comparison
  succeeds. Rationale: two-qubit operations drive routing cost, and selecting a
  hardware basis before routing would conflate optimization with target
  legality. Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Let `CompilerTarget::supports` decide whether an ordinary `qco.swap`
  requires post-routing lowering. Rationale: the target is the sole capability
  authority; routing SWAPs still lower on ordinary targets that do not report
  SWAP, while a target-native SWAP must remain legal even when the target has no
  global synthesis basis. Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: For an operand-symmetric target basis supported only at the reverse
  ordered locus, emit each entangler with reversed operands and map its results
  back to the original logical wires. Rationale: global target basis selection
  accepts either orientation for symmetric gates, but final conformance checks
  the exact provider order. Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Do not require `CompilerTarget::synthesisBasis()` at pass
  construction or pass entry. Rationale: absent operations mean all operations
  are native, and an incomplete explicit target can still describe a conforming
  program. Missing-basis failure matters only after an unsupported operation
  actually needs lowering. Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Keep optimization, target-native synthesis, and conformance as
  separate manual factories rather than textual passes. Rationale:
  `CompilerTarget` is an immutable typed C++ value that cannot be faithfully
  represented by generic pass options, and separate factories make each stage
  independently testable and benchmarkable. Date/Author: 2026-08-03, GPT-5.6 via
  Codex.
- Decision: Preserve Simon Hofmann's authorship in the final commit. Rationale:
  the implementation materially carries forward the progressive post-routing
  behavior and SWAP-removal coverage from his #1969 work while excluding the
  unapproved Python and CLI designs. Date/Author: 2026-08-03, GPT-5.6 via Codex.

## Outcomes & Retrospective

The implementation now has one capability authority and three separately
observable transform stages. On the final CT-01 base, sixteen focused tests pass
along with all 199 decomposition and 215 compiler tests. The relevant CMake
interface-header targets, CLI build and surface checks, and full Python stub
regeneration pass; regeneration leaves the tracked stubs unchanged. Independent
review approved the remediated code with no remaining code blockers.
Changed-file hooks, full repository lint, stale-surface search, and
`git diff --check` pass. Focused `clang-tidy` is clean for the new synthesis
source, basis adapter, and both modified test sources; the Weyl source reports
only three pre-existing warnings on unchanged header lines. The result is
contained in one signed local commit directly on the final CT-01 squash and is
not pushed.

The main design lesson is that target support and resynthesis profitability must
not share a configuration surface. The canonical U/CX optimizer is useful
without hardware knowledge, while post-routing lowering and conformance require
the exact target and provider-site order.

## Context and Orientation

`mlir/include/mlir/Compiler/Target.h` defines the immutable target. Its
`supports(Operation*, locus)` query recognizes QCO operation semantics and
checks provider site IDs, ordered loci, arity, and parameter count. Its
`synthesisBasis()` query returns one globally usable single-qubit basis and
entangler only when both exist.

`mlir/lib/Dialect/QCO/Transforms/NativeSynthesis/TargetSynthesis.cpp` contains
the constant-window scanner, three pass implementations, provider-site tracing,
and diagnostics. Public factory declarations live in
`mlir/include/mlir/Dialect/QCO/Transforms/Passes.h`.

`mlir/include/mlir/Dialect/QCO/Transforms/Decomposition/SynthesisBasis.h` and
its source adapt the typed target basis for the existing Euler and Weyl
emitters. `mlir/lib/Dialect/QCO/Transforms/Decomposition/Weyl.cpp` emits the
selected entangler and single-qubit factors.

QCO qubits use linear static single assignment: each operation consumes a qubit
value and returns its successor. A mapped hardware qubit starts at `qco.static`;
structured control flow passes the value through block arguments and results.
Conformance must trace that value chain rather than interpret SSA positions as
hardware IDs.

Focused tests are in
`mlir/unittests/Dialect/QCO/Transforms/NativeSynthesis/test_target_synthesis.cpp`.
Typed basis coverage remains in
`mlir/unittests/Dialect/QCO/Transforms/Decomposition/test_weyl_decomposition.cpp`.

## Plan of Work

The implementation first deletes the duplicate gateset files and rewires Weyl
synthesis to the target-derived adapter. It then removes the obsolete generated
pass, registration, CLI option, `QCOProgram` method, binding, stub, and their
menu tests.

The new source retains the proven two-qubit window scanner but separates its
uses. The optimizer compares original and canonical entangler counts. Target
synthesis classifies actual mapped operations and rewrites only lowering needs.
Conformance performs an independent read-only walk and exact target query.

The final work updates the existing Unreleased changelog entry without inventing
a pull-request reference or upgrade note, formats changed sources, runs focused
and repository-required validation, obtains an independent read-only review,
addresses findings, and creates one signed commit with `Assisted-by` and Simon
Hofmann `Co-authored-by` trailers.

## Concrete Steps

Run all commands from the repository root in the task's isolated worktree.
Configure with the repository's release preset and an LLVM/MLIR 22 installation:

    .agent/run.sh env MLIR_DIR=<LLVM 22 MLIR cmake directory> \
      cmake --preset release

Build and run the three affected C++ suites:

    .agent/run.sh cmake --build build/release --target \
      mqt-core-mlir-unittest-target-synthesis \
      mqt-core-mlir-unittest-decomposition \
      mqt-core-mlir-unittests-compiler

    .agent/run.sh \
      build/release/mlir/unittests/Dialect/QCO/Transforms/NativeSynthesis/\
      mqt-core-mlir-unittest-target-synthesis

    .agent/run.sh \
      build/release/mlir/unittests/Dialect/QCO/Transforms/Decomposition/\
      mqt-core-mlir-unittest-decomposition

    .agent/run.sh \
      build/release/mlir/unittests/Compiler/\
      mqt-core-mlir-unittests-compiler

The observed summaries are:

    [  PASSED  ] 16 tests.
    [  PASSED  ] 199 tests.
    [  PASSED  ] 215 tests.

Finish validation with changed-file hooks, the repository lint suite, and diff
checks:

    .agent/run.sh prek run --files <all changed source and documentation files>
    .agent/run.sh uvx nox -s lint
    .agent/run.sh git diff --check
    .agent/run.sh git status --short

## Validation and Acceptance

Acceptance requires the following observable behavior:

- Two adjacent constant CX operations optimize away with equivalent unitary
  behavior, while a three-CX SWAP form, an isolated SWAP, and a runtime RXX run
  remain quantum-structurally unchanged before routing.
- Target-native synthesis removes unsupported ordinary SWAP and produces only
  operations accepted by a U/CX target, preserving the complete unitary. A
  target-native SWAP remains unchanged without requiring a synthesis basis.
- A reverse-only symmetric entangler is emitted at the supported ordered locus
  while preserving complete unitary behavior, and a native `qco.pow` shell is
  checked without separately rejecting its implementation body.
- An absent operation set succeeds without synthesis. An explicit incomplete
  target succeeds for supported operations and reports “no globally usable
  synthesis basis” only when an unsupported operation actually needs lowering.
- A supported runtime-parameterized gate remains unchanged. An unsupported
  runtime gate reports that its unitary matrix is unavailable at compile time.
- Conformance distinguishes sparse provider IDs and ordered direction, rejects
  operation-type, arity, parameter-count, unknown-site, and measurement
  mismatches, and traces QCO and SCF structured control flow.
- The duplicate enum/parser and all native-menu text, CLI, C++ program, Python,
  and generated pass surfaces are absent.
- Focused tests, decomposition tests, compiler tests, changed-file checks,
  repository lint, and `git diff --check` pass.
- The final commit is signed, carries the required `Assisted-by` and
  `Co-authored-by: Simon Hofmann <simon.t.hofmann@tum.de>` trailers, and is not
  pushed.

## Idempotence and Recovery

Configuration, builds, tests, and hooks are repeatable in the task-local build
and cache directories. A failed pass test uses a disposable module fixture.
Network fetch failures during initial configuration should be retried with
authorized network access; source behavior must not be weakened to accommodate
an environment boundary.

Do not reset or clean the worktree. Inspect status and remove only
task-generated artifacts if necessary. Do not modify another task's worktree or
shared branch metadata. External publication remains unauthorized.

## Artifacts and Notes

Initial development used CT-01 commit
`b6eb95521cb76224c137496f113f38b9ce295854`. The final one-commit SYN patch is
restacked directly onto the squash-merged CT-01 base
`f775395a25fddba0a3b54996416d1311bc6ebe71`. The stable patch ID before and after
the restack is `fb73dd9d29254dfca2edf68df7eb9bf983284cc4`, proving that no SYN
patch content changed during the history rewrite.

Relevant #1969 source commits are `3be6d8e43` for target-native naming and
`1aa4c975b` for progressive post-routing synthesis and SWAP-removal coverage,
both authored by Simon Hofmann.

Independent read-only review initially blocked the candidate on three issues:
reverse-only symmetric-entangler loci, a SWAP capability override, and nested
`qco.pow` body checking. The exact remediated source rebuilt successfully, all
sixteen focused tests passed, and the reviewer reported that all three findings
were closed with no new code blockers. The reviewer also reran
`git diff --check` against the initial development base.

Pre-restack local validation rebuilt `mqt-core-mlir-unittest-target-synthesis`,
`mqt-core-mlir-unittest-decomposition`, and `mqt-core-mlir-unittests-compiler`,
then observed 16, 199, and 215 passing tests, respectively. Changed-file `prek`
hooks and `uvx nox -s lint` passed. The first final lint attempt encountered
sandbox DNS failure; the authorized network retry reused the worktree-local
environment and passed every hook. The four metadata messages for deliberately
deleted native-menu files were non-fatal and every reported hook passed.

Post-restack validation rebuilt those three binaries from final CT-01 and again
observed 16, 199, and 215 passing tests. It also built
`MLIRQCOTransforms_verify_interface_header_sets`,
`MQTCompilerPipeline_verify_interface_header_sets`,
`MQTCompilerTarget_verify_interface_header_sets`, `mqt-cc`, and
`mqt-cc_verify_interface_header_sets`. The resulting CLI help contains neither
`--native-gates` nor `fuse-two-qubit-unitary-runs`. Stub generation built the
bindings with LLVM/MLIR 22.1.3 and produced no tracked diff. Five focused
`clang-tidy` invocations passed; the only diagnostics are three inherited
warnings on unchanged `Weyl.h` lines 167 and 171. Changed-file hooks, full
repository lint, stale-surface search outside historical plans, both worktree
and committed `git diff --check`, and the final status audit pass.

## Interfaces and Dependencies

`CompilerTarget` remains dependency-light and does not depend on transform
libraries. `MLIRQCOTransforms` links privately to `MQTCompilerTarget`. The
public, independently constructible factories are:

    std::unique_ptr<Pass> createOptimizeTwoQubitUnitaryRuns();
    std::unique_ptr<Pass>
    createTargetNativeSynthesis(const CompilerTarget& target);
    std::unique_ptr<Pass>
    createVerifyTargetConformance(const CompilerTarget& target);

There is no textual target-native pass, synthetic target, native-gate menu, or
parallel capability model.
