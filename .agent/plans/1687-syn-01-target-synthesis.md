# Split target-independent gate fusion from target-native synthesis

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

Two-qubit gate fusion and hardware lowering must answer different questions.
Before routing, target-independent two-qubit gate fusion should rewrite a
sequence only when doing so strictly reduces its two-qubit gate count, without
choosing a hardware basis. After routing, the compiler should lower operations
that the real `mlir::CompilerTarget` does not support and remove routing SWAPs
on targets that do not declare SWAP native. A final, independently runnable pass
rejects unsupported operation types, arities, and parameter counts as well as
dynamic allocations and unknown static target sites.

After this change, C++ pipeline code can construct these three stages
independently. Focused tests demonstrate a profitable CX cancellation, preserve
isolated and runtime-parameterized gates before routing, lower SWAP to a
target-selected basis after routing, and reject operation, arity, parameter,
allocation, and site mismatches.

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
- [x] (2026-08-03 16:18Z) Implemented independently constructible two-qubit gate
      fusion, target-native synthesis, and target-conformance passes.
- [x] (2026-08-03 17:02Z) Replaced the broad menu suite with sixteen focused
      stage-contract tests and built and ran the target-synthesis,
      decomposition, and compiler unit-test binaries successfully.
- [x] (2026-08-03 18:33Z) Updated the existing Unreleased changelog entry and
      completed source formatting, changed-file checks, repository lint, and
      initial diff checks.
- [x] (2026-08-03 19:01Z) Resolved all three blockers from independent read-only
      review: single-orientation symmetric-entangler site tuples, native SWAP
      authority, and native `qco.pow` body handling. The reviewer approved the
      remediated code and the focused suite passes all sixteen tests.
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
- [x] (2026-08-03 19:55Z) Published draft PR #1998, fixed its sole changed-file
      `clang-tidy` finding, and added four focused regressions after Codecov
      exposed untested adapter, lowering, fusion, and static-site paths. The
      resulting 20/199 focused tests pass and the two implementation files reach
      91% combined local line coverage.
- [x] (2026-08-03 20:43Z) Observed terminal all-green CI for exact published
      head `b0a520372`, including 92.1% C++ patch coverage and strict
      documentation, then restacked the four SYN commits onto merged MAP-01
      commit `c9e0c0ca5`. Resolved the compiler test by retaining MAP-01's
      undirected coupling input while removing the obsolete native-menu call,
      retained both PR links in the changelog, and passed 20/199/215 tests plus
      the focused high-level mapping API test.
- [x] (2026-08-03 19:54Z) Addressed the latest design and efficiency review as
      one simplification: made `CompilerTarget::SingleQubitBasis` the only basis
      enum, deleted the decomposition adapter, removed entangler operand
      reversal and redundant matrix assertions, selected U/CZ for generic
      fusion, reused Weyl decompositions, validated target sites directly, and
      made target lowering failure-atomic. A direct preplanned `IRRewriter`
      traversal replaces general greedy machinery. Rebuilt the public
      interface-header targets and observed 20/20 target-synthesis, 199/199
      decomposition, and 215/215 compiler tests. Focused changed-source
      `clang-tidy`, targeted hooks, full repository lint, stale-surface search,
      and `git diff --check` pass. A fresh independent exact-working-tree review
      approved the result with no correctness, bloat, or efficiency findings.
- [x] (2026-08-03) Removed the flaky large-scope global-phase timing test,
      adopted `SiteTuple`/target terminology, renamed the pre-routing factory to
      `createFuseTwoQubitGates`, and simplified `CompilerTarget` to homogeneous
      operation capabilities. Removed directional fallback, per-operation site
      tracing, and the two control-flow tests that existed only for that tracer;
      conformance now validates allocation form, quantum function inputs, and
      static target IDs directly. Focused release builds pass all 21
      target-synthesis tests and all 8 compiler-target tests.

## Milestones

### Milestone 1: Make `CompilerTarget` the only capability model

The completed first milestone removes
`mlir/Dialect/QCO/Transforms/Decomposition/NativeGateset.h` and its source. Gate
identity, gate aliases, arity, parameter counts, homogeneous capability checks,
and basis selection now remain in `mlir/include/mlir/Compiler/Target.h` and
`mlir/lib/Compiler/Target.cpp`. `CompilerTarget::SingleQubitBasis` is also the
single type consumed by Euler and Weyl synthesis; the duplicate `EulerBasis`
enum and `NativeSynthesisBasis` adapter are deleted. Weyl retains only cached
decomposers for the target-selected entangler.

The generated `fuse-two-qubit-unitary-runs` pass and all of its string-menu
surfaces are removed rather than retained as a synthetic target. The later PIPE
slice owns high-level target pipeline composition, so this slice exposes only
the three typed C++ pass factories.

### Milestone 2: Separate fusion, lowering, and verification

The completed target-independent two-qubit gate-fusion stage scans maximal
constant sequences on one pair of linear QCO wires. It evaluates a canonical
U/CZ Weyl decomposition and rewrites only when the synthesized entangler count
is strictly smaller than the sequence's original two-qubit operation count. It
does not accept a target and does not rewrite isolated or runtime-parameterized
gates.

The completed post-routing stage asks the supplied `CompilerTarget` whether each
one- or two-qubit unitary belongs to its homogeneous operation set. Supported
operations, including supported runtime-parameterized gates, remain untouched.
Unsupported constant operations and ordinary `qco.swap` operations are lowered
through the target's usable synthesis basis. A target-native SWAP remains
untouched. The pass preflights every lowering need before mutation, asks for a
basis only when needed, and reports an unsupported runtime gate without
partially rewriting the module.

The completed conformance stage checks each real unitary, measurement, and reset
operation against that same homogeneous capability set. It rejects dynamic qubit
allocations and `qco.static` identifiers absent from the target, without tracing
every operation operand through its SSA lineage. Diagnostics report the actual
operation spelling, arity, and parameter count.

### Milestone 3: Prove the contracts and hand off one atomic change

The focused target-synthesis test binary now proves the three stage boundaries,
including unitary equivalence for gate fusion, SWAP lowering, and homogeneous CZ
support. It also proves explicit CZ emission, failure-atomic diagnostics,
static-site validation, dynamic-allocation rejection, and that native SWAP and
`qco.pow` shells remain untouched. The decomposition and compiler suites prove
that the shared basis type covers every supported entangler and that removing
the old high-level menu API does not break the remaining compiler pipeline.

This slice must not add `compileForTarget`, alter default pipeline composition,
or remove the coupling-only `placeAndRoute` overload; those are PIPE
responsibilities. Publishing this latest revision or otherwise mutating GitHub
requires separate revision-scoped authorization.

## Surprises & Discoveries

- Observation: The old two-qubit pass performed four jobs: single-qubit
  target-basis lowering, two-qubit gate fusion, isolated two-qubit lowering, and
  residual menu checking. Evidence: its `hasNonNativeGate` condition could
  rewrite a pre-routing window without reducing the entangler count.
- Observation: CT-01 already recognizes all fifteen gates understood by the
  deleted menu, including gate aliases and the same entangler preference. The
  old `NativeGateset` duplicated the enum, parser switch, basis resolution, and
  operation classifier.
- Observation: Mapping emits `qco.static` with hardware identifiers, while
  `CompilerTarget` permits sparse target IDs. Conformance only needs to validate
  those declarations once; operation capabilities are homogeneous and do not
  require operand-by-operand site tracing.
- Observation: Applying an MLIR greedy rewrite driver can reorder an unrelated
  constant even when the quantum pattern does not match. The target-specific
  transforms now precompute their work and use `IRRewriter` directly, so a no-op
  fusion preserves the module byte-for-byte.
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
- Observation: Current targets expose homogeneous gate sets. Ordered
  `Operation::siteTuples()` retain calibration data, while
  `CompilerTarget::supports()` depends only on canonical operation name, arity,
  and parameter count.
- Observation: `qco.pow`, like `qco.ctrl` and `qco.inv`, is a target-visible
  unitary shell with a region body. Synthesis and conformance must classify the
  shell and skip its implementation body.
- Observation: A generic walk rewrite driver cannot safely anchor this
  multi-operation fusion at the run head because the rewrite erases operations
  the driver has not visited. Precollecting non-overlapping run heads and using
  `IRRewriter` directly is both safer and lighter.
- Observation: The earlier per-operation target-site tracer was unnecessary once
  support became homogeneous. Removing it eliminates both its quadratic worst
  case and its structured-control-flow special cases. Profitable windows also
  reuse the same prepared Weyl decomposition for counting and emission.

## Decision Log

- Decision: Remove the native-menu pass and all high-level menu APIs in this
  slice instead of retaining a compatibility adapter. Rationale: the series
  intentionally moves callers to a typed `CompilerTarget`; a synthetic menu
  target would preserve the parallel configuration model this change removes.
  Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Use `CompilerTarget::SingleQubitBasis` directly throughout target,
  Euler, and Weyl code and delete `NativeSynthesisBasis`. Rationale: one enum
  and one synthesis-basis value remove a conversion switch, adapter files, and
  an otherwise redundant coverage test while preserving dependency direction.
  Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Measure pre-routing profitability by a strict reduction in two-qubit
  basis uses and materialize a canonical U/CZ sequence only after the comparison
  succeeds. Rationale: two-qubit operations drive routing cost, and selecting a
  hardware basis before routing would conflate optimization with target
  legality; CZ avoids introducing an arbitrary control direction. Date/Author:
  2026-08-03, GPT-5.6 via Codex.
- Decision: Let `CompilerTarget::supports` decide whether an ordinary `qco.swap`
  requires post-routing lowering. Rationale: the target is the sole capability
  authority; routing SWAPs still lower on ordinary targets that do not report
  SWAP, while a target-native SWAP must remain legal even when the target has no
  global synthesis basis. Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Treat operation capabilities as homogeneous across a target. Ordered
  site tuples retain calibration only; synthesis and conformance query canonical
  name, arity, and parameter count without directional fallback. Rationale:
  current target gate sets are uniform and bidirectional, so site tracing and
  reverse probes add code without changing compilation behavior. Date/Author:
  2026-08-03, GPT-5.6 via Codex.
- Decision: Preflight all target-lowering needs and apply planned rewrites
  directly with `IRRewriter`. Rationale: failure remains atomic and generic
  greedy/fixpoint work is unnecessary. Date/Author: 2026-08-03, GPT-5.6 via
  Codex.
- Decision: Do not require `CompilerTarget::synthesisBasis()` at pass
  construction or pass entry. Rationale: absent operations mean all operations
  are native, and an incomplete explicit target can still describe a conforming
  program. Missing-basis failure matters only after an unsupported operation
  actually needs lowering. Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: Keep gate fusion, target-native synthesis, and conformance as
  separate manual factories rather than textual passes. Rationale:
  `CompilerTarget` is an immutable typed C++ value that cannot be faithfully
  represented by generic pass options, and separate factories make each stage
  independently testable and benchmarkable. Date/Author: 2026-08-03, GPT-5.6 via
  Codex.
- Decision: Preserve Simon Hofmann's authorship in the implementation commit.
  Rationale: the implementation materially carries forward the progressive
  post-routing behavior and SWAP-removal coverage from his #1969 work while
  excluding the unapproved Python and CLI designs. Date/Author: 2026-08-03,
  GPT-5.6 via Codex.

## Outcomes & Retrospective

The implementation now has one homogeneous capability authority and three
separately observable transform stages. The latest cleanup removes the
decomposition basis adapter, general greedy rewrite machinery, directional
capability fallback, and per-operation target-site tracer. It retains explicit
CZ emission and failure-atomic lowering while adding direct static-site and
dynamic-allocation and quantum-function-input conformance coverage. Final
release builds pass 21 target-synthesis, 215 compiler, 33 dialect-utils, 27
mapping, and 199 decomposition tests. The SC device suite passes 41 tests with
one expected job-ID skip. Both affected interface-header targets build, all
repository hooks pass, focused LLVM 22.1.8 `clang-tidy` reports no new
diagnostics, and an independent review approves the exact working tree.

The main design lesson is that target support and fusion profitability must not
share a configuration surface. Canonical U/CZ gate fusion is useful without
hardware knowledge, while post-routing lowering and conformance require the
target's homogeneous operation set and declared static sites.

## Context and Orientation

`mlir/include/mlir/Compiler/Target.h` defines the immutable target. Its
`supports(Operation*)` query recognizes QCO operation semantics and checks
canonical name, arity, and parameter count. Ordered operation site tuples retain
calibration only. Its `synthesisBasis()` query returns one usable single-qubit
basis and entangler only when both exist.

`mlir/lib/Dialect/QCO/Transforms/NativeSynthesis/TargetSynthesis.cpp` contains
the two-qubit gate-fusion scanner, three pass implementations, static-site
validation, and diagnostics. Public factory declarations live in
`mlir/include/mlir/Dialect/QCO/Transforms/Passes.h`.

`mlir/include/mlir/Dialect/QCO/Transforms/Decomposition/Euler.h` aliases the
target-owned `SingleQubitBasis`; no decomposition-layer basis DTO remains.
`mlir/lib/Dialect/QCO/Transforms/Decomposition/Weyl.cpp` caches the selected
entangler decomposer, returns a prepared decomposition, and emits its
single-qubit factors without recomputing it.

QCO qubits use linear static single assignment: each operation consumes a qubit
value and returns its successor. After mapping, `qco.static` operations declare
the assigned target sites. Since operation capabilities are homogeneous,
conformance validates these declarations once and does not trace each operand's
SSA lineage.

Focused tests are in
`mlir/unittests/Dialect/QCO/Transforms/NativeSynthesis/test_target_synthesis.cpp`.
Typed basis coverage remains in
`mlir/unittests/Dialect/QCO/Transforms/Decomposition/test_weyl_decomposition.cpp`.

## Plan of Work

The implementation deletes the duplicate gateset and basis-adapter files and
rewires Euler and Weyl synthesis directly to the target-owned basis type. It
also removes the obsolete generated pass, registration, CLI option, `QCOProgram`
method, binding, stub, and their menu tests.

The new source retains the proven two-qubit gate-fusion scanner but separates
its uses. The fusion pass compares original and canonical entangler counts.
Target synthesis classifies actual mapped operations and rewrites only lowering
needs. Conformance performs an independent read-only walk and exact target
query.

The final work updates the existing Unreleased changelog entry without an
upgrade note, formats changed sources, runs focused and repository-required
validation, obtains independent read-only reviews, and publishes signed, focused
commits. The implementation commit preserves Simon Hofmann's `Co-authored-by`
trailer; later metadata, lint, and coverage follow-ups carry the required
`Assisted-by` trailer.

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

The current observed summaries are:

    [  PASSED  ] 21 tests.
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

- Two adjacent constant CX operations fuse away with equivalent unitary
  behavior, while a three-CX SWAP form, an isolated SWAP, and a runtime RXX run
  remain quantum-structurally unchanged before routing.
- Target-native synthesis removes unsupported ordinary SWAP and produces only
  operations accepted by the selected target basis, preserving the complete
  unitary. A target-native SWAP remains unchanged without requiring a synthesis
  basis.
- Homogeneous operation capabilities apply in both operand orientations while
  ordered site tuples retain calibration data only. Synthesis preserves complete
  unitary behavior without an operand-reversal option, and a native `qco.pow`
  shell is checked without separately rejecting its implementation body.
- An absent operation set succeeds without synthesis. An explicit incomplete
  target succeeds for supported operations and reports “no usable synthesis
  basis” only when an unsupported operation actually needs lowering.
- A supported runtime-parameterized gate remains unchanged. An unsupported
  runtime gate reports that its unitary matrix is unavailable at compile time
  without partially rewriting an earlier constant gate.
- Conformance accepts sparse target IDs, rejects operation-type, arity,
  parameter-count, unknown-site, measurement, and dynamic-allocation mismatches,
  and does not reconstruct per-operation site provenance.
- The duplicate enum/parser and all native-menu text, CLI, C++ program, Python,
  and generated pass surfaces are absent.
- Focused tests, decomposition tests, compiler tests, changed-file checks,
  repository lint, and `git diff --check` pass.
- Every commit is signed and carries the required `Assisted-by` trailer. The
  implementation commit additionally carries
  `Co-authored-by: Simon Hofmann <simon.t.hofmann@tum.de>`.

## Idempotence and Recovery

Configuration, builds, tests, and hooks are repeatable in the task-local build
and cache directories. A failed pass test uses a disposable module fixture.
Network fetch failures during initial configuration should be retried with
authorized network access; source behavior must not be weakened to accommodate
an environment boundary.

Do not reset or clean the worktree. Inspect status and remove only
task-generated artifacts if necessary. Do not modify another task's worktree or
shared branch metadata. External publication requires separate authorization;
draft PR #1998 was published only after that authorization was granted.

## Artifacts and Notes

Initial development used CT-01 commit
`b6eb95521cb76224c137496f113f38b9ce295854`. The SYN implementation patch is
restacked directly onto the squash-merged CT-01 base
`f775395a25fddba0a3b54996416d1311bc6ebe71`. The stable patch ID before and after
the restack is `fb73dd9d29254dfca2edf68df7eb9bf983284cc4`, proving that no SYN
implementation content changed during the history rewrite. Changelog,
changed-file lint, and coverage follow-ups remain separate focused commits.

Relevant #1969 source commits are `3be6d8e43` for target-native naming and
`1aa4c975b` for progressive post-routing synthesis and SWAP-removal coverage,
both authored by Simon Hofmann.

Independent read-only review initially blocked the candidate on three issues:
single-orientation symmetric-entangler site tuples, a SWAP capability override,
and nested `qco.pow` body checking. The exact remediated source rebuilt
successfully, all sixteen focused tests passed, and the reviewer reported that
all three findings were closed with no new code blockers. The reviewer also
reran `git diff --check` against the initial development base.

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

The coverage follow-up added regressions for interleaved two-qubit gate fusion,
constant single-qubit target lowering, and the former site tracer. The latest
simplification deletes tracer-only control-flow tests and adds direct
static-site and dynamic-allocation coverage alongside structural CZ emission and
failure-atomic lowering.

After MAP-01 merged as `c9e0c0ca5`, the four SYN commits were rebased onto that
commit. The integration range-diff removes only a compiler-target link already
provided by MAP-01, retains MAP-01's canonical undirected coupling input in the
compiler API test, and keeps both PR links in the changelog. The lint and
coverage follow-up patches remain identical. Fresh release builds pass 21 target
synthesis, 199 decomposition, and 215 compiler tests, including the focused
`QCOProgramOptimizationAPIs` test.

## Interfaces and Dependencies

`CompilerTarget` remains dependency-light and does not depend on transform
libraries. `MLIRQCOTransforms` publicly links `MQTCompilerTarget` because its
public decomposition headers use compiler-target basis types. The public,
independently constructible factories are:

    std::unique_ptr<Pass> createFuseTwoQubitGates();
    std::unique_ptr<Pass>
    createTargetNativeSynthesis(const CompilerTarget& target);
    std::unique_ptr<Pass>
    createVerifyTargetConformance(const CompilerTarget& target);

There is no textual target-native pass, synthetic target, native-gate menu, or
parallel capability model.

Revision note (2026-08-03): Updated the completed plan after PR publication, CI
feedback, and the MAP-01 merge. This records the focused coverage regressions,
current test and coverage evidence, multi-commit publication state, the
integration conflict resolutions, and the fact that Simon Hofmann's authorship
belongs to the implementation commit rather than every follow-up.
