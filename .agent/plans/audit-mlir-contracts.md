# Audit and harden MLIR contracts

This ExecPlan is a living document. The sections Progress, Surprises &
Discoveries, Decision Log, and Outcomes & Retrospective are updated while the
work proceeds. This plan is maintained in accordance with .agent/PLANS.md.

## Purpose / Big Picture

Complete GitHub issue #2255 as one reviewable change. Every MQT-owned MLIR pass,
handwritten verifier entrypoint, rewrite/conversion pattern, and dialect
registration surface is audited. Unsupported but structurally valid input must
produce a diagnostic rather than crash or partially mutate the IR; successful
passes must return verifier-valid and semantically faithful IR; verifiers must
own only local invariants; patterns must report match success and failure
truthfully; and every dialect a pass can create must be registered.

The observable result is a focused regression for each confirmed objective
violation, plus a green complete MLIR test label and repository lint checks.

## Progress

- [x] (2026-08-27 22:27Z) Read the repository guidance, MLIR development policy,
      issue #2255, and ExecPlan requirements; created an isolated worktree at
      origin/main commit baecdc55f130a26a21222d6fe5c613db7eee3633.
- [x] (2026-08-28 00:36Z) Completed the exhaustive inventory: 27 passes, 26
      verifier entrypoints, 235 rewrite/conversion patterns, and seven
      registration surfaces.
- [x] (2026-08-28 00:36Z) Reduced search candidates to reachable contract
  violations by constructing verifier-valid or locally accepted inputs.
- [x] (2026-08-28 05:00Z) Hardened pass anchors and registrations, preflighted
      unsupported input before mutation, made conversions atomic, repaired local
      verifier ownership, bounded recursive/resource-sensitive processing, and
      corrected rewrite, metadata, numeric, mapping, iterator, and lifetime
      cases.
- [x] (2026-08-28 05:20Z) Ran the focused suites for QC IR, QCO IR, QIR
      metadata, QIR Base, JeFF round-trip, QCO-to-QC, QC-to-QCO, mapping, QCO
      utilities, and optimizations; all were green before the final focused
      review.
- [x] (2026-08-28 05:45Z) Resolved the final focused-review findings:
      invalidated ancestor QTensor caches after nested stores, diagnosed
      duplicate-static aliases after lifetime end, and made captured-register
      lifetime failures diagnostic and pattern-atomic.
- [x] (2026-08-28 08:01Z) Completed the initial release build, all 3,075
      mqt-mlir-unittests, repository lint, diff checks, and changed-file audit.
      Standalone C++ lint cannot start on this host because clang-tidy 22 is not
      installed; Nox reports that exact environment limitation.
- [x] (2026-08-28 09:09Z) Closed the final QIR Base ordering interaction:
      preserved per-qubit irreversible ordering, kept independent operations
      legal, and made Mapping retain terminal measurement/reset suffixes after
      routing swaps. Rebuilt the release tree and passed all 3,086
      mqt-mlir-unittests.
- [x] (2026-08-28 10:45Z) Completed three independent boundary reviews and
      resolved their remaining findings: accepted LLVM function-interface entry
      points, preserved Qiskit's parameter-vector shrink semantics, routed every
      JeFF file/byte/CLI import through one bounded preflight, capped aggregate
      OpenQASM classical width, and enforced QCO linearity before DD
      construction.
- [x] (2026-08-28 11:00Z) Corrected JeFF entry-point metadata to use the
      schema-defined function index, contained recoverable jeff-mlir fatal
      reports, and covered framing, incomplete structures, declarations,
      undefined values, invalid reconstructed IR, and assertion-prone shapes
      through byte, file, and command-line imports.
- [x] (2026-08-28 11:05Z) Completed the pre-rebase release build, all 3,102
      mqt-mlir-unittests, five focused Python/Qiskit metadata regressions, the
      full repository lint session, and diff checks. Standalone C++ lint still
      cannot start because this host does not provide clang-tidy 22.
- [x] (2026-08-31) Rebased the audit onto origin/main commit
      35d3dc2cb87dc9ed4904e9db7eb43257ad3d4527 and re-audited the newly landed
      static-qubit canonicalizer, QCO DD sampling/runtime paths, and partial
      Layout/Mapping APIs. The refreshed census is 27 passes, 26 verifier
      entrypoints, 236 patterns (124 conversion and 112
      canonicalization/optimization), and seven registration surfaces.
- [x] (2026-08-31) Prevented HoistStaticQubit from moving values across a nested
      IsolatedFromAbove boundary, bounded QCO DD call and region nesting at 64
      with a shared 10,000-step analysis/execution budget, and enforced one
      program-wide identity per static-qubit index with no reacquisition after
      deallocation.
- [x] (2026-08-31) Completed the refreshed release build and all 3,133 tests in
      the mqt-mlir-unittests label.
- [x] (2026-08-31) Passed the six QCO DD Python regressions, five focused
      Python/Qiskit metadata regressions, repository lint, and diff checks.
      Reconfirmed that standalone C++ lint cannot start because clang-tidy 22 is
      unavailable on this host.

## Surprises & Discoveries

- Observation: several conversions diagnosed unsupported input only after
  mutating the source module. Clone/verify/commit or complete read-only
  preflight was necessary to make failure atomic.
- Observation: QIR aggregate metadata cannot count every pointer-bearing store.
  Only aggregate roots reachable from qubit-bearing QIS operands represent qubit
  capacity; unrelated result arrays otherwise inflate the count.
- Observation: QCO QTensor insert was previously treated as identity even when
  it changed a slot. Correct lowering requires a memref.store and conservative
  cache invalidation when aliasing cannot be disproved.
- Observation: current `main` defines each static-qubit index as one physical
  identity for the whole program. Repeated references are canonicalized to one
  entry-block root; any later use after deallocation, including a syntactic
  reacquisition of the same index, must fail cleanly and atomically.
- Observation: HoistStaticQubit may move a zero-operand static root to the
  function entry only when the function is its nearest IsolatedFromAbove
  ancestor. Crossing a nested isolation boundary makes the retained users
  capture a value from above and invalidates otherwise verified IR.
- Observation: loop-unroll verification of a temporary clone must retain its
  parent module so sibling symbol references resolve, while verifying only the
  transformed operation because the temporary module is intentionally partial.
- Observation: recursive walkers and expansion-producing passes needed explicit
  depth, iteration, or resource budgets. Deep but valid nested regions also
  required iterative traversal tests.
- Observation: QCO DD sampling performs a recursive interprocedural analysis
  before execution, so bounding only the runtime walker is insufficient. The
  sampling analysis and runtime now share the 64-call/region-nesting policy and
  a 10,000-step work budget.
- Observation: angle normalization documentation used the wrong half-open
  interval; known NaN/Inf expressions and integer size products required
  explicit checks before conversion or allocation arithmetic.
- Observation: MLIR canonicalizer maxIterations is a best-effort production
  bound. Non-convergence becomes a failure only under a test-only convergence
  option that GreedyRewriteConfig does not expose, so the production cleanup
  pass cannot truthfully promise a synthetic failure regression.
- Observation: region presence alone is not the QIR Base boundary. Region-based
  control flow is unsupported except qc.ctrl; preserved non-control region ops
  such as tensor.generate remain legal, including in helper functions.
- Observation: QIR Base's irreversible ordering is a per-qubit constraint, not a
  global operation-order constraint. Independent quantum operations and a
  zero-target global phase may follow a measurement or reset, while aliases of
  the same static qubit must still be rejected.
- Observation: Mapping's wire driver advances through terminal measurements and
  resets before routing. Inserting a swap at the old terminal iterator position
  put that swap after the irreversible operation; moving the terminal suffix
  across the swap through the existing SSA rewiring preserves logical state and
  keeps the irreversible operation terminal.
- Observation: module-wide MQT metadata validation must inspect
  FunctionOpInterface rather than only func.func. LLVM functions are a supported
  compiler checkpoint and must retain the same uniqueness and definition
  requirements; source-level QC/QCO transformation helpers remain intentionally
  typed to func.func.
- Observation: a Qiskit ParameterVectorElement may legally retain an index at or
  beyond the vector's current shrunken size. Group identity/name/size must be
  consistent, but the element index is not bounded by that current size.
- Observation: the JeFF schema defines entrypoint as a function-list index, but
  both MQT conversions had treated it as a string-table index. Programs with
  auxiliary strings exposed the mismatch only after a binary round trip.
- Observation: jeff-mlir treats several recoverable data errors as LLVM fatal
  errors and assumes particular operation/region arities before verification. A
  shared MQT preflight must therefore validate version, value indices, every
  instruction family, callable signatures, control-flow shapes, nesting, and
  aggregate sizes before invoking it; its remaining fatal reports must be
  translated back into diagnostics at the same boundary.
- Observation: OpenQASM output growth is governed by aggregate classical width,
  not only per-register widths. DD construction likewise needs QCO linearity
  verification in the shared preparation path, not only at selected callers.

## Decision Log

- Decision: complete all of #2255 in one PR, while keeping each regression and
  source fix narrow. Rationale: the user explicitly requested one comprehensive
  audit; separable tests keep the large scope reviewable. Date/Author:
  2026-08-27, Codex.
- Decision: reject unsupported semantic shapes during a read-only preflight
  unless the target dialect can preserve them. Rationale: a diagnostic is safer
  than a partial or silently lossy conversion and does not expand this audit
  into feature work. Date/Author: 2026-08-28, Codex.
- Decision: use clone/verify/commit where a conversion has many independently
  fallible stages and a complete preflight would duplicate the lowering.
  Rationale: it provides module atomicity with existing MLIR APIs and no new
  framework. Date/Author: 2026-08-28, Codex.
- Decision: move non-local MQT metadata invariants out of dialect callbacks and
  into the explicit program validator. Rationale: operation verification must
  not depend on unrelated siblings, while compiler boundaries still enforce
  whole-program uniqueness. Date/Author: 2026-08-28, Codex.
- Decision: define the QIR Base region boundary with RegionBranchOpInterface,
  exempt qc.ctrl, and inspect convertible QC/CBit/MemRef operations nested in
  entry-function non-control containers. Rationale: this rejects unsupported
  helper/entry control flow without rejecting preserved tensor regions.
  Date/Author: 2026-08-28, Codex.
- Decision: enforce QIR Base ordering by canonical qubit identity and repair the
  mapper's insertion point for terminal irreversible suffixes. Rationale: this
  retains the profile contract without rejecting valid commuting operations or
  lowering target-mapped programs into post-measurement gates. Date/Author:
  2026-08-28, Codex.
- Decision: use one JeFF byte deserializer for the typed API, file API, and CLI;
  preflight its complete supported 0.3.0 shape and compile only the third-party
  translation object with a narrow fatal-to-exception redirect caught by that
  boundary. Rationale: jeff-mlir's API otherwise terminates the process after
  semantic or MLIR verification failures, while duplicating its full lowering
  would create a second implementation. Date/Author: 2026-08-28, Codex.
- Decision: cap emitted OpenQASM classical storage at 1,048,576 aggregate bits
  and run verifyLinearity in the common DD preparation helper. Rationale: both
  checks belong at their shared resource-owning boundary and cover every public
  caller without duplicated policy. Date/Author: 2026-08-28, Codex.
- Decision: hoist qc.static only within its nearest isolation scope and treat a
  static index as one program-wide physical identity that cannot be reacquired
  after deallocation. Rationale: canonicalization must preserve SSA isolation,
  while conversion and runtime lifetime checks must agree on static-qubit
  identity. Date/Author: 2026-08-31, Codex.
- Decision: apply the same finite limits to QCO DD sampling analysis and
  execution: 64 nested calls/regions and 10,000 shared work/steps. Rationale:
  both phases traverse user-controlled interprocedural control flow and must
  fail or conservatively select dynamic sampling before exhausting host
  resources. Date/Author: 2026-08-31, Codex.

## Outcomes & Retrospective

The refreshed census and implementation are complete: 27 pass implementations,
26 handwritten verifier entrypoints, 236 patterns (124 conversion and 112
canonicalization/optimization), and seven registration surfaces. Confirmed
defects covered valid-input crashes, partial mutation, invalid or lossy
successful output, non-local verification, undeclared dialects,
recursion/resource exhaustion, false rewrite results, isolation-breaking motion,
missed folded constants, numeric overflow/non-finite values, and duplicated or
stale conversion state.

After rebasing onto origin/main at 35d3dc2cb87dc9ed4904e9db7eb43257ad3d4527, the
complete release build and all 3,133 tests in the mqt-mlir-unittests label pass.
Current focused checks are green at QC IR 347/347, QC-to-QCO 176/176, QCO-to-QC
147/147, QCO utilities 149/149, and JeFF round-trip 152/152. The six QCO DD
Python 3.13 tests, five focused Python/Qiskit parameter-vector tests, repository
lint, and git diff checks also pass. The only unavailable check is standalone
C++ lint: `uvx nox -s cpp-lint` aborts before analysis with
`clang-tidy 22 is required` because that binary is absent from the host.
Pre-rebase focused-suite counts remain recorded below as historical checkpoints
rather than current per-suite totals.

## Context and Orientation

Pass declarations live below mlir/include/mlir in TableGen files and their
implementations below mlir/lib. Operation and attribute definitions live below
mlir/include/mlir/Dialect, handwritten verifiers and canonicalizers below
mlir/lib/Dialect, conversions below mlir/lib/Conversion, and direct GoogleTest
coverage in mirrored paths below mlir/unittests.

The 27-pass census is: CBit-to-MemRef, JeFF-to-QCO, QCO-to-JeFF, QCO-to-QC,
QC-to-QCO, QIR Adaptive, QIR Base, NormalizeGlobalPhases, UnrollModifiers,
ShrinkQubitRegisters, DecomposeMultiControlled, Mapping,
FuseSingleQubitUnitaryRuns, FuseTwoQubitUnitaryRuns, TargetNativeGates,
VerifyTargetGates, HadamardLifting, MeasurementLifting,
MergeSingleQubitRotationGates, PauliTwirling, QuantumLoopUnroll,
RemoveDeadGates, ReplaceClassicalControls, ReuseQubits, QIR attribute
attachment, QIR cleanup, and QTensor shrinking.

The 26 verifier entrypoints comprise 19 operation verifiers plus CBit
RegisterType, the QC and QCO unitary interfaces, three MQT dialect attribute
callbacks, and the program validator. The 236-pattern census comprises 124
conversion patterns and 112 canonicalization/optimization patterns. Conversion
distribution is CBit-to-MemRef 3, JeFF-to-QCO 25, QCO-to-JeFF 27, QCO-to-QC 23,
QC-to-QCO 22, QIR Adaptive 10, QIR Base 8, and QIR Common 6.

The seven registration surfaces are the five dialect initializers for CBit, MQT,
QC, QCO, and QTensor, plus the program compiler registry in Programs.cpp and
command-line compiler registry in mqt-cc.cpp. The audit checks both dialect
initialization and every pass getDependentDialects declaration.

The normative contract rules are in docs/mlir/development.md. The audit adds no
lit/FileCheck infrastructure, runtime dependency, public API migration, or
style-only cleanup.

## Milestones

### Milestone 1: Complete and disposition the census

Inventory every pass root and dependency declaration, verifier entrypoint,
pattern implementation, and registration site. Search for fatal errors,
assertions, unchecked casts/indexing, recursive walks, hand-matched constants,
and fallible work after mutation. The milestone is complete when every item is
recorded as clean, affected directly, or affected through a shared fix.

### Milestone 2: Prove objective violations

For each candidate, construct the smallest verified or locally accepted IR that
reaches it. Discard structurally unreachable findings. Keep valid-input crashes,
invalid/lossy successful output, non-local verifier assumptions, partial
mutation, false rewrite contracts, missing registrations, and unbounded work.
The milestone is complete when each retained case has a failing baseline
reproducer and a named ownership boundary for the fix.

### Milestone 3: Harden contracts with focused regressions

Put validation before the first mutation or lower a clone and commit only after
verification. Replace process termination with operation/pass diagnostics, make
verifier checks local, register produced dialects, bound recursive or expanding
work, and use MLIR constant matching rather than producer-specific casts. Add
direct GoogleTests beside existing coverage. The milestone is complete when
every retained case passes and successful output verifies.

### Milestone 4: Whole-suite closure and PR handoff

Build the final tree, run all mqt-mlir-unittests, run both lint sessions and
diff checks, then inspect status/name/stat output for unrelated or generated
files. Record exact results here. The milestone is complete when the branch is
PR-ready and no known #2255 contract defect remains.

## Plan of Work

Maintain the complete census while inspecting implementation and declaration
pairs. Validate candidates with direct tests rather than changing code from
search results alone. Apply the smallest fix at the owning boundary and keep
failure atomic. Re-run the narrow binary immediately after each cluster, then
perform a second read-only review of high-risk conversion state and semantic
preflights. Finish with the full labeled suite, lint, and diff audit.

## Concrete Steps

Run all commands from the repository root.

    rg -n 'def .*: Pass|runOnOperation|getDependentDialects' mlir/include mlir/lib
    rg -n 'LogicalResult .*verify|::verify\(' mlir/lib/Dialect
    rg -n 'RewritePattern|ConversionPattern|matchAndRewrite' mlir/lib
    rg -n 'assert\(|reportFatal|llvm_unreachable|\.front\(\)|cast<' mlir/lib

Configure and build:

    cmake --preset release
    cmake --build build/release --parallel 8

Run the complete checks after focused suites are green:

    ctest --test-dir build/release -L mqt-mlir-unittests --output-on-failure --parallel 8
    uvx nox -s cpp-lint
    uvx nox -s lint
    git diff --check

## Validation and Acceptance

Acceptance requires a named, fully dispositioned census; a regression for every
confirmed objective violation; no process termination or partial mutation for
unsupported valid input; verifier-valid successful output; truthful pattern
results; complete dialect registration; bounded recursion/resource use; and a
green full mqt-mlir-unittests label. Lint results and any environment-only
limitation must be recorded exactly. No generated, unrelated, or style-only file
may remain in the diff.

## Idempotence and Recovery

Inventory searches and tests are repeatable. Source edits use focused patches.
If a test exposes partial mutation, move validation ahead of the first mutation
or use clone/verify/commit rather than rollback. The isolated worktree protects
the user's existing checkout. If Ninja reports a corrupt log after interruption,
run ninja -C build/release -t recompact and rebuild; do not clean unrelated
state.

## Artifacts and Notes

Initial baseline: baecdc55f130a26a21222d6fe5c613db7eee3633 from origin/main.

Refreshed base: 35d3dc2cb87dc9ed4904e9db7eb43257ad3d4527 from origin/main.

Pre-rebase focused checkpoint:

    QC IR:             345/345
    QCO IR:            510/510
    QIR metadata:      133/133
    QIR Base:          144/144
    QIR Adaptive:      159/159
    JeFF round-trip:   151/151
    QCO-to-QC:         147/147
    QC-to-QCO:         176/176
    Mapping:            97/97
    QCO utilities:     121/121
    Optimizations:     219/219
    Decomposition and native synthesis: 242/242
    QTensor transforms:   3/3

Latest full closure transcript:

    cmake --build build/release --parallel 8
    # completed successfully

    ctest --test-dir build/release -L mqt-mlir-unittests --output-on-failure --parallel 8
    100% tests passed, 0 tests failed out of 3133

    uvx nox -s tests-3.13 -- test/python/test_qco_dd.py
    6 passed

    uvx nox -s tests-3.13 -- test/python/test_mlir_qiskit_translation.py -k \
      'parameter_vector_element_outside_current_size_round_trips or parameter_vector_metadata_is_preflighted'
    5 passed

    uvx nox -s lint
    nox > Session lint was successful

    uvx nox -s cpp-lint
    nox > Session cpp-lint aborted: clang-tidy 22 is required.

    git diff --check
    # no output

## Interfaces and Dependencies

Use existing typed TableGen pass anchors, signalPassFailure, operation
diagnostics, dialect registries, RegionBranchOpInterface, m_Constant,
GreedyRewriteConfig, and direct GoogleTest/CTest targets. Add no runtime or test
dependency. The Python API is unchanged. C++ API additions are limited to shared
helper declarations; no existing public API is removed or changed.

Revision note: expanded the initial QIR Base investigation into the requested
complete #2255 contract audit; replaced the stale mid-audit disposition with the
full pass, verifier, pattern, and registration census; the final review also
closed the QIR Base/Mapping irreversible-ordering interaction. The origin/main
refresh added one canonicalization pattern to the census, closed
HoistStaticQubit's isolation boundary, aligned program-wide static-qubit
identity and DD analysis/runtime bounds, and refreshed the full-suite evidence.
