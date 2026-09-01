# Audit MLIR contracts

This ExecPlan is a living document. The sections Progress, Surprises &
Discoveries, Decision Log, and Outcomes & Retrospective are updated while the
work proceeds. This plan is maintained in accordance with .agent/PLANS.md.

## Purpose / Big Picture

Complete GitHub issue #2255 as a dispositioned audit, not as one bulk code
change. Every MQT-owned MLIR pass, handwritten verifier entrypoint,
rewrite/conversion pattern, and dialect registration surface is audited. A
dialect-IR finding is retained only when the input passes every applicable
validator and a baseline reproducer shows a crash, wrong output, partial
mutation, unbounded resource use, or another stated contract violation.

Applicable validation includes MLIR structural verification and MQT-owned
whole-program checks such as QCO linearity and MQT program metadata. Individual
passes may assume those invariants. External input, verifier, conversion, and
shared resource-owning boundaries remain responsible for the checks they own.

The observable result is a complete census with each finding accepted, narrowed,
rejected, or deferred. Accepted findings land only in focused pull requests
after human review, with a regression that demonstrates the supported contract.

## Revised Audit Rules

- Identify the owner of each invariant before proposing a fix. Dialect verifiers
  own dialect validity; ingestion and conversion boundaries own malformed
  external input; passes own only failures reachable from valid IR.
- Require an executed baseline reproducer. A hypothetical stack, crash, or
  compatibility concern is not a defect without evidence that the supported path
  can reach it.
- Verify pass inputs and outputs in tests. Do not add whole-IR verification or
  one-use guards to each pass solely to tolerate invalid dialect IR.
- Keep invalid-IR tests at the verifier or ingestion boundary that owns the
  invariant. Do not duplicate them across transformation passes.
- Follow the repository rule that recursion must be bounded. For an established
  MLIR walk, require evidence that a proposed replacement enforces a documented
  bound or fixes a supported-path failure; an arbitrary-depth passing test does
  not establish that the replacement improves the contract.
- Prefer an upstream dependency fix when the defect belongs upstream. Do not add
  a local workaround until an observable MQT failure justifies it.
- Stop the audit at the finding ledger. A human selects findings for separate,
  focused implementation pull requests.

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
- [x] (2026-09-01) Reconciled the audit with the first 16 focused replacement
      pull requests: ten merged, two open, and four closed without merge.
- [x] (2026-09-01) Narrowed `#2300` to its demonstrated missing-entry-point
      defect and removed the speculative program-sized traversal worklists.
- [x] (2026-09-01) Withdrew `#2303`, `#2305`, `#2306`, and `#2309` after review.
      The `#2309` regressions use invalid QCO IR.
- [x] (2026-09-01) Revised the audit rules so passes may assume input accepted
      by all owning verifiers, and so each retained finding requires a
      valid-input baseline reproducer at the correct ownership boundary.
- [x] (2026-09-01) Reclassified the same invalid-input pattern across the full
      snapshot: generic pass-entry metadata/linearity checks, defensive QCO
      one-use guards, and their invalid-IR regressions are not actionable
      findings.
- [x] (2026-09-01) Merged current `main` at
      6b95f1c434a319fb66e946309e17df771a4e3be2, including the accepted fixes
      from `#2300`, `#2307`, and `#2308`, and reclassified every residual file
      and test hunk against the revised invariant.
- [x] (2026-09-01) Pruned merged and focused duplicates, closed findings,
      malformed-IR defenses, generic pass-entry validation, the unsupported
      shared depth policy, and unproved traversal replacements. The remaining
      implementation snapshot contains only valid, uniquely retained findings.
- [x] (2026-09-01) Independently cross-reviewed the retained compiler,
      conversion, dialect, pass, and QIR slices. Repaired one nonlinear matrix
      fixture, removed malformed QIR ABI defenses, corrected recursive QIR
      provenance, and cleared the surviving changed-line lint findings.
- [x] (2026-09-01) Removed the final unsupported findings: a QC register-alias
      proof that rejected valid benchmark programs and a registration-only
      assertion without an executed failure case.
- [x] (2026-09-01) Reduced the latest-main delta from the historical 118 files
      and 14,011 changed lines to 79 files and 11,169 changed lines. The 2,842
      removed lines are merged/focused duplicates, closed findings, invalid-IR
      handling, speculative traversal/depth policies, and their tests.
- [x] (2026-09-01) Built the reconciled branch with LLVM/MLIR 23.1.0 and passed
      all 3,111 tests in the `mqt-mlir-unittests` label, repository lint, and
      diff checks. Standalone C++ lint cannot start because this host lacks
      clang-tidy 22. The focused Python metadata session builds and reaches the
      expected diagnostics, but this Apple host cannot translate their C++
      exceptions through nanobind.

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
- Observation: failure to prove two dynamic QC register indices distinct does
  not prove they alias. Rejecting that case broke
  `GeneratesEveryBenchmarkMethodAsQCAndJeff`; retain only the established checks
  for operands known to be identical.
- Observation: expansion-producing work reachable from valid input can require
  explicit limits, and repository policy treats unbounded recursion as a
  correctness risk. The proposed `#2300` worklists increased memory use, while
  the `#2306` depth-256 regression also passed the old MLIR walk. Neither change
  demonstrated a better bound for a supported path.
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
- Observation: MLIR structural verification does not enforce QCO or QTensor
  linearity. Several original tests therefore called structurally verified IR
  valid even though the owning QCO validator rejected it.
- Observation: the QTensor one-use guard rejected during `#2295` review was
  proposed again in `#2303`. A focused extraction must check earlier review
  decisions before treating residual snapshot code as a new finding.
- Observation: the macOS exception workaround in `#2305` had no demonstrated
  user-visible failure. Dependency-boundary concerns should remain upstream
  candidates until an MQT reproducer exists.

## Decision Log

- Superseded decision: complete all of #2255 in one PR, while keeping each
  regression and source fix narrow. Rationale: the user explicitly requested one
  comprehensive audit; separable tests were expected to keep the large scope
  reviewable. Superseded by focused delivery on 2026-09-01. Date/Author:
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
- Decision: supersede the original one-pull-request delivery. The audit records
  findings; each accepted finding is reviewed and implemented separately.
  Rationale: focused review accepted useful fixes and exposed speculative or
  wrongly owned changes that a bulk implementation hid. Date/Author: 2026-09-01,
  Codex.
- Decision: treat input accepted by all applicable dialect and program
  validators as the precondition for transformation passes. Do not add
  pass-local verification or invalid-IR regressions for those same invariants.
  Rationale: QCO and QTensor validity have owning validators, and duplicating
  their checks across passes creates code and tests for unsupported programs.
  Date/Author: 2026-09-01, Codex.
- Decision: require a demonstrated supported-path failure or a documented,
  measurable bound improvement before replacing native MLIR traversal. Require
  an observable MQT failure before adding a local dependency workaround.
  Rationale: `#2305` and `#2306` did not reproduce an MQT defect, while the
  original `#2300` worklists increased memory use without enforcing a stated
  limit. Date/Author: 2026-09-01, Codex.

## Outcomes & Retrospective

The census is complete: 27 pass implementations, 26 handwritten verifier
entrypoints, 236 patterns (124 conversion and 112
canonicalization/optimization), and seven registration surfaces. The original
implementation snapshot is not an accepted set of fixes. It contains useful
findings, already merged findings, and changes later rejected as speculative or
wrongly owned. The historical commits preserve that snapshot. The current branch
removes rejected and separately owned code and keeps only valid residual
findings for review; it is still not intended to merge in bulk.

As of 2026-09-01, ten focused replacements have merged: `#2291`, `#2293`,
`#2294`, `#2295`, `#2296`, `#2300`, `#2301`, `#2304`, `#2307`, and `#2308`. Two
remain open: `#2290` and `#2302`. Four closed without merge: `#2303`, `#2305`,
`#2306`, and `#2309`. Review narrowed `#2290` to constant-like folding and
`#2300` to the demonstrated missing-entry-point case. Review rejected `#2309`
because its pass-local check and regressions target invalid QCO IR.

The original branch passed its recorded build, test, lint, and diff checks.
Those results prove internal consistency only; they do not establish that each
change protects a supported contract. The review dispositions and revised audit
rules now control which findings remain actionable.

## Focused Finding Reconciliation

Merged findings:

- `#2291`: preserve static-qubit isolation during cleanup.
- `#2293`: make MLIR region moves failure-atomic.
- `#2294`: preserve attributes on reused QIR declarations.
- `#2295`: make QTensor shrinking sparse and atomic. Review removed the
  redundant one-use guard because QTensor linearity owns that invariant.
- `#2296`: stop QCO wire traversal at unknown carriers.
- `#2300`: handle gate counts without an entry point. Review removed manual
  program-sized worklists and restored native MLIR traversal.
- `#2301`: keep terminal measurements after routing swaps.
- `#2304`: bound OpenQASM export resource use.
- `#2307`: bound CBit zero-initialization lowering.
- `#2308`: preserve QTensor insert updates in QCO-to-QC.

Open findings:

- `#2290`: harden MLIR constant folding.
- `#2302`: make QIR metadata attachment idempotent.

Closed findings:

- `#2303`: redundant QTensor one-use handling for invalid IR.
- `#2305`: no demonstrated MQT failure justified a local nanobind workaround;
  revisit upstream if a concrete failure appears.
- `#2306`: the proposed depth-256 regression also passed the existing MLIR walk
  and did not demonstrate a supported bound. Revisit recursion policy in one
  shared change if native MLIR walks need an explicit repository-wide limit.
- `#2309`: its tests use invalid QCO IR. Hadamard lifting may assume QCO
  linearity, so the pass-local check and regressions were rejected.

Broader snapshot changes withdrawn by the revised invariant:

- Drop generic metadata or QCO-linearity input checks added to
  `NormalizeGlobalPhases`, `UnrollModifiers`, QCO-to-QC, QCO-to-jeff, Mapping,
  `FuseSingleQubitUnitaryRuns`, `TargetSynthesis`, `HadamardLifting`,
  `MeasurementLifting`, `MergeSingleQubitRotationGates`, and QIR Common. Keep
  target-specific support and representability checks in those components.
- Drop branch-only defensive one-use fallbacks in `QCOUtils.h`, QCO barrier and
  rotation canonicalization, Mapping, two-qubit target synthesis, Hadamard
  lifting, and classical-control replacement. Their zero-use or multi-use inputs
  violate QCO linearity.
- Drop the matching pass and pattern regressions that construct nonlinear QCO or
  QTensor programs. This includes the unused-output canonicalization, synthesis,
  Hadamard-lifting, and classical-control tests, plus nonlinear-input tests for
  modifier unrolling, normalization, mapping, measurement lifting, single-qubit
  fusion, rotation merging, and QCO conversions.
- Drop the `NormalizeGlobalPhases` TableGen promise that the pass diagnoses QCO
  linearity. The owning validator, not this pass, defines that contract.
- Do not count duplicate program-entry rejection in QCO-to-jeff or QIR Common as
  a conversion finding. Keep target-specific entry existence and shape checks.
  The exactly-one-QIR-entry rule in `#2302` remains at its owning QIR metadata
  boundary.
- Repair the QCO-to-jeff mixed-allocation reproducer before extracting that
  valid target limitation. Its current fixture leaves quantum values unused; the
  corrected fixture must sink them and pass QCO linearity first.

The revised invariant retains owning and output-side checks. Keep
`mqt::verifyProgramMetadata`, `qco::verifyLinearity`, compiler and external
ingestion boundaries, resource and nesting limits reachable from valid IR,
target-representability preflights, and clone/lower/verify/commit checks that
validate newly produced IR before mutation is committed. Pass tests may call the
applicable validators before and after the pass without adding those checks to
each pass implementation.

## Complete Residual Snapshot Disposition

The original implementation is preserved at
`0141a0b4f8bbf63608f74fdd5b8608e2f2c40e95`. The last pre-reconciliation snapshot
is `9fcc02eb67586628d8244b425d032630313a86f2`. These commits preserve every
removed implementation and reproducer. The current delta keeps code and tests
only for the valid findings listed below. All residual production, test,
TableGen, CMake, binding, documentation, and tool changes were reread against
current `main` at `6b95f1c434a319fb66e946309e17df771a4e3be2`.

The retained items below are findings, not approval of the historical
implementation. Each needs a focused extraction from current `main`, a minimal
valid-input regression, and its own review.

### Compiler and external-input boundaries

- Retain compiler-owned program metadata validation at QC/QCO import and at
  public pipeline input and output. The historical evidence is
  `EnforcesProgramMetadataAtImportAndPassBoundaries`, the `mqt-cc`
  duplicate-entry case, and `test_parameter_vector_metadata_is_preflighted`. The
  compiler is the owner; individual passes are not.
- Retain the finding that malformed serialized JeFF input must not terminate the
  process. `JeffFunctionDeclarationsAreRejected`,
  `MalformedJeffStructuresAreRejected`, and
  `InvalidJeffSemanticsAreRejectedWithoutExiting` exercise byte, file, and
  command-line imports. Any extraction must minimize the local preflight and
  fatal-error bridge and prefer an upstream fix where possible.
- Keep `#2300` as the sole owner of `QCProgramCountGatesWithoutEntryPoint`. The
  manual compiler worklists are not part of that finding.

### Conversion boundaries

- JeFF-to-QCO retains schema-correct function-index entry-point decoding, target
  entry-shape validation, status-result restoration, and
  clone/lower/output-verify/commit atomicity. Evidence:
  `RejectsMalformedJeffMetadataBeforeMutation`,
  `RejectsUnsupportedQCOEntryShapeBeforeMutation`,
  `RejectsStaleEntryPointMetadataAfterJeffConversion`,
  `RestoresStatusResultAtEndOfEntryPoint`, and the no-mutation assertion in
  `RejectsClassicalIfResultsPrecisely`.
- QCO-to-JeFF retains target-specific entry shape, allocation-mode, rank-one
  tensor, and 255-control limits, schema-correct function-index serialization,
  and output atomicity. The mixed-allocation fixture must be repaired to sink
  every quantum value before extraction. Drop the duplicate-static fixture: the
  owning QCO validator already rejects it.
- QCO-to-QC retains the valid target limitations demonstrated by
  `RejectsYieldPermutationWithoutMutation` and
  `RejectsMixedAllocationModesWithoutMutation`, plus output atomicity. `#2308`
  exclusively owns `PreservesQTensorInsertSlotUpdates` and
  `InvalidatesQTensorCacheAcrossLoopSlotSwap`.
- QC-to-QCO retains mixed allocation, static reference identity and lifetime,
  conditional captured-register lifetime, ordered QTensor materialization,
  fallible live-value lookup, and output linearity/atomicity. Evidence:
  `DuplicateStaticReferencesShareOneEvolvingQCOValue`,
  `RejectsStaticUseAfterDeallocationWithoutMutation`,
  `RejectsMixedAllocationModesWithoutMutation`, and
  `RejectsConditionallyDeallocatedCapturedRegisterWithoutMutation`. Unknown
  dynamic register-index relationships remain accepted; only operands known to
  be identical are rejected by the established conversion checks.
- QIR Adaptive retains entry shape and reserved-symbol checks, entry-only
  QC/CBit/MemRef restrictions, the aggregate classical-result budget, release
  placement, one-epilogue return lowering, supported control-flow and allocation
  limits, global-phase lowering, and atomic output verification. The historical
  tests cover missing entries, `__quantum__` collisions,
  static/dynamic/conditional/repeated releases, nested and inconsistent returns,
  path-dependent classical output, helper-function QC, mixed allocation,
  controlled global phase, non-hoistable phase, rank-zero loads, and excessive
  classical capacity.
- QIR Base retains entry and resource-shape validation, supported control-flow
  and MemRef restrictions, static-ID/register-element accounting, reset and
  irreversible-order rules, and atomic output verification. The historical tests
  cover same-qubit and aliased-static ordering, independent qubits and global
  phase, sparse and exhausted static IDs, repeated/dynamic/rank-zero loads,
  multi-block/backedge/structured/affine/helper control flow, preserved
  non-control regions, entry arguments, and excessive classical capacity.

### Owning verifiers and metadata

- Retain non-local program metadata checks only in `mqt::verifyProgramMetadata`.
  Evidence: `AcceptsDefinedLLVMEntryPoint`,
  `ProgramMetadataRejectsDuplicateEntryPoints`,
  `ProgramMetadataRejectsNonFuncEntryPoint`,
  `ProgramMetadataRejectsDuplicateInputNames`,
  `ProgramMetadataRejectsInconsistentParameterGroups`, and
  `AcceptsParameterGroupsOutsideCurrentVectorSize`.
- Retain owning QC/QCO operation-verifier findings for yield parentage, modifier
  operand/result arity, duplicate yields, captured quantum values, barrier
  arity, and direct or folded non-finite phase angles. Keep only the smallest
  verifier-level assertion for each invariant when extracted.
- Retain bounded, failure-returning matrix queries and their semantic
  correctness findings: pass-through targets, yield permutations, a sole
  wider-than-two-qubit operation, unknown nested unitaries, excessive control
  width, and finite large integral powers. Drop the arbitrary 64-modifier-depth
  policy.
- Retain finite and bounded angle arithmetic at the owning numeric helpers and
  valid-output transformations. Historical evidence includes
  `IntegralPowerExtractionDoesNotOverflowGlobalPhase`,
  `DynamicPhasesUseBoundedRuntimeArithmetic`,
  `HugeFiniteStaticPhasesProduceVerifiedOutput`,
  `DynamicMaxPhasesRemainFiniteAfterFusion`, and the QC/QCO verifier tests for
  non-finite phase angles.
- `#2302` owns exactly-one-entry and idempotent passthrough/module-flag
  metadata. Separate retained metadata findings cover static-resource provenance
  and capacity, malformed or unknown origins, sparse qubit/result capacities,
  all `inttoptr` users, result arrays that must not inflate qubit capacity,
  malformed record declarations, and CFG-sensitive feature classification.
- QIR cleanup retains nested runtime-call discovery, deduplicated required
  resource counts, and proven side-effect-free array alloc/release matching.
  Evidence: `CleanupFindsRuntimeCallsNestedInFunctions`,
  `CleanupDoesNotDuplicateRequiredResourceCounts`, and
  `CleanupOnlyRemovesProvenSideEffectFreeArrayPairs`.

### Valid transformation and resource findings

- NormalizeGlobalPhases retains terminatorless module blocks, independent QC/QCO
  aggregation, impure extraction boundaries, and finite static/dynamic angle
  arithmetic. Drop its pass-local program and linearity validation and
  iterative-traversal claim. The dedicated
  `.agent/audits/global-phase-normalization.md` remains the assertion-level
  audit for that pass.
- Modifier canonicalization and decomposition retain classical support
  operations or refuse a rewrite before mutation. Evidence spans the QC/QCO
  modifier canonicalizer tests, `PreservesClassicalBodyCalls`,
  `LeavesPostUnitaryClassicalBodyCallInPlace`,
  `PreservesModifierSupportOperationsWhenRefusingLift`,
  `AllMeasuredFastPathsPreserveClassicalBodyCalls`, and the two
  `PartialMeasured...RefusesUnsafeSupportingOpHoist` tests.
- Mapping retains option and target validation, target-specific representability
  checks, the classical-only no-op, clone/verify/commit atomicity, its own
  bounded structured recursion, and terminal reset routing. `#2301` already owns
  terminal measurement routing. Drop generic metadata/linearity validation and
  nonlinear-input tests.
- Target synthesis retains support-call preservation and atomic refusal. Drop
  unused-output handling because those fixtures violate QCO linearity.
- Quantum-loop unrolling retains the factor 4096 and projected-operation 100,000
  resource limits, identity-yield correctness, required Arith dependency,
  sibling-symbol-safe verification, and clone/verify/commit atomicity. Evidence
  is the `Excessive...`, `NestedExpansion...`, `PreservesYieldOnlyPermutation`,
  `DynamicTripCountFailureIsAtomic`, and
  `UnrollsFunctionWithSiblingSymbolReference` tests. Drop the generic
  64-region-depth guard.
- DD execution retains its public-boundary aggregate classical-bit cap, shared
  10,000-step sampling/execution budget, and explicit call/region nesting
  limits. Evidence: `RejectsExcessiveClassicalRegisterCapacity`,
  `RejectsExcessiveRegionNesting`, and `RejectsExcessiveFuncCallNesting`.
- The JeFF deserializer's own depth and aggregate-size limits remain valid
  because serialized bytes are untrusted external input. This does not justify
  the same 64-region precheck in each transformation pass.

### Changes withdrawn from every area

- Remove pass-entry `verifyProgramMetadata`, `verifyLinearity`, and generic
  whole-module verification when they only repeat the declared input invariant.
  Retain output verification and checks at compiler, external-input, or
  target-representability boundaries.
- Remove defensive `hasOneUse()` branches added solely for zero-use or multi-use
  QCO/QTensor programs. The owning linearity validator rejects those programs
  before a pass runs.
- Remove every matching invalid-input regression, including nonlinear QCO
  conversion, normalization, unrolling, mapping, synthesis, lifting, merging,
  and classical-control fixtures; duplicate-static and static-reacquisition
  QCO-to-QC fixtures; unused-output gate and barrier canonicalization fixtures;
  and all additions from closed `#2303` and `#2309`.
- Remove the shared `verifyRegionNestingDepth(..., 64)` helper, pass-entry
  calls, TableGen promises, dependencies, documentation, and deep-region tests.
  Keep only bounds owned by an actual recursive resource boundary.
- Remove manual worklists that merely replace native MLIR walks. `#2300` and
  `#2306` established no supported-path failure or better finite bound.
- Remove registration-only assertions and dependency additions unless an
  executed reproducer demonstrates an unloaded-dialect failure.
- Remove branch copies of all merged or focused findings. Their implementation
  and regression live in `#2290`, `#2291`, `#2293`, `#2294`, `#2295`, `#2296`,
  `#2300`, `#2301`, `#2302`, `#2304`, `#2307`, and `#2308`. Remove all code from
  closed `#2303`, `#2305`, `#2306`, and `#2309`.

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

For each candidate, construct the smallest input accepted by MLIR structural
verification and every applicable MQT-owned validator. Discard invalid-dialect
inputs and structurally unreachable findings. Keep valid-input crashes,
invalid/lossy successful output, non-local verifier assumptions, partial
mutation, false rewrite contracts, missing registrations, and demonstrated
unbounded work. The milestone is complete when each retained case has an
executed failing baseline reproducer and a named ownership boundary.

### Milestone 3: Review and extract accepted findings

Record the evidence, owner, risk, and minimal proposed remedy for each retained
finding. Stop the audit before implementation. After human selection, implement
one finding per focused pull request and add the smallest regression that fails
on the baseline and protects the supported contract. The milestone is complete
when every finding is accepted, narrowed, rejected, or deferred.

### Milestone 4: Reconcile focused outcomes

Update this ledger after each focused review or merge. Record accepted,
narrowed, rejected, deferred, and superseded findings. Do not infer that a green
bulk branch validates every proposed contract. The milestone is complete when
the ledger matches GitHub and no rejected snapshot change remains listed as an
actionable defect.

## Plan of Work

Maintain the complete census while inspecting implementation and declaration
pairs. For every candidate, identify the owning contract and run the applicable
validators before writing a reproducer. Record and discard candidates that need
invalid dialect IR. Require evidence before changing native traversal or adding
a dependency workaround. Stop at the ledger. For a human-selected finding, start
a focused branch from current `main`, implement only that finding, and run its
narrow test and lint checks before handoff.

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

Audit acceptance requires a named, fully dispositioned census and executed
evidence for every retained finding. Dialect-IR reproducers must pass all owning
validators before a pass runs. Invalid-IR cases must be assigned to their
verifier or ingestion boundary instead of duplicated in passes. Each accepted
implementation pull request must leave verifier-valid output, pass its focused
regression and lint checks, and contain no generated, unrelated, or style-only
change. Record exact check results and environment-only limits on that focused
pull request, not as proof for the bulk snapshot.

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

Reconciliation base: 30bb9d1f8e9d81840aa42f47c3e577b8c76c4d63 from origin/main,
including merged focused replacements through `#2301`.

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

Final reconciled closure transcript:

    cmake --preset release \
      -DMLIR_DIR=/private/tmp/mqt-llvm-23.1.0/lib/cmake/mlir \
      -DLLVM_DIR=/private/tmp/mqt-llvm-23.1.0/lib/cmake/llvm
    # configured successfully with LLVM/MLIR 23.1.0

    cmake --build build/release --parallel 8
    # completed successfully

    ctest --test-dir build/release -L mqt-mlir-unittests --output-on-failure --parallel 8
    100% tests passed, 0 tests failed out of 3111

    uvx nox -s tests-3.13 -- test/python/test_mlir_qiskit_translation.py -k \
      parameter_vector_metadata_is_preflighted
    # package built with LLVM/MLIR 23.1.0 and all three cases emitted their
    # expected diagnostics; this Apple host then failed to translate the C++
    # exceptions through nanobind

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
complete `#2255` contract audit and recorded the full pass, verifier, pattern,
and registration census. The first focused review wave showed that the original
acceptance gate over-weighted malformed-IR defenses and hypothetical failures.
This revision makes all owning validators and an executed supported-path
reproducer mandatory, records merged and closed outcomes, and treats the bulk
implementation as a historical snapshot rather than an approved change set.
