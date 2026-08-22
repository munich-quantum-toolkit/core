# Legalize payload-scoped classical execution features

This ExecPlan is a living document. Keep `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` current while the work proceeds.
The repository-wide conventions in `.agent/PLANS.md` apply.

## Purpose / Big Picture

Target compilation must answer a payload-specific question: can the selected
execution profile run the program that remains after ordinary compiler
normalization and target-driven legalization? The old branch instead attached a
flat list of MLIR control-flow shapes to the hardware target and checked the
entire input module before canonicalization. That made QIR Adaptive support leak
into unrelated outputs, treated syntax as execution semantics, rejected unused
helpers, and duplicated MLIR's folding and reachability machinery.

After this change, a `CompilerTarget` contains optional metadata for one or more
selected `ProgramFormat` profiles. Each `ExecutionProfile` lists atomic semantic
features and records whether optional feature metadata is complete. Target
compilation materializes the selected profile as a serializable typed
`mqt.target_env` module attribute, normalizes first, legalizes finite counted
loops when the profile has no native iteration, and checks only the marked entry
point with MLIR dialect-conversion legality. A failed in-place compilation is
transactional because all passes run on a copy.

The observable result is that static/dead control and structural regions need no
runtime feature, unused helpers do not affect a program, reachable calls fail
closed until interprocedural execution semantics are modeled, and QDMI QIR
Adaptive metadata authorizes only its mandatory baseline for that payload.

## Progress

- [x] (2026-08-22) Re-read the requested architecture review and verify every
  material claim against `codex/classical-control-support` at `f07024023`.
- [x] (2026-08-22) Audit the target API, QDMI adapter/client, compiler pipeline,
  QCO/SCF canonicalization and loop utilities, tests, and docs.
- [x] (2026-08-22) Replace the obsolete preflight-oriented ExecPlan with this
  payload-scoped legalization plan.
- [x] (2026-08-22) Introduce `ProgramFeature` and immutable payload-scoped
      `CompilerTarget::ExecutionProfile` metadata, including unknown versus
      known empty profile metadata.
- [x] (2026-08-22) Add the typed, serializable `mqt.target_env` attribute and
  make target passes query it without treating execution features as a
  data-layout spec.
- [x] (2026-08-22) Make `QCOProgram::compileForTarget` accept the selected
      output profile and run transactionally on a copy.
- [x] (2026-08-22) Replace `TargetControlAnalysis` and the recursive
      whole-module preflight with cleanup-first target legalization and
      `ConversionTarget` legality rooted at `mqt.entry_point`.
- [x] (2026-08-22) Restore the released QDMI vector getter, add an ABI-safe
  optional query, remove trusted caller augmentation, and construct
  payload-scoped profiles.
- [x] (2026-08-22) Rewrite focused C++/Python tests and user documentation to
  describe and validate the new contract; retain one compact constant
  `qco.index_switch` regression.
- [x] (2026-08-22) Build affected libraries and run focused and complete
  relevant tests.
- [ ] Run `uvx nox -s lint`.

## Surprises & Discoveries

- `TargetControlAnalysis(getOperation())` and recursive verification start at
  the `ModuleOp`, so an uncalled LLVM helper can reject a valid entry program.
- The verifier runs before `populateQCOCleanupPipeline` and reimplements
  folding, selected-region traversal, region depth, zero-trip loops, and
  captures.
- Existing canonicalization already removes constant `qco.if`,
  `qco.index_switch`, `scf.if`, `scf.index_switch`, and `scf.execute_region`;
  SCCP handles cross-operation constants.
- The existing `qco::QuantumLoopUnroll` only selects loops with a quantum init
  argument and fails on non-constant full unrolling. Target legalization must
  attempt all finite entry-point `scf.for` loops and leave residual loops for
  the legality diagnostic.
- QDMI distinguishes a missing supported-format property from a successful empty
  list: `TEST_SESSION` returns `QDMI_ERROR_NOTSUPPORTED`, while the SC device
  reports success with size zero.
- QIR Adaptive format support guarantees mid-circuit measurement, result use as
  `i1`, and forward branching. Loops, switch-like branching, wider integer/float
  computation, functions, and related module features are optional.

## Decision Log

- Decision: replace `CompilerTarget::ClassicalControl` with atomic
  `ProgramFeature` values stored per `ProgramFormat` in `ExecutionProfile`.
  Rationale: execution semantics belong to a selected payload, and a generic
  `Conditional` flag cannot express condition provenance or prerequisites. Date:
  2026-08-22.
- Decision: make the complete profile list optional. `nullopt` means it was not
  reported; an engaged empty list means known empty; a present profile can mark
  optional feature metadata incomplete. Rationale: these are different facts and
  must survive fail-closed handling. Date: 2026-08-22.
- Decision: materialize the selected profile as a typed `mqt.target_env`
  attribute containing format, supported features, and completeness. Rationale:
  passes receive serializable MLIR input rather than captured out-of-band state.
  It may implement `DLTIQueryInterface`, but it is not a
  `DataLayoutSpecInterface`. Date: 2026-08-22.
- Decision: keep program requirements separate from execution-profile
  capabilities and compiler lowering legality; do not persist a stale
  requirements attribute. Date: 2026-08-22.
- Decision: normalize with SCCP and QCO cleanup before checking residual
  control, then use `ConversionTarget` with an empty rewrite set for legality.
  Rationale: MLIR canonicalization and conversion replace bespoke evaluation.
  Date: 2026-08-22.
- Decision: root legality at `mqt.entry_point`; ignore uncalled helpers and
  reject reachable `func.call` until interprocedural semantics are modeled.
  Date: 2026-08-22.
- Decision: if counted iteration is unavailable, fully unroll finite `scf.for`
  loops with an exact, expansion-bounded rewrite and clean up again. A residual
  loop is a runtime requirement. Date: 2026-08-22.
- Decision: compile a cloned `QCOProgram` and replace the original only after
  success. Rationale: ownership provides atomicity without a read-only
  preflight. Date: 2026-08-22.
- Decision: preserve the released throwing QDMI format getter and add a
  separately named optional getter. Rationale: changing only a C++ return type
  is ABI-unsafe. Date: 2026-08-22.
- Decision: remove QDMI caller augmentation. Infer only mandatory QIR Adaptive
  features for the Adaptive profile and mark optional features unknown. Date:
  2026-08-22.
- Decision: keep constant `qco.index_switch` canonicalization with one compact
  regression. Date: 2026-08-22.
- Decision: defer the QDMI specification extension and Qiskit/PennyLane SDK
  projections to separate changes. Date: 2026-08-22.

## Outcomes & Retrospective

The payload-scoped implementation and focused validation are complete.

The target-pipeline test block now covers payload isolation,
measurement-feedback and lifecycle prerequisites, mixed-domain arithmetic,
measured-qubit provenance through structured results, bounded aggregate
finite-loop legalization, bounded feedback provenance, residual-loop features,
entry-point reachability, generic reachable-call rejection, scalar and aggregate
math/LLVM computation, fail-closed unmodelled classical producers, CBit aliases,
unsupported output formats, cycle-safe recursive LLVM aggregates, structural
quantum-state limits, and transactional failure without pinning the removed
preflight evaluator. The focused target-compilation filter and compact
index-switch regression pass. The complete QCO IR binary passes all 487 tests,
and the complete compiler binary passes all 193 tests. Registering the MQT
dialect in the compiler test fixture also validates textual round-trips for
empty-feature target environments.

## Context and Orientation

`mlir/include/mlir/Compiler/ProgramFormat.h` owns output-format and
program-feature enums. `Target.h` and `Target.cpp` own immutable target and
profile metadata. Target storage is already private shared state, so replacing
the PR-only flat list does not change `CompilerTarget` object layout.

The MQT dialect owns metadata that survives quantum dialect conversions. The
target environment belongs there because it is the selected compilation contract
and must serialize with the module.

`Programs.cpp` owns `QCOProgram` value semantics and the default pipeline. It is
where the final `ProgramFormat` is known. `TargetCompilation.cpp` owns
target-driven QCO transformation, mapping, synthesis, and conformance; it uses
the selected environment for semantic legality and `CompilerTarget` for topology
and native operations.

`QDMIAdapter.cpp` snapshots devices. QDMI program formats describe accepted
payloads; string/module variants of one QIR profile map to one Core profile.

Focused tests live in `test_compiler_target.cpp`, `test_compiler_pipeline.cpp`,
`test_compiler_qdmi_adapter.cpp`, `test/qdmi/test_client.cpp`, and the
corresponding Python tests.

## Plan of Work

First, move `ProgramFormat` to a shared header and add atomic `ProgramFeature`
values. Add immutable profile construction, canonical ordering, duplicate-format
rejection, optional profile-list metadata, and selected-format queries. Project
the API into Python without the unreleased flat control names.

Second, add generated `mqt::TargetEnvAttr` with a stable textual form and
helpers for format, features, completeness, and feature queries. Register it and
allow it only as `mqt.target_env` on a module.

Third, thread `ProgramFormat` through transactional target compilation. The
target pipeline attaches the chosen environment before transformation.

Fourth, run SCCP and cleanup. When needed, fully unroll finite entry-point loops
and clean up again. Replace manual dispatch with `ConversionTarget` legality for
known QCO/SCF control, QTensor indices, quantum tensors/captures, and
measurement-derived conditions. Unknown branch interfaces and reachable calls
are illegal; ordinary operations are legal.

Fifth, add the optional QDMI format query and construct OpenQASM 3 and QIR Base
profiles with unknown optional-feature metadata, plus the mandatory QIR Adaptive
baseline with optional-feature metadata unknown, without cross-profile
promotion.

Finally, replace implementation-pinning tests and prose. Delete the unreleased
upgrading section, fold #2162 into the compiler-target changelog item, and
document cleanup-first, three-layer legality.

## Concrete Steps

Run from the clean worktree:

    cd /private/tmp/mqt-core-classical-control-redesign
    git status --short
    git diff --check

Configure a disconnected release build with local dependency sources if needed.
Build the smallest affected targets first, then compiler and unit-test aggregate
targets. Run focused filters for target metadata, target compilation, QDMI,
static index-switch canonicalization, and Python bindings. Then run the complete
relevant CTest aggregate and:

    uvx nox -s lint

Do not push unless the user separately requests it.

## Validation and Acceptance

Acceptance requires:

1. One target exposes different QIR Adaptive and QIR Base features; absent
   features fail closed.
2. Unknown profiles, known empty profiles, and unknown optional features remain
   distinguishable in C++ and Python.
3. `mqt.target_env` round-trips in textual MLIR.
4. Normalized-away control requires no runtime feature.
5. A finite loop compiles without native iteration and leaves no `scf.for`; a
   residual dynamic loop fails without `CountedIteration`.
6. Unsupported control in an unused helper is ignored; a reachable call fails.
7. Forward branching requires measurement-feedback semantics and does not
   authorize arbitrary external or wider classical computation.
8. Failed target compilation leaves the caller's program unchanged.
9. QDMI Adaptive features stay scoped to Adaptive and optional features remain
   unknown.
10. Focused/complete tests, `git diff --check`, and lint pass.

## Idempotence and Recovery

All edits are ordinary patches; builds and tests are repeatable. Compilation is
safe to retry because it runs on a copy. Work occurs in a separate clean
worktree; do not use destructive Git commands. Regenerate stubs with repository
tooling if necessary rather than modifying unrelated generated content.

## Artifacts and Notes

Useful work retained from the original implementation includes immutable target
storage, QDMI snapshotting, target compilation, and constant `qco.index_switch`
canonicalization. The abstraction and placement change; those contributions are
not discarded.

The QDMI v1.4 feature-property proposal is an external follow-up. It should use
per-format atomic feature records and preserve unknown versus known empty. It is
not implemented here.

## Interfaces and Dependencies

The intended C++ surface includes:

    enum class ProgramFeature : uint8_t { ... };

    class CompilerTarget::ExecutionProfile {
    public:
      static llvm::Expected<ExecutionProfile>
      create(ProgramFormat, std::vector<ProgramFeature> = {},
             bool optionalFeaturesKnown = true);
      ProgramFormat format() const noexcept;
      llvm::ArrayRef<ProgramFeature> features() const noexcept;
      bool optionalFeaturesKnown() const noexcept;
      bool supports(ProgramFeature) const noexcept;
    };

    std::optional<llvm::ArrayRef<ExecutionProfile>>
    CompilerTarget::executionProfiles() const noexcept;
    const ExecutionProfile*
    CompilerTarget::executionProfile(ProgramFormat) const noexcept;
    bool CompilerTarget::supportsProgramFeature(ProgramFormat,
                                                 ProgramFeature) const noexcept;

    bool QCOProgram::compileForTarget(const CompilerTarget&, ProgramFormat,
                                      bool enableTiming = false,
                                      bool enableStatistics = false);

    std::optional<std::vector<QDMI_Program_Format>>
    qdmi::Device::tryGetSupportedProgramFormats() const;

Implementation uses existing SCCP/canonicalization passes, exact widened-APInt
trip-count evaluation, an aggregate-bounded `scf.for` rewrite driven by
`applyOpPatternsGreedily`, `ConversionTarget`, `RewritePatternSet`, and
`applyPartialConversion`. No new third-party dependency is required.
