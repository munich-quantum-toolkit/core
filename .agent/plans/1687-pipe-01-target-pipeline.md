# Compose the compiler-target pipeline

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core now has an immutable MLIR-owned `CompilerTarget`, a target-backed
mapping pass, target-independent two-qubit gate fusion, target-native synthesis,
and final target-conformance verification. Those pieces are independently usable
but are not yet composed behind one safe high-level operation. After this
change, a C++ user can call `QCOProgram::compileForTarget(target)` or pass an
optional target to `runDefaultPipeline` and receive a program that has been
decomposed, optimized, mapped, synthesized, verified, and cleaned up in the only
valid order.

The observable proof is a compiler unit test that starts with a multi-controlled
gate, compiles it to a topology-constrained U/CZ target, and finishes with
target-site assignments and only supported operations. The same test would fail
if mapping ran before multi-qubit decomposition, if native synthesis ran before
routing, or if the final verifier were omitted.

This slice also removes the coupling-only `QCOProgram::placeAndRoute` and Python
`QCOProgram.place_and_route` convenience APIs. Mapping remains directly
benchmarkable through `qco::createMappingPass`, while fusion, native synthesis,
and conformance remain directly benchmarkable through their existing pass
factories. The MLIR Compiler Collection has not been released, so no
compatibility shim or upgrade-guide entry is required.

## Milestones

The first milestone establishes one compiler-owned composition point. It adds
the reusable target pipeline populator, makes `QCOProgram::compileForTarget`
delegate to it, and teaches the default pipeline to use that same sequence
without duplicating pass construction. It is complete when both C++ entry points
produce an equivalent target-conforming QCO program.

The second milestone removes the obsolete coupling-only surface. It deletes the
C++ and Python `placeAndRoute`/`place_and_route` APIs, updates every
in-repository caller, and regenerates the authoritative Python stub. It is
complete when no coupling-to-target adapter or generated declaration remains.

The third milestone proves the integration boundary. Focused tests must exercise
decomposition, optimization, mapping, native synthesis, conformance, and the QIR
path, while explicitly rejecting outputs that cannot preserve a target
assignment. The complete compiler and pass-specific suites, header-set
verification, repository lint, and an independent exact-head review then make
the revision ready for publication.

## Progress

- [x] (2026-08-03 23:00Z) Verified that PR #1998 passed every check at exact
      head `4d598062a71f228d50463061b0b171bb7f30dc6a`, squash-merged it, and
      verified merged `main` commit `108454d1714dbc1b4f0079272d33a017ea35b8e4`.
- [x] (2026-08-03 23:00Z) Created a clean PIPE-01 worktree and branch from that
      exact merged `origin/main`, then read `AGENTS.md`, `docs/ai_usage.md`, and
      `.agent/PLANS.md`.
- [x] (2026-08-03 23:00Z) Inventoried the current compiler, pass, binding,
      generated-stub, CMake, test, and changelog surfaces and recorded the
      compact design below.
- [x] (2026-08-04) Added the reusable target compilation pipeline and the C++
      high-level target APIs.
- [x] (2026-08-04) Removed the coupling-only C++ and Python high-level APIs and
      regenerated the authoritative Python stub.
- [x] (2026-08-04) Added focused behavioral coverage for pass order, native
      conformance, target-aware QIR compilation, optional-target default
      compilation, and invalid target/output combinations.
- [x] (2026-08-04) Built the compiler and pass-specific test targets and the
      compiler interface-header set; passed 217 compiler, 27 mapping, 21 target
      synthesis, 229 decomposition, and 25 focused Python MLIR tests; passed
      changed-source clang-tidy 22.1.8, repository lint, and diff checks.
- [x] (2026-08-04) Completed two independent exact-head review rounds. The first
      had one nonblocking QIR site-ID test improvement, which was implemented;
      the amended head `81d55fe0e4f2397f51b074a9dd93ff967c15b00f` passed the
      second review with no findings.
- [x] (2026-08-04) Published draft PR #1999 from the exact current `main` base
      and added its consolidated compiler-target changelog entry together with
      foundation PR #1993.
- [x] (2026-08-04) Corrected the consolidated changelog attribution after the
      publication review identified Simon Hofmann's co-authorship of foundation
      PR #1993.
- [ ] Confirm the final changelog-bearing head independently and monitor PR
      #1999's exact-head CI.

## Surprises & Discoveries

- Observation: `QCOProgram::placeAndRoute` is the only remaining caller that
  converts a coupling list into a `CompilerTarget`. Every mapping-pass caller
  already uses `createMappingPass(const CompilerTarget&, MappingPassOptions)`.
  Evidence: a repository-wide symbol search on merged `main` finds the
  conversion only in `mlir/lib/Compiler/Programs.cpp`, its declaration, one
  compiler test, one nanobind definition, and the generated stub.
- Observation: the prior draft integration ran the generic QCO pipeline before
  multi-controlled decomposition and then appended target compilation. That
  ordering cannot guarantee that routing sees only one- and two-qubit
  operations. Evidence: merged `runDefaultPipeline` currently runs cleanup, the
  textual QCO pipeline, and cleanup; there is no target path yet.
- Observation: `MQTCompilerPipeline` already links `MQTCompilerTarget`,
  `MLIRQCOTransforms`, and `MQT::MLIRSupport`, and `mqt-cc` already links
  `MQTCompilerPipeline`. A small compiler-owned target-pipeline source therefore
  needs no new dependency boundary and can later be reused by the final
  QDMI/mqt-cc integration.
- Observation: target-independent two-qubit gate fusion uses CZ as its fixed
  symmetric generic entangler and rewrites only strictly profitable constant
  runs. It belongs after the ordinary generic optimization passes and before
  mapping; target-native synthesis remains after mapping.
- Observation: Jeff conversion deliberately lowers `qco.static` to a generic
  allocation and therefore discards physical site identifiers. A target-aware
  Jeff result would falsely claim successful compilation while losing the
  mapping. Evidence: `QCOToJeff.cpp` converts each static allocation without
  carrying its index into the Jeff program.
- Observation: LLVM's `llvm-prefer-static-over-anonymous-namespace` and
  clang-tidy's `readability-static-definition-in-anonymous-namespace` require a
  file-local test helper to be `static` outside the anonymous namespace. Moving
  only that helper satisfied both checks without an exemption.
- Observation: the first stub and lint invocations were blocked only while
  fetching their pinned Python tools inside the network-restricted sandbox.
  Authorized retries succeeded; stub generation then needed the installed MLIR
  22.1 package selected explicitly. No source or dependency change was required
  for either environment boundary.
- Observation: the first independent exact-head review found no required change.
  Its only nonblocking follow-up was to prove explicitly that sparse target site
  IDs survive QIR conversion. The target-aware QIR test now walks the generated
  `llvm.inttoptr` operands and checks all three site IDs.
- Observation: the publication review found that the consolidated changelog
  entry initially omitted Simon Hofmann despite his co-authorship of foundation
  PR #1993. The entry now credits both contributors.

## Decision Log

- Decision: add
  `populateTargetCompilationPipeline(OpPassManager&, const CompilerTarget&)`
  under `mlir/Compiler/TargetCompilation.{h,cpp}` and build it into
  `MQTCompilerPipeline`. Rationale: the compiler subtree owns `CompilerTarget`
  and high-level program compilation, the existing link graph already provides
  every required pass, and the final mqt-cc/QDMI bridge can reuse one sequence
  without duplicating pass order. Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: make the canonical sequence multi-controlled decomposition with a
  minimum of two controls, the default QCO optimization passes,
  target-independent two-qubit gate fusion, target-backed mapping, target-native
  synthesis, target-conformance verification, and QCO cleanup. Rationale:
  routing is defined only for one- and two-qubit operations, optimization should
  reduce work before routing, routing may insert SWAPs, native synthesis must
  lower those SWAPs, and conformance must inspect the final mapped program
  before cleanup. Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: expose
  `QCOProgram::compileForTarget(const CompilerTarget&, bool, bool)` with timing
  and statistics defaulting off, and pass the same options through target-aware
  `runDefaultPipeline`. Rationale: both high-level entry points use exactly one
  pass manager and one sequence while preserving the existing observability
  controls without a private duplicate implementation. Date/Author: 2026-08-03,
  GPT-5.6 via Codex.
- Decision: add `const CompilerTarget* target = nullptr` before the textual QCO
  pipeline argument of the sole `runDefaultPipeline` function. Reject a target
  for `QCImport` and raw `QCO` outputs because those checkpoints deliberately
  stop before optimization. Rationale: a non-null target must never be silently
  ignored, and a pointer expresses an optional borrowed immutable target without
  introducing a second overload or copying requirement. Date/Author: 2026-08-03,
  GPT-5.6 via Codex.
- Decision: reject a custom textual QCO pipeline when a compiler target is
  supplied. Rationale: target compilation has one validated order beginning with
  multi-qubit decomposition; injecting an opaque pipeline before or inside that
  sequence either violates the routing precondition or adds a fallible,
  callback-heavy composition API. Advanced and benchmark users retain
  `QCOProgram::runPassPipeline` and the individual pass factories. Date/Author:
  2026-08-03, GPT-5.6 via Codex.
- Decision: remove `placeAndRoute` and `place_and_route` outright, regenerate
  the binding stub, and add no migration shim or `UPGRADING.md` entry.
  Rationale: compiler-target mapping is the single modern abstraction, the
  compiler collection is unreleased, and retaining a coupling-only path would
  recreate target construction and configuration redundancy at the public
  boundary. Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: do not expose `CompilerTarget` or `compile_for_target` to Python in
  this slice. Rationale: the final INT-1687 work owns the FoMaC/QDMI adapter,
  Python target binding, and mqt-cc device experience. PIPE-01 only removes the
  obsolete Python coupling API and leaves `compile_program` passing a null
  target until that bridge lands. Date/Author: 2026-08-03, GPT-5.6 via Codex.
- Decision: reject target-aware Jeff output alongside `QCImport` and raw QCO,
  while supporting `QCOOptimized`, QC, and QIR output. Rationale: QC and QIR
  preserve the mapped static-site semantics, whereas the current Jeff conversion
  intentionally discards static site identifiers. Extending the Jeff
  representation is outside this compact pipeline-composition slice.
  Date/Author: 2026-08-04, GPT-5.6 via Codex.

## Outcomes & Retrospective

The implementation keeps target compilation inside the MLIR compiler layer,
defines the pass sequence once, and removes rather than adapts the obsolete
coupling-only surface. The behavioral suite includes a deliberately unsupported
measurement target so that removing final conformance would make a test fail,
plus direct QIR coverage and explicit rejection of Jeff because its current
lowering cannot preserve physical site assignments. All focused C++, Python,
header, clang-tidy, lint, and diff validation passes. The first independent
exact-head review found no required changes, and its sole nonblocking QIR
site-ID follow-up is addressed. Draft PR #1999 now carries the consolidated
compiler-target changelog entry with the complete #1993 attribution. Final
exact-head confirmation and CI remain.

## Context and Orientation

`mlir/include/mlir/Compiler/Target.h` and `mlir/lib/Compiler/Target.cpp` define
the validated, cheaply copyable `mlir::CompilerTarget`. An absent topology means
all-to-all connectivity; an explicit topology is canonicalized and connected. An
absent operation set means all operations are native; an explicit operation set
supplies a homogeneous gate capability set and a target-wide synthesis basis
when one can be derived.

`mlir/include/mlir/Dialect/QCO/Transforms/Mapping/Mapping.h` exposes
`qco::createMappingPass(const CompilerTarget&, MappingPassOptions)`. The pass
assigns program qubits to target site identifiers and routes two-qubit
operations over an explicit topology. It deliberately supports only one- and
two-qubit operations, so multi-controlled gates must be decomposed first.

`mlir/include/mlir/Dialect/QCO/Transforms/Passes.h` exposes
`qco::createFuseTwoQubitGates`, `qco::createTargetNativeSynthesis`, and
`qco::createVerifyTargetConformance`. Fusion is target independent and rewrites
only strictly profitable constant two-qubit windows using symmetric CZ-based
resynthesis. Native synthesis lowers unsupported one- and two-qubit operations,
including routing SWAPs, into the target-wide basis. Conformance then rejects
unassigned qubits, unknown static sites, and unsupported final operations.

`mlir/include/mlir/Support/Passes.h` and `mlir/lib/Support/Passes.cpp` own
reusable cleanup and optimization populators.
`populateDecomposeMultiControlledPipeline(pm, 2)` lowers supported
multi-controlled forms. `populateDefaultQCOOptimizationPipeline` performs the
ordinary target-independent QCO optimization. `populateQCOCleanupPipeline`
canonicalizes, normalizes global phase, eliminates common subexpressions,
shrinks QTensor allocations, and removes dead values.

`mlir/include/mlir/Compiler/Programs.h` and `mlir/lib/Compiler/Programs.cpp`
define move-aware typed programs and `runDefaultPipeline`. The local `runPasses`
helper currently templates over a populator even though every caller has the
same void contract. PIPE-01 replaces that template with
`llvm::function_ref<void(OpPassManager&)>`, retaining module diagnostics and
adding optional timing/statistics configuration.

`bindings/mlir/register_mlir.cpp` defines the Python binding and
`python/mqt/core/mlir.pyi` is generated from it by the `stubs` Nox session. The
generated stub must never be edited by hand. The binding currently exposes the
obsolete coupling-only `place_and_route`; this slice deletes that definition and
regenerates the stub. It does not yet bind `CompilerTarget`.

The focused compiler tests live in
`mlir/unittests/Compiler/test_compiler_pipeline.cpp` and build as
`build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler`.
Pass-specific mapping and synthesis behavior remains covered by their dedicated
unit tests. PIPE-01 adds only integration behavior that cannot be proven by
those isolated suites.

The task may modify the new target-compilation header and source,
`mlir/lib/Compiler/CMakeLists.txt`, `Programs.h`, `Programs.cpp`, the compiler
unit test, the nanobind source, the generated Python stub, `CHANGELOG.md`, and
this ExecPlan. It must not add the QDMI/FoMaC adapter, expose target
construction to Python, alter mqt-cc device options, or modify another worktree.

## Plan of Work

First add `mlir/include/mlir/Compiler/TargetCompilation.h` with the single
public void populator and `mlir/lib/Compiler/TargetCompilation.cpp` with the
canonical pass sequence. Add the source to `MQTCompilerPipeline`; the existing
public-header glob and link dependencies are sufficient.

Next replace the templated `runPasses` helper in `Programs.cpp` with an explicit
`llvm::function_ref<void(OpPassManager&)>` contract and optional timing and
statistics flags. Add `QCOProgram::compileForTarget` to `Programs.h` and
implement it by calling the shared populator. Delete the coupling-list
conversion, `placeAndRoute`, and now-unused includes.

Then add the optional target parameter to `runDefaultPipeline`. Validate
target/output and target/custom-pipeline combinations before consuming or
mutating the input. Keep the target-free path unchanged. For a target-aware
path, run only the canonical target compilation pipeline and pass through
timing/statistics; do not prepend or append a second generic pipeline. Update
all in-repository positional callers for the new signature.

Delete the Python `place_and_route` binding. Keep `compile_program` target-free
until INT-1687 provides a Python compiler-target object, then regenerate
`python/mqt/core/mlir.pyi` through the repository's stubs session.

Add compiler tests that construct a three-site line target with homogeneous U
and CZ capabilities. Compile a program containing a multi-controlled gate and
assert success, static target-site assignment, absence of routing SWAPs, and
support for every remaining relevant operation. Exercise both
`QCOProgram::compileForTarget` and target-aware `runDefaultPipeline`, including
direct QIR output, and test that raw checkpoints, Jeff, and a custom textual
pipeline reject a non-null target. Remove the old coupling-only API call from
the general optimization API test.

Finally add a concise unreleased changelog entry for the coordinated
compiler-target pipeline, leaving the pull-request reference to be filled only
after GitHub assigns it. Do not add an upgrade guide entry.

## Concrete Steps

Run all commands from the repository root of this task worktree.

Inspect the initial state:

    git status --short --branch
    git rev-parse HEAD

The expected head is the merged SYN-01 commit
`108454d1714dbc1b4f0079272d33a017ea35b8e4`, with no worktree changes.

After each source-edit batch, configure and build with an installed MLIR 22.1
package. Set `MLIR_DIR` to that installation's `lib/cmake/mlir` directory:

    MLIR_DIR=<path-to-llvm-22.1>/lib/cmake/mlir \
      ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release --target \
      mqt-core-mlir-unittests-compiler \
      mqt-core-mlir-unittest-mapping \
      mqt-core-mlir-unittest-target-synthesis \
      mqt-core-mlir-unittest-decomposition \
      MQTCompilerPipeline_verify_interface_header_sets

Run focused compiler coverage:

    ./.agent/run.sh \
      ./build/release/mlir/unittests/Compiler/\
      mqt-core-mlir-unittests-compiler \
      --gtest_filter='CompilerPipelineTest.*Target*'

Then run the complete compiler, mapping, synthesis, and multi-controlled
decomposition suites:

    ./.agent/run.sh \
      ./build/release/mlir/unittests/Compiler/\
      mqt-core-mlir-unittests-compiler
    ./.agent/run.sh \
      ./build/release/mlir/unittests/Dialect/QCO/Transforms/Mapping/\
      mqt-core-mlir-unittest-mapping
    ./.agent/run.sh \
      ./build/release/mlir/unittests/Dialect/QCO/Transforms/NativeSynthesis/\
      mqt-core-mlir-unittest-target-synthesis
    ./.agent/run.sh \
      ./build/release/mlir/unittests/Dialect/QCO/Transforms/Decomposition/\
      mqt-core-mlir-unittest-decomposition

Regenerate and verify the Python stub after the binding deletion:

    ./.agent/run.sh uvx nox -s stubs
    git diff -- python/mqt/core/mlir.pyi

Build the binding and run focused Python MLIR tests if the stubs session does
not already build it:

    ./.agent/run.sh uv sync --inexact --only-group build --only-group test
    ./.agent/run.sh uv sync --inexact --no-dev \
      --no-build-isolation-package mqt-core
    ./.agent/run.sh uv run --no-sync pytest test/python/test_mlir.py

Finish with repository validation and diff inspection:

    ./.agent/run.sh uvx nox -s lint
    git diff --check
    git status --short
    git diff --stat
    git diff

Run changed-source clang-tidy with LLVM/Clang 22 if the configured build exposes
the repository's established lint target or compilation database. Record the
exact command and result in this plan.

## Validation and Acceptance

The new `TargetCompilation` populator is public from `MQTCompilerPipeline` and
contains each required pass exactly once in this order: multi-controlled
decomposition, default QCO optimization, target-independent two-qubit gate
fusion, target-backed mapping, target-native synthesis, target-conformance
verification, and QCO cleanup.

`QCOProgram::compileForTarget` succeeds on a multi-controlled circuit and a
three-site line target with U/CZ capabilities. The resulting program verifies,
uses `qco.static` target assignments, contains no `qco.alloc` or `qco.swap`, and
contains no operation rejected by `CompilerTarget::supports`.

`runDefaultPipeline` without a target preserves every current output-format and
custom-pipeline behavior. With a target, it produces the same conforming target
program for `QCOOptimized`, QC, and QIR output. A target combined with
`QCImport`, raw `QCO`, Jeff, or a custom textual QCO pipeline returns no result
with a regular diagnostic instead of silently ignoring the target.

No `placeAndRoute`, `place_and_route`, coupling-to-target conversion helper, or
generated stub entry remains. The mapping, fusion, synthesis, and conformance
pass factories remain independently callable.

All focused suites and repository lint pass. The final diff contains no
compatibility shim, no `UPGRADING.md` change, no QDMI/FoMaC adapter, no mqt-cc
device integration, and no unrelated cleanup.

## Idempotence and Recovery

All inspection, configuration, build, test, stub-generation, and lint commands
are repeatable. CMake and tool caches remain local to this worktree through
`.agent/run.sh`. If configuration fails because MLIR is not auto-discovered,
repeat it with an installed LLVM/MLIR 22.1 `MLIR_DIR`; do not change source or
dependency metadata for that environment boundary.

Stub generation is authoritative: if it fails, leave the binding source as the
source of truth, repair the environment or binding issue, and rerun the stubs
session rather than editing `python/mqt/core/mlir.pyi`.

No destructive Git operation is needed. Preserve unrelated changes if any appear
and stop before overwriting them. External publication, review-thread mutation,
and merge actions remain governed by the user's explicit authorization and the
repository AI policy.

## Artifacts and Notes

Initial repository evidence:

    origin/main and worktree HEAD:
      108454d1714dbc1b4f0079272d33a017ea35b8e4
      ✨ Split target optimization from native synthesis (#1998)

    remaining coupling-only surface:
      mlir/include/mlir/Compiler/Programs.h
      mlir/lib/Compiler/Programs.cpp
      bindings/mlir/register_mlir.cpp
      python/mqt/core/mlir.pyi
      mlir/unittests/Compiler/test_compiler_pipeline.cpp

    reusable pass factories:
      qco::createMappingPass(const CompilerTarget&, MappingPassOptions)
      qco::createFuseTwoQubitGates()
      qco::createTargetNativeSynthesis(const CompilerTarget&)
      qco::createVerifyTargetConformance(const CompilerTarget&)

Update this section with concise test counts, lint results, and the final
diffstat as implementation proceeds.

Current validation evidence:

    compiler interface-header verification:
      MQTCompilerPipeline_verify_interface_header_sets built successfully
    compiler tests:
      217 tests from 8 suites passed
    mapping tests:
      27 tests from 1 suite passed
    target-synthesis tests:
      21 tests from 2 suites passed
    decomposition tests:
      229 tests from 25 suites passed
    Python MLIR tests:
      25 tests passed on Python 3.14
    generated stub:
      place_and_route and its now-unused Sequence import were removed
    clang-tidy:
      LLVM 22.1.8 with the active Xcode SDK and libc++ paths;
      Programs.cpp, TargetCompilation.cpp, test_compiler_pipeline.cpp,
      and register_mlir.cpp passed without diagnostics in changed sources
    repository lint:
      every all-file hook passed
    whitespace:
      git diff --check passed
    independent review:
      no required changes; the sole nonblocking QIR site-ID follow-up was
      implemented and revalidated
      the publication review's missing #1993 co-author credit was corrected
    publication:
      draft PR #1999 created from exact head
      81d55fe0e4f2397f51b074a9dd93ff967c15b00f

## Interfaces and Dependencies

At completion, `mlir/include/mlir/Compiler/TargetCompilation.h` declares:

    namespace mlir {
    class CompilerTarget;
    class OpPassManager;

    void populateTargetCompilationPipeline(
        OpPassManager& pm, const CompilerTarget& target);
    }

`QCOProgram` declares:

    [[nodiscard]] bool
    compileForTarget(const CompilerTarget& target,
                     bool enableTiming = false,
                     bool enableStatistics = false);

The sole default-pipeline entry point declares:

    [[nodiscard]] std::optional<CompilerProgram>
    runDefaultPipeline(
        CompilerInput&& program, ProgramFormat output,
        const CompilerTarget* target = nullptr,
        std::string_view qcoPipeline = "mqt-qco-default",
        bool enableTiming = false, bool enableStatistics = false);

`MQTCompilerPipeline` owns `TargetCompilation.cpp` and links only the existing
MLIR compiler target, QCO transform, and support libraries. CoreFoMaC, QDMI,
CoreIR, and a dynamic provider boundary are not added by this slice.

Revision note (2026-08-03, GPT-5.6 via Codex): created the initial
self-contained PIPE-01 execution plan after SYN-01 merged, recording the
canonical pass order, API removals, test contract, validation commands, and
coordination boundary.

Revision note (2026-08-04, GPT-5.6 via Codex): recorded implementation progress,
added narrative milestones, rejected target-aware Jeff because it cannot
preserve static site IDs, added direct QIR and conformance integration coverage,
and made the validation commands portable and complete.

Revision note (2026-08-04, GPT-5.6 via Codex): recorded the first independent
exact-head review and addressed its sole nonblocking follow-up by asserting that
sparse physical site IDs survive target-aware QIR conversion.

Revision note (2026-08-04, GPT-5.6 via Codex): recorded the no-finding amended
head review, draft PR #1999 publication, and consolidated compiler-target
changelog entry.

Revision note (2026-08-04, GPT-5.6 via Codex): recorded and corrected the
publication review's missing co-author attribution for foundation PR #1993.
