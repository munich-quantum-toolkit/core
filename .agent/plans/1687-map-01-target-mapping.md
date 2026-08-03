# Map QCO programs to compiler targets

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

After this change, callers can run the QCO `place-and-route` pass against an
immutable `mlir::CompilerTarget` instead of constructing a second graph wrapper
from a dense coupling set. The mapper uses the target's validated topology for
placement and routing, emits provider-defined site identifiers in `qco.static`
operations, and continues to insert ordinary `qco.swap` operations without
making target-native gate or direction decisions.

The behavior is visible in the existing mapping unit-test binary. Tests
construct targets with noncontiguous site identifiers, map scalar and tensor
allocations, and verify that every routed two-qubit operation lies on a target
edge. A small program mapped to a much larger target also demonstrates that
unused target sites are not all materialized as IR operations.

This task is based on the merged compiler-target foundation at
`f775395a25fddba0a3b54996416d1311bc6ebe71`. It changes the mapper and its tests
plus the compiler pipeline's sole mapping-pass call site. The high-level
`QCOProgram::placeAndRoute` coupling-set API remains in place for a later
pipeline integration task.

## Progress

- [x] (2026-08-03 12:29Z) Verified the initial stacked worktree was clean, on
      the expected branch, and exactly at the pre-squash compiler-target head
      `b6eb95521cb76224c137496f113f38b9ce295854`.
- [x] (2026-08-03 12:29Z) Read the repository agent policy, AI-usage policy,
      ExecPlan requirements, MQT review workflow, remediation protocol, and
      review rubric.
- [x] (2026-08-03 12:29Z) Audited the pre-MAP mapper, its vote-and-restore
      behavior from pull request #1951, the compiler-target API, and existing
      mapping tests.
- [x] (2026-08-03 12:31Z) Created this living plan and passed the complete
      targeted pre-commit hook set on it.
- [x] (2026-08-03 13:09Z) Replaced the mapper's private graph wrapper with the
      `CompilerTarget` topology contract; the old coupling-set entry point is
      now only a temporary forwarder for PIPE to delete.
- [x] (2026-08-03 13:09Z) Added one recursive discovery/planning walk that
      supports top-level scalar and tensor allocations, rejects nested dynamic
      allocations and higher-arity unitaries before mutation, and reuses its
      records for routing and placement.
- [x] (2026-08-03 13:09Z) Materialized only program qubits and vacant target
      vertices used by the selected routing plan, with provider site IDs in
      output.
- [x] (2026-08-03 13:09Z) Added focused mapping, diagnostic, allocation,
      sparse-workspace, empty-operation-set, and noncontiguous-ID tests while
      preserving all #1951 regressions.
- [x] (2026-08-03 13:09Z) Updated the existing `Unreleased` mapping changelog
      entry without adding a pull-request number or upgrade note.
- [x] (2026-08-03 13:09Z) Built the focused targets; passed 27 mapping tests,
      218 compiler tests, changed-source clang-tidy 22.1.8, repository lint, and
      `git diff --check`.
- [x] (2026-08-03 13:17Z) Recorded the outcome and created the focused signed
      commit with the required AI-assistance trailer.
- [x] (2026-08-03 15:13Z) Restacked the sole MAP commit without conflicts onto
      the merged compiler-target squash
      `f775395a25fddba0a3b54996416d1311bc6ebe71`; stable patch IDs confirm the
      non-plan implementation patch is unchanged.
- [x] (2026-08-03 15:18Z) Reconfigured and rebuilt the target, transforms, and
      test interfaces against the merged base; passed 27 mapping tests, all 218
      compiler tests, changed-source clang-tidy 22.1.8, targeted hooks, full
      repository lint, and `git diff --check`.
- [x] (2026-08-03 17:51Z) Confirmed the sole unresolved review thread requested
      deletion of the coupling-set mapping factory, removed its declaration,
      conversion helper, and implementation, and migrated `Programs.cpp` to
      construct and pass a `CompilerTarget` directly.
- [x] (2026-08-03 17:57Z) Rebuilt the target, transforms, compiler pipeline, and
      affected tests; passed 27 mapping tests, all 218 compiler tests, the
      focused high-level mapping test, changed-file clang-tidy 22.1.8, targeted
      hooks, full repository lint, and `git diff --check`.
- [x] (2026-08-03 18:01Z) An independent read-only review approved the exact
      worktree diff with no required changes or new findings.
- [x] (2026-08-03 18:02Z) Created the focused signed review follow-up commit
      with the required AI-assistance trailer.

## Surprises & Discoveries

- Observation: the pre-MAP mapper's `AugmentedDevice` duplicated topology
  storage, neighbour lookup, distances, and maximum degree that `CompilerTarget`
  already validates and caches. Evidence: before this task,
  `mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp` defined the wrapper,
  while `mlir/include/mlir/Compiler/Target.h` exposes `numQubits`,
  `areAdjacent`, `forEachNeighbour`, `distanceBetween`, and `maxDegree`.
- Observation: the pre-MAP mapper intentionally maintained a full permutation
  over all target vertices, even when the program used fewer qubits. Its
  placement phase therefore materialized every target site. The implemented cold
  routing preview retains the full virtual permutation while identifying only
  the vacant program indices whose qubits are touched by real SWAPs.
- Observation: pull request #1951 added branch-layout voting and restoration to
  `qco.index_switch`. That implementation is already on the task base and must
  remain the source of truth; importing the older pull request #1687 mapper
  would risk dropping those regressions.
- Observation: the compiler-target foundation merged as squash
  `f775395a25fddba0a3b54996416d1311bc6ebe71` after MAP-01 was developed on its
  pre-squash stack. Replaying the sole MAP commit onto that squash required no
  conflict resolution. Before this plan-only evidence update, `git range-diff`
  marked the old and new patches with `=`; in the final state, the non-plan
  stable patch IDs match, so the current-main target implementation was
  preserved unchanged.
- Observation: the installed clang-tidy 22.1.8 does not infer Apple libc++ from
  the AppleClang compilation command. The first invocation therefore failed on
  `<type_traits>` and was not treated as validation. Re-running with the Xcode
  SDK and libc++ include paths made explicit produced clean results for both
  changed translation units and the public mapping header.
- Observation: after the restack, CMake did not recognize the cached MLIR
  version when invoked without `MLIR_DIR`. Re-running the same release
  configuration with the installed LLVM 22.1.3 MLIR package path explicit
  succeeded without a source or build-system change.
- Observation: the coupling-set factory was used only by
  `QCOProgram::placeAndRoute`; all mapping tests and every other pass caller
  already used `CompilerTarget`. Moving the unavoidable coupling-to-target
  conversion to that high-level API boundary removes 50 lines from the mapping
  implementation and 13 lines from its public header without changing mapping
  behavior.

## Decision Log

- Decision: store a cheap value copy of `CompilerTarget` in the pass. Rationale:
  the type shares immutable storage, so pass instances own a stable target
  contract without rebuilding or duplicating topology. Date/Author: 2026-08-03,
  Codex.
- Decision: use dense zero-based compiler vertices for `Layout`, A* search, and
  restoration, and translate through `CompilerTarget::siteForVertex` only when
  creating `qco.static`. Rationale: layout algorithms require dense indices,
  while output IR must retain provider-defined identifiers. Date/Author:
  2026-08-03, Codex.
- Decision: route every one-qubit unitary directly and every two-qubit unitary
  solely according to undirected target adjacency. Do not query operation
  support, native bases, calibration, or direction. Rationale: MAP-01 owns
  topology mapping only; native synthesis belongs to a separate task.
  Date/Author: 2026-08-03, Codex.
- Decision: keep the complete target-sized virtual layout, perform a
  deterministic cold preview for the chosen initial layout, and materialize only
  active program indices plus vacant indices touched by preview SWAPs.
  Rationale: the routing algorithm can use workspace without growing small
  programs to the complete target in IR. Date/Author: 2026-08-03, Codex.
- Decision: use one recursive discovery/planning walk to validate nested dynamic
  allocations and higher-arity unitaries, collect top-level allocation handles,
  and record whether two-qubit routing is needed. Store each discovered tensor
  chain once. Reuse these records for initial wires, sparse preview, and
  allocation replacement; the later recursive walk is mutation-only for
  structured-control extension. Rationale: unsupported input is diagnosed
  atomically without redundant whole-function diagnostic walks. Date/Author:
  2026-08-03, Codex.
- Decision: preserve the current vote, restore, converge, and region-dispatch
  implementation unchanged except where the target topology abstraction or
  sparse-wire bookkeeping requires mechanical adaptation. Rationale: the #1951
  regressions are the current mapping contract. Date/Author: 2026-08-03, Codex.
- Decision: expose only the target-taking mapping factory. Convert the current
  high-level `QCOProgram::placeAndRoute` coupling input directly into a
  `CompilerTarget` inside `Programs.cpp`, where that legacy surface is already
  isolated, and link `MQTCompilerPipeline` to `MQTCompilerTarget`. Rationale:
  the mapping library should have one target-native contract and no forwarding
  shim; PIPE can later remove the remaining high-level coupling API without
  touching Mapping again. Date/Author: 2026-08-03, Codex.

## Outcomes & Retrospective

Implementation and validation are complete. The mapper now owns a cheap
`CompilerTarget` value, uses its dense topology throughout placement and
routing, translates to provider site IDs only for `qco.static`, supports scalar
and mixed allocation forms, and materializes only active or preview-touched
workspace qubits. The single discovery/planning walk makes unsupported nested
allocations and higher-arity operations fail before mutation.

The existing mapping suite, including #1951 vote-and-restore behavior, passes
with the focused new coverage. The compiler suite, changed-source clang-tidy,
and repository lint also pass. High-level pipeline ownership remains unchanged,
while the mapping library now exposes only its target-native factory. The
conflict-free, patch-equivalent restack and fresh validation demonstrate the
same result against the merged compiler-target foundation.

## Context and Orientation

`mlir::CompilerTarget`, declared in `mlir/include/mlir/Compiler/Target.h` and
implemented in `mlir/lib/Compiler/Target.cpp`, is an immutable target model.
Provider site identifiers can be sparse or noncontiguous, but the target stores
them in a stable site order and exposes dense compiler vertices for algorithms.
An absent topology denotes all-to-all connectivity; an explicit topology is
validated as a connected undirected graph and has cached distances.

The QCO mapping pass lives in
`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp`, with its public factory
in `mlir/include/mlir/Dialect/QCO/Transforms/Mapping/Mapping.h`. The
target-taking factory stores a cheap `CompilerTarget` value, runs SABRE-style
layout refinement and A* routing against its topology, and rewrites dynamic
qubits to `qco.static`. `QCOProgram::placeAndRoute` constructs a target at its
existing coupling-input boundary and calls this same factory. A `Layout` remains
a complete virtual permutation, but the IR materializes only active program
qubits and vacant indices touched by routing.

The mapper walks linear qubit SSA chains using `WireIterator`. Scalar
`qco.alloc` produces one chain directly. A `qtensor.alloc` produces a tensor;
top-level `qtensor.extract` operations establish tensor-backed qubit chains, and
matching insertions return them. Structured `scf.for`, `scf.while`, `qco.if`,
and `qco.index_switch` operations carry qubit chains through nested regions. The
mapper extends these operations with any extra vacant routing wires it
materializes.

`mlir/unittests/Dialect/QCO/Transforms/Mapping/test_mapping.cpp` owns mapper
behavior tests. Its executable is
`build/release/mlir/unittests/Dialect/QCO/Transforms/Mapping/mqt-core-mlir-unittest-mapping`.
The current suite covers straight-line programs, nested structured control flow,
layout convergence, and the #1951 index-switch vote-and-restore regression.

The task may modify the mapper header and implementation, the compiler
pipeline's direct mapping caller and target link, their CMake link dependencies,
the mapping unit tests, this ExecPlan, and the existing mapping entry in
`CHANGELOG.md`. It must not remove or redesign `QCOProgram::placeAndRoute`;
pipeline integration is owned by a later task. No other worktree may be
modified, and no GitHub action is authorized.

## Plan of Work

First, add one public mapping factory taking `const CompilerTarget&`. Link the
QCO transforms library and mapping unit test to `MQTCompilerTarget` without
creating a dependency cycle. At the existing high-level coupling-input boundary
in `Programs.cpp`, construct a validated `CompilerTarget` and call the
target-taking factory directly. Retain the pass options so benchmark code can
construct a pass directly for a given target.

In `Mapping.cpp`, remove `AugmentedDevice` and store `CompilerTarget` directly.
Adapt every algorithmic query to the target's dense-vertex API: `numQubits`,
`areAdjacent`, `distanceBetween`, `forEachNeighbour`, and `maxDegree`. For the
F-graph, construct dense vertices with `llvm::seq(target.numQubits())`. Do not
call `supports`, inspect operations, or filter neighbours by native-gate
direction.

Before mutation, recursively walk the entry function once. Reject any scalar or
tensor allocation outside the entry function's top-level body with an
operation-local diagnostic. Reject every non-barrier `UnitaryOpInterface` with
more than two input qubits and explain that it must be decomposed first. During
the same discovery, record top-level allocation handles and whether two-qubit
routing is required. Traverse each recorded tensor chain once and retain the
operations for placement. These records become the initial routing wires, drive
the sparse preview decision, and replace allocations without another discovery
walk.

Extend computation discovery to append top-level scalar allocation results
before tensor extracts. Extend placement to replace scalar allocations with
their assigned static values and erase them. Tensor allocation, extraction,
insertion, and deallocation behavior remains otherwise unchanged, including the
existing extraction-before-insertion restriction. This order creates one stable
program index sequence for scalar and tensor qubits and supports programs
produced by the qubit-reuse pass.

Add touched-program tracking to cold and hot SWAP insertion. After choosing an
initial layout, run one cold forward preview. Start the materialization set with
all active program indices, add every vacant index touched by preview SWAPs,
sort it, and create `qco.static` only for those entries. Use
`target.siteForVertex(layout.getHardwareIndex(program))` as each static
operation's index. Add sinks and wire metadata only for materialized vacant
programs. The following hot route must reproduce the preview and can assert that
both operands of every emitted SWAP have wires.

Keep `qco.swap` emission in its existing ordinary, symmetric form. Preserve all
current route, vote, restore, converge, and structured-region behavior. Update
comments, pass documentation, and diagnostics so they describe target topology
and the supported one-/two-qubit boundary accurately.

Refactor the mapping test fixture to hold a `CompilerTarget`. Adapt executable
checking to translate static provider IDs back to dense vertices before testing
adjacency. Preserve the existing nine-qubit grid and every current regression.
Add focused tests that demonstrate arbitrary one- and two-qubit unitary names
are routed without native-capability checks; nested higher-arity operations and
nested scalar/tensor allocations fail with diagnostics; scalar-only, mixed
scalar/tensor, and reuse-shaped programs map successfully; noncontiguous target
IDs appear in `qco.static`; and a small routed program on a large target creates
only the needed static workspace rather than one operation per target site.

Finally, extend the existing `Unreleased` place-and-route changelog entry to
mention compiler-target topology and scalar allocation support. Do not invent a
pull-request reference and do not add an upgrade note because the high-level API
has not changed yet.

## Concrete Steps

Run all commands from the repository root. Configure the task worktree with the
repository's release preset and LLVM/MLIR 22 toolchain:

    ./.agent/run.sh cmake --preset release

Build the target model, QCO transforms, and mapping test executable:

    ./.agent/run.sh cmake --build --preset release \
      --target MQTCompilerTarget MLIRQCOTransforms \
               mqt-core-mlir-unittest-mapping

Run the mapping tests directly while iterating:

    ./.agent/run.sh \
      ./build/release/mlir/unittests/Dialect/QCO/Transforms/Mapping/\
      mqt-core-mlir-unittest-mapping

Run the compiler-target and compiler-pipeline tests because the public factory
and link graph depend on their target contract:

    ./.agent/run.sh \
      ./build/release/mlir/unittests/Compiler/\
      mqt-core-mlir-unittests-compiler

Run targeted pre-commit hooks on this plan before committing:

    ./.agent/run.sh prek run --files \
      .agent/plans/1687-map-01-target-mapping.md

Use the configured compilation database for changed-source clang-tidy, then run
the repository lint session:

    ./.agent/run.sh uvx nox -s lint

Finish with:

    git diff --check
    git status --short

Record exact pass counts and any environment-limited check in this plan before
the final commit.

## Validation and Acceptance

The implementation is accepted when the mapping test binary passes every
pre-existing test, including index-switch vote-and-restore, and the following
new behavior is observed.

A `CompilerTarget` with noncontiguous provider IDs maps using dense internal
vertices, while every emitted `qco.static` index is one of the provider IDs.
Arbitrary one- and two-qubit QCO unitaries pass through the topology-only
mapper; two-qubit operations are adjacent after routing regardless of target
operation metadata or operand direction. Inserted routing operations remain
ordinary `qco.swap`.

Top-level scalar allocations, tensor allocations, mixed allocation forms, and
reuse-shaped scalar programs map and verify. Scalar and tensor allocations
inside nested regions fail before placement with a specific diagnostic. A
non-barrier unitary acting on more than two qubits fails even when nested inside
structured control flow, with a diagnostic requiring one-/two-qubit
decomposition.

For a small program on a large connected target, the number of `qco.static`
operations is smaller than the target size and equals the active qubits plus
only vacant vertices actually touched by routing. The result verifies, all
routed two-qubit operations are adjacent, and every materialized qubit has a
terminal use.

The factory remains directly callable as
`qco::createMappingPass(const CompilerTarget&, MappingPassOptions)`. The
high-level `QCOProgram::placeAndRoute` API is still present. The changelog entry
is updated without a fabricated pull-request link and `UPGRADING.md` remains
unchanged.

All focused mapping and compiler tests pass. Changed-source clang-tidy,
repository lint, and `git diff --check` pass, or an environmental boundary is
recorded with exact evidence rather than hidden.

## Idempotence and Recovery

Configuration, builds, and tests are repeatable inside the task worktree.
Generated build output stays under `build/` and caches stay under `.cache/`.
Formatting hooks may update the files in scope; inspect and include only
intentional changes. If a build fails partway, rerun the same target build
without deleting another worktree or shared resource.

The cold preview does not mutate IR. If preview routing fails, the pass reports
failure before dynamic allocation replacement. Hot routing begins only after the
exact materialized workspace is known.

No external GitHub mutation is authorized. Do not fetch into another worker's
branch, push, open or edit a pull request, comment, resolve threads, merge, or
remove any worktree.

## Artifacts and Notes

The current restack evidence is:

    HEAD parent and merge-base:
      f775395a25fddba0a3b54996416d1311bc6ebe71
    Branch delta relative to that head:
      1 commit, clean worktree before the plan evidence update
    Patch comparison:
      final range-diff differs only in this plan; non-plan stable patch IDs match
    Current target API:
      dense vertices plus provider site IDs, validated connected topology,
      cached adjacency, neighbours, distances, and maximum degree
    Current mapper:
      CompilerTarget directly, with no coupling-set overload or adapter

Fresh validation evidence after the restack:

    configure:
      MLIR_DIR=.../llvm-22.1.3/lib/cmake/mlir cmake --preset release
      passed after the cached invocation omitted the required version hint
    focused build:
      MQTCompilerTarget, MLIRQCOTransforms,
      mqt-core-mlir-unittest-mapping,
      mqt-core-mlir-unittests-compiler
      passed in 261 build steps
    mapping tests:
      27 tests from 1 suite passed
    compiler tests:
      218 tests from 8 suites passed
    clang-tidy:
      LLVM 22.1.8, explicit Xcode SDK/libc++ paths
      Mapping.cpp and test_mapping.cpp passed with scoped header filters
    targeted hooks:
      the exact eight-file MAP patch passed after formatting this plan
    repository lint:
      uvx nox -s lint passed the complete all-file hook set
    whitespace:
      git diff --check passed

Focused validation evidence after removing the legacy factory:

    live PR head:
      1b8c4a2abf63d72385cfb5bcc032a0c4c1c1dbbc
    live base and refreshed origin/main:
      f775395a25fddba0a3b54996416d1311bc6ebe71
    focused build:
      MQTCompilerTarget, MLIRQCOTransforms, MQTCompilerPipeline,
      mqt-core-mlir-unittest-mapping,
      mqt-core-mlir-unittests-compiler
      passed
    focused compiler API test:
      CompilerPipelineTest.QCOProgramOptimizationAPIs passed
    mapping tests:
      27 tests from 1 suite passed
    compiler tests:
      218 tests from 8 suites passed
    clang-tidy:
      LLVM 22.1.8, explicit Xcode SDK/libc++ paths;
      Programs.cpp, Mapping.cpp, Mapping.h, and
      test_compiler_pipeline.cpp passed without diagnostics
    targeted hooks and repository lint:
      passed
    source audit:
      every createMappingPass call passes CompilerTarget and no coupling-set
      factory declaration, definition, adapter, or helper remains
    independent review:
      approved with no required changes or new findings
    whitespace:
      git diff --check passed

## Interfaces and Dependencies

At completion, `mlir/include/mlir/Dialect/QCO/Transforms/Mapping/Mapping.h`
declares:

    std::unique_ptr<Pass>
    createMappingPass(const CompilerTarget& target,
                      MappingPassOptions options);

`MappingPass` owns a `CompilerTarget` value. It uses only `numQubits`,
`siteForVertex`, `areAdjacent`, `distanceBetween`, `forEachNeighbour`, and
`maxDegree` from that target. Operation-capability, native-gate, calibration,
duration, fidelity, and directed-locus APIs are deliberately out of scope.

`MLIRQCOTransforms` and `MQTCompilerPipeline` depend on `MQTCompilerTarget`. The
mapping test target also links the target library directly when needed. No new
third-party dependency is introduced.

Revision note: the initial plan recorded the exact pre-squash stacked base,
current vote-and-restore contract, approved MAP-01 scope, implementation
strategy, and required validation before feature edits began. This revision
records the conflict-free, patch-equivalent restack onto the merged
compiler-target squash and the review-driven removal of the compatibility
forwarder. It also records the single discovery/planning traversal, completed
implementation, and validation evidence.
