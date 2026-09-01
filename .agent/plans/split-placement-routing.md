# Split deterministic placement from topology routing

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core currently runs its complete place-and-route pass even when a compiler
target has all-to-all connectivity and no routing is necessary. After this
change, target compilation uses a small deterministic placement pass for such
targets. The placement pass converts dynamic scalar and tensor-backed qubits to
the first available static target sites without inserting routing qubits or
changing structured control flow. The existing mapping pass becomes responsible
only for targets with an explicit coupling graph and reuses the same allocation
rewrite.

A user can observe the split by compiling a two-qubit program for an all-to-all
target with more sites than the program uses. The result contains one
`qco.static` operation per program qubit and no routing `qco.swap` operations.
An explicit-topology target still runs the existing router and preserves its
current behavior.

## Progress

- [x] (2026-09-01 11:23Z) Inspected the current target, mapping, placement,
  tensor, pipeline, and test contracts on `main`.
- [x] (2026-09-01 11:31Z) Extracted allocation discovery, capacity checking, and
      allocation rewriting into internal utilities shared by placement and
      mapping.
- [x] (2026-09-01 11:31Z) Added the deterministic placement pass and dispatched
  non-explicit targets to it from target compilation.
- [x] (2026-09-01 11:31Z) Made the mapping pass reject non-explicit targets
  before changing the IR.
- [x] (2026-09-01 11:34Z) Added direct scalar, tensor/output, compact,
  deterministic, and failure-atomicity tests plus target-pipeline coverage.
- [x] (2026-09-01 11:45Z) Updated pass and user documentation and validated the
  focused binaries, MLIR docs, C++ lint, and full repository lint.

## Surprises & Discoveries

- Observation: On current `main`, an absent coupling topology means all-to-all
  connectivity. The `Unknown`, `AllToAll`, and `Explicit` distinction is added
  by the separate compiler-target work. Evidence: `CompilerTarget` documents an
  absent topology as all-to-all, and `CompilerTarget::areAdjacent` returns true
  for any two distinct sites when `hasExplicitTopology()` is false.
- Observation: The existing allocation rewrite also creates vacant static qubits
  because the router represents every hardware site as a program token. A
  compact deterministic layout can reuse the rewrite unchanged by containing
  only the program qubits. Evidence: `place` iterates
  `layout.nHardwareQubits()`, while `Layout::fromMapping` sizes the layout from
  the supplied mapping.
- Observation: The repository's changed-file `cpp-lint` session compares commits
  and therefore did not select this intentionally uncommitted diff. Evidence: it
  reported `No source files need checking` after its successful lint build. The
  modified translation units were instead checked directly with `clang-tidy` and
  line filters from the same `build/cpp-lint` compilation database.

## Decision Log

- Decision: Add a dedicated `place-qubits` pass and keep `place-and-route` as
  the routing pass. Rationale: Placement and routing are established compiler
  stages with different information requirements. A separate pass removes the
  topology-dependent router from all-to-all compilation. Date/Author: 2026-09-01
  / OpenAI Codex.
- Decision: Share internal allocation discovery and rewriting rather than
  invoking one pass from another. Rationale: The router must retain the wire and
  layout state returned by placement, which a nested pass invocation cannot
  expose safely. Date/Author: 2026-09-01 / OpenAI Codex.
- Decision: Assign program qubit `i` to target vertex `i` in the standalone
  placement pass. Rationale: The established discovery order is deterministic,
  compact, and does not invent a topology-dependent optimization. Date/Author:
  2026-09-01 / OpenAI Codex.
- Decision: Keep the prerequisite change compatible with the current
  all-to-all-or-explicit target model. Rationale: Unknown connectivity does not
  exist on `main`; after this prerequisite merges, the compiler-target branch
  must route `Unknown` through placement only after rejecting remaining
  non-barrier multi-site operations before placement changes the IR.
  Date/Author: 2026-09-01 / OpenAI Codex.

## Outcomes & Retrospective

Target compilation now runs compact deterministic placement for the current main
branch's all-to-all target representation and retains topology-aware mapping
only for explicit coupling graphs. Both passes use the same allocation
discovery, capacity validation, and rewrite. The public surface grows by one
target-aware pass factory; no new dependency, option, or target-model concept
was introduced.

The direct mapping binary passed all 86 tests, including scalar placement,
tensor placement with classical outputs, compact noncontiguous site selection,
and both placement and direct-mapping failure atomicity. The compiler binary
passed all 139 tests, including compact all-to-all target compilation and the
existing explicit-topology pipeline. Generated MLIR documentation, repository
format/lint, focused direct `clang-tidy`, and `git diff --check` all pass.

The only deferred integration is intentional: after the compiler-target facts
branch is rebased, its `Unknown` connectivity case must reject remaining
non-barrier multi-site operations before invoking this placement pass. That case
cannot be represented on current `main` and does not belong in this
prerequisite.

## Context and Orientation

`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp` contains the current
`MappingPass`. The pass first discovers top-level `qco.alloc` and
`qtensor.alloc` roots, replaces those dynamic roots with `qco.static` target
sites, and then routes two-qubit operations by inserting `qco.swap` operations.
The discovery code enforces the tensor lifetime shape supported by mapping: all
tensor qubits are extracted before any are inserted, and allocations occur in
the entry function body.

Placement means assigning each program qubit to one hardware site. Routing means
changing that assignment during the program so that every multi-qubit operation
acts on connected sites. A target with all-to-all connectivity needs placement
but not routing. An explicit topology is a listed set of connected site pairs
and requires the router.

`mlir/include/mlir/Dialect/QCO/Transforms/Passes.td` declares generated MLIR
passes and their documentation.
`mlir/include/mlir/Dialect/QCO/Transforms/Mapping/Mapping.h` declares the
target-aware pass factories. `mlir/lib/Compiler/TargetCompilation.cpp` builds
the target compilation pipeline. Direct mapping tests live in
`mlir/unittests/Dialect/QCO/Transforms/Mapping/test_mapping.cpp`, and end-to-end
target pipeline tests live in
`mlir/unittests/Compiler/test_compiler_pipeline.cpp`.

This worktree contains only this task. Preserve unrelated changes and do not
modify another task's worktree. This plan does not authorize any GitHub action.

## Plan of Work

In `Mapping.cpp`, move the wire bookkeeping, allocation records, discovery
routine, and allocation rewrite out of `MappingPass` into the file's anonymous
namespace. Keep the code internal. Rename the rewrite to `applyPlacement` and
pass the immutable `CompilerTarget` explicitly. The function returns the wire
state required by the existing router.

Add a generated `PlacementPass` definition in `Passes.td` and implement it in
`Mapping.cpp`. The pass obtains the entry function, discovers the dynamic
qubits, checks that the target has enough sites, constructs the identity mapping
`0..numProgramQubits-1`, applies placement, and returns. The pass must validate
all conditions before applying the first rewrite so failures leave the module
unchanged. Declare a target-aware `createPlacementPass` factory in `Mapping.h`.

Change `MappingPass` to reject a target without an explicit topology before it
discovers or changes the program. Keep its higher-arity operation diagnostic,
layout search, routing, statistics, and control-flow handling unchanged. Call
the shared `applyPlacement` utility after the router chooses its layout.

Change `populateTargetCompilationPipeline` to select `PlacementPass` when
`hasExplicitTopology()` is false and `MappingPass` otherwise. Include `Target.h`
because the pipeline now inspects the target. Current `main` defines the
non-explicit case as all-to-all. When the compiler-target facts branch is
rebased, retain this split: use compact placement for `AllToAll`, routing for
`Explicit`, and validate that an `Unknown` target has no remaining non-barrier
multi-site operation before compact placement.

Extend the mapping unit test binary with direct placement tests. Prove stable
site order, compact output on a larger all-to-all target, scalar and tensor
allocation replacement, verified output, and rejection of direct mapping for a
non-explicit target without mutation. Extend the compiler pipeline test with an
all-to-all target larger than its two-qubit program and assert that target
compilation emits only the used static sites and no routing swaps.

Update the TableGen pass description to state the exact contracts and build the
generated MLIR pass documentation. Do not add a standalone changelog entry
because MQT Core v4 target compilation is unreleased.

## Concrete Steps

Run all commands from the repository root.

After each source edit, build the focused targets:

    cmake --preset release
    cmake --build --preset release --target \
      mqt-core-mlir-unittest-mapping mqt-core-mlir-unittests-compiler

Run direct placement and routing tests first:

    ./build/release/mlir/unittests/Dialect/QCO/Transforms/Mapping/\
      mqt-core-mlir-unittest-mapping

Run the compiler tests that exercise target-pipeline dispatch:

    ./build/release/mlir/unittests/Compiler/\
      mqt-core-mlir-unittests-compiler \
      --gtest_filter='CompilerPipelineTest.*Target*'

Build generated pass documentation and then run the required checks:

    cmake --build --preset release --target mlir-doc
    uvx nox -s cpp-lint
    uvx nox -s lint

Expected focused test output ends with all selected tests passing. The final
working tree contains only the plan, pass split, tests, and pass documentation
source changes.

## Validation and Acceptance

The direct placement tests must verify the input module before the pass and the
output module after success. For a target whose site identifiers are not dense,
program qubit zero must use the first listed target site and program qubit one
must use the second. A target with extra sites must not materialize unused
`qco.static` operations. Scalar `qco.alloc`, `qtensor.alloc`, `qtensor.extract`,
`qtensor.insert`, and `qtensor.dealloc` operations must be removed according to
the existing supported tensor contract.

Running `MappingPass` directly with a non-explicit target must fail before
changing the printed module. Running it with an explicit topology must retain
the existing executable routing behavior. End-to-end target compilation for an
all-to-all target must succeed, use only the static sites needed by the program,
and contain no router-inserted swap.

After the compiler-target facts branch is rebased, an additional direct test
must prove that unknown connectivity plus a non-barrier operation on two or more
sites fails before placement changes the module. Unknown connectivity with only
single-site operations must use the same deterministic compact placement.

## Idempotence and Recovery

All inspection, configure, build, and test commands are repeatable. CMake can
regenerate the build tree after the TableGen pass changes. If a rewrite test
fails, rerun only the mapping test binary with a GoogleTest filter. Do not reset
or discard unrelated files. Use `git diff` to identify and repair only this
task's changes.

## Artifacts and Notes

Initial evidence from current `main`:

    CompilerTarget::hasExplicitTopology() == false
      means all-to-all connectivity.
    MappingPass::place(...) iterates layout.nHardwareQubits().
    Layout::fromMapping([0, 1]) creates a compact two-site layout.

Final validation evidence:

    mqt-core-mlir-unittest-mapping: 86 tests passed.
    mqt-core-mlir-unittests-compiler: 139 tests passed.
    cmake --build --preset release --target mlir-doc: passed.
    uvx nox -s cpp-lint: build passed; no uncommitted files selected.
    clang-tidy with build/cpp-lint and modified-line filters: passed.
    uvx nox -s lint: passed.
    git diff --check: passed.

## Interfaces and Dependencies

The final public C++ factory is:

    std::unique_ptr<Pass>
    createPlacementPass(const CompilerTarget& target);

The generated pass is named `PlacementPass` and uses the command-line argument
`place-qubits`. It has no options. `MappingPass` keeps its existing factory and
options. Both implementations use the existing `CompilerTarget`, `Layout`,
`WireIterator`, `TensorIterator`, QCO, QTensor, and MLIR rewrite APIs. No new
dependency is added.

Plan revision 2026-09-01: Created the initial self-contained implementation and
validation plan. The plan records the coordination boundary with the separate
compiler-target facts work because current `main` does not represent unknown
connectivity.

Plan revision 2026-09-01 11:45Z: Recorded the completed implementation,
validation workaround for an uncommitted diff, test counts, and the remaining
pull request 2218 rebase adaptation.
