# Map QCO programs to compiler targets

Status: historical implementation record.

Later mapping ownership:
[separate placement and routing](split-placement-routing.md).

## Goal and scope

After this change, callers can run the QCO `place-and-route` pass against an
immutable `mlir::CompilerTarget` instead of constructing a second graph wrapper
from a dense coupling set. The mapper uses the target's validated topology for
placement and routing, emits device-defined site identifiers in `qco.static`
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

## Constraints

- the pre-MAP mapper's `AugmentedDevice` duplicated topology storage, neighbour
  lookup, distances, and maximum degree that `CompilerTarget` already validates
  and caches. Evidence: before this task,
  `mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp` defined the wrapper,
  while `mlir/include/mlir/Compiler/Target.h` exposes `numQubits`,
  `areAdjacent`, `forEachNeighbour`, `distanceBetween`, and `maxDegree`.

- the pre-MAP mapper intentionally maintained a full permutation over all target
  vertices, even when the program used fewer qubits. Its placement phase
  therefore materialized every target site. The implemented cold routing preview
  retains the full virtual permutation while identifying only the vacant program
  indices whose qubits are touched by real SWAPs.

- pull request #1951 added branch-layout voting and restoration to
  `qco.index_switch`. That implementation is already on the task base and must
  remain the source of truth; importing the older pull request #1687 mapper
  would risk dropping those regressions.

- the compiler-target foundation merged as squash
  `f775395a25fddba0a3b54996416d1311bc6ebe71` after MAP-01 was developed on its
  pre-squash stack. Replaying the sole MAP commit onto that squash required no
  conflict resolution. Before this plan-only evidence update, `git range-diff`
  marked the old and new patches with `=`; in the final state, the non-plan
  stable patch IDs match, so the current-main target implementation was
  preserved unchanged.

- after the restack, CMake did not recognize the cached MLIR version when
  invoked without `MLIR_DIR`. Re-running the same release configuration with the
  installed LLVM 22.1.3 MLIR package path explicit succeeded without a source or
  build-system change.

- the coupling-set factory was used only by `QCOProgram::placeAndRoute`; all
  mapping tests and every other pass caller already used `CompilerTarget`.
  Moving the unavoidable coupling-to-target conversion to that high-level API
  boundary removes 50 lines from the mapping implementation and 13 lines from
  its public header without changing mapping behavior.

## Decisions

- store a cheap value copy of `CompilerTarget` in the pass. Rationale: the type
  shares immutable storage, so pass instances own a stable target contract
  without rebuilding or duplicating topology.

- use dense zero-based compiler vertices for `Layout`, A* search, and
  restoration, and translate through `CompilerTarget::siteForVertex` only when
  creating `qco.static`. Rationale: layout algorithms require dense indices,
  while output IR must retain device-defined identifiers.

- route every one-qubit unitary directly and every two-qubit unitary solely
  according to undirected target adjacency. Do not query operation support,
  native bases, calibration, or direction. Rationale: MAP-01 owns topology
  mapping only; native synthesis belongs to a separate task.

- keep the complete target-sized virtual layout, perform a deterministic cold
  preview for the chosen initial layout, and materialize only active program
  indices plus vacant indices touched by preview SWAPs. Rationale: the routing
  algorithm can use workspace without growing small programs to the complete
  target in IR.

- use one recursive discovery/planning walk to validate nested dynamic
  allocations and higher-arity unitaries, collect top-level allocation handles,
  and record whether two-qubit routing is needed. Store each discovered tensor
  chain once. Reuse these records for initial wires, sparse preview, and
  allocation replacement; the later recursive walk is mutation-only for
  structured-control extension. Rationale: unsupported input is diagnosed
  atomically without redundant whole-function diagnostic walks.

- preserve the current vote, restore, converge, and region-dispatch
  implementation unchanged except where the target topology abstraction or
  sparse-wire bookkeeping requires mechanical adaptation. Rationale: the #1951
  regressions are the current mapping contract.

- expose only the target-taking mapping factory. Convert the current high-level
  `QCOProgram::placeAndRoute` coupling input directly into a `CompilerTarget`
  inside `Programs.cpp`, where that legacy surface is already isolated, and link
  `MQTCompilerPipeline` to `MQTCompilerTarget`. Rationale: the mapping library
  should have one target-native contract and no forwarding shim; PIPE can later
  remove the remaining high-level coupling API without touching Mapping again.

## Outcome and validation

The mapper uses the target's dense topology, translates site IDs only at
static-qubit materialization, and creates only active or routing-workspace
qubits. One planning walk rejects unsupported allocations and operations before
mutation. Mapping tests, including vote-and-restore behavior, compiler tests,
changed-source clang-tidy, and lint passed.

## Code and ownership

`mlir::CompilerTarget`, declared in `mlir/include/mlir/Compiler/Target.h` and
implemented in `mlir/lib/Compiler/Target.cpp`, is an immutable target model.
Device site identifiers can be sparse or noncontiguous, but the target stores
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

Pipeline integration and removal of the coupling-only API belong to
[the target pipeline](1687-pipe-01-target-pipeline.md).

## Acceptance

The implementation is accepted when the mapping test binary passes every
pre-existing test, including index-switch vote-and-restore, and the following
new behavior is observed.

A `CompilerTarget` with noncontiguous target IDs maps using dense internal
vertices, while every emitted `qco.static` index is one of the target IDs.
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

## Interfaces

At completion, `mlir/include/mlir/Dialect/QCO/Transforms/Mapping/Mapping.h`
declares:

    std::unique_ptr<Pass>
    createMappingPass(const CompilerTarget& target,
                      MappingPassOptions options);

`MappingPass` owns a `CompilerTarget` value. It uses only `numQubits`,
`siteForVertex`, `areAdjacent`, `distanceBetween`, `forEachNeighbour`, and
`maxDegree` from that target. Operation capabilities, native gates, and
calibration metadata are deliberately out of scope.

`MLIRQCOTransforms` and `MQTCompilerPipeline` depend on `MQTCompilerTarget`. The
mapping test target also links the target library directly when needed. No new
third-party dependency is introduced.
