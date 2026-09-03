# Support directional target gates during mapping

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core target compilation must accept devices that expose a two-qubit gate on
each topology edge in only one operand order. After this change, mapping uses
the undirected topology for reachability but prefers a layout whose ordered
operands match the device. Native synthesis repairs an unavoidable opposite
order without changing program semantics, and final conformance verifies the
exact ordered placement.

The result is visible by compiling alternating CX directions for a two-site
one-way target: compilation succeeds, every emitted two-qubit operation uses the
reported direction, and final target conformance succeeds.

## Progress

- [x] (2026-09-03) Added exact ordered operation applicability to the target
      model, QDMI adapter, typed attributes, and Python bindings.
- [x] (2026-09-03) Added cached directional mapping costs while preserving
      undirected topology traversal.
- [x] (2026-09-03) Added site-aware native synthesis and exact final conformance
      checks.
- [x] (2026-09-03) Added focused C++ and Python regressions, regenerated stubs,
      and completed the Core build and lint checks.

## Surprises & Discoveries

- Observation: QDMI site tuples are sparse calibration records and cannot also
  represent operation availability. Evidence: operations with default
  calibration have no `SiteTuple` entries even though their QDMI site list is
  complete.
- Observation: mathematically symmetric gates can still be reported in one
  syntactic operand order. Runtime-parameter RXX cannot be matrix-decomposed, so
  native synthesis must clone it with swapped inputs and restore the result
  order.
- Observation: a qubit emerging from structured control flow may have several
  possible sites. A one-qubit operation is conformant only when it is supported
  on every possible site; a direction-dependent two-qubit operation requires an
  exact ordered placement.
- Observation: target site IDs span all nonnegative `int64_t` values, including
  values reserved internally by LLVM dense containers. Scalar site-ID caches
  therefore use standard unordered sets; maximum-value round-trip tests cover
  this boundary.

## Decision Log

- Decision: keep `CompilerTarget` immutable and represent mapping policy in a
  cached `MappingTarget` wrapper. Rationale: mapping costs are derived data,
  useful as one coherent view, and should not mutate the device snapshot.
  Date/Author: 2026-09-03, contributor.
- Decision: preserve exact ordered applicability for every operation, including
  mathematically symmetric gates. Rationale: the target model records what a
  backend actually reports; any safe operand reorder is an explicit synthesis
  transformation and final conformance remains exact. Date/Author: 2026-09-03,
  contributor.
- Decision: keep Mapping synthesis-free and base its directional penalty on the
  target-wide synthesis entangler. Rationale: arbitrary non-native two-qubit
  gates are lowered through that entangler, while actual reversal belongs in
  native synthesis. Date/Author: 2026-09-03, contributor.
- Decision: use a unit routing penalty for an adjacent edge available only in
  reverse. Rationale: it models the additional local direction repair while
  leaving nonadjacent cost equal to shortest-path SWAP distance. Date/Author:
  2026-09-03, contributor.

## Outcomes & Retrospective

The Core compiler now preserves exact ordered applicability through target
materialization, QDMI snapshots, mapping, native synthesis, and conformance. The
focused Compiler, MQT IR, Mapping, NativeSynthesis, and Python MLIR suites pass.
Generated stubs, repository lint, C++ lint, documentation, and
`git diff --check` also pass. Regression coverage includes one-way entanglers,
ambiguous structured-control-flow sites, and the full nonnegative site-ID
domain.

## Context and Orientation

`mlir/include/mlir/Compiler/Target.h` and `mlir/lib/Compiler/Target.cpp` define
the immutable device snapshot. An operation may be unrestricted or may list
exact ordered physical site tuples. `SiteTuple` remains calibration-only.

`mlir/include/mlir/Compiler/MappingTarget.h` and
`mlir/lib/Compiler/MappingTarget.cpp` form a cheap wrapper around that snapshot.
For each explicit topology edge they cache whether the synthesis entangler is
native in the forward order, reverse order, or both. Mapping uses the undirected
edge to move qubits and the cached ordered cost to select a layout.

`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp` implements placement and
routing. Its lookahead window must retain semantic operand order while its
pair-block bookkeeping remains order-independent.

`mlir/lib/Dialect/QCO/Transforms/NativeSynthesis/TargetSynthesis.cpp` lowers
operations after mapping. It derives each linear qubit value's static target
site through structured control flow, checks exact native support, and either
keeps, reorders, or decomposes an operation. The final conformance pass uses the
same ordered site facts.

`mlir/lib/Compiler/QDMIAdapter.cpp` snapshots device data. QDMI operation site
lists become applicability; per-site duration and fidelity differences become
sparse calibration tuples. The MQT dialect attribute files serialize both states
without losing an explicit empty applicability list.

## Plan of Work

First, complete the compiler-target representation and its typed MLIR attribute.
Validate tuple arity, distinct nonnegative sites, known target sites, and
calibration references. Cache one- and two-site applicability for constant
ordered queries, while retaining exact tuple matching for higher arities.

Second, wrap the target in `MappingTarget`. Construct its adjacent direction
costs once, proxy the topology operations required by Mapping, and change only
goal, heuristic, placement delegation, and advance checks. Preserve the merged
window traversal and use semantic operand order recovered from each unitary's
outputs.

Third, make native synthesis site-aware. Propagate static sites to a fixed point
through QCO and SCF structured operations. Reject ambiguous or nonadjacent
placements before rewriting. Reverse asymmetric basis synthesis mathematically;
for a symmetric operation supported only in reverse, clone it with swapped
operands and map its outputs back, which also supports runtime parameters.

Finally, validate the layers independently and together and regenerate the
Python stubs through the repository's Nox session.

## Concrete Steps

Run all commands from the repository root. Set `MLIR_DIR` to the directory that
contains `MLIRConfig.cmake` for MLIR 23.1 or newer, then configure Core:

    cmake --preset release

Build and run the focused binaries:

    cmake --build build/release --target \
      mqt-core-mlir-unittests-compiler \
      mqt-core-mlir-unittest-mqt-ir \
      mqt-core-mlir-unittest-mapping \
      mqt-core-mlir-unittest-target-synthesis
    build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler
    build/release/mlir/unittests/Dialect/MQT/IR/mqt-core-mlir-unittest-mqt-ir
    build/release/mlir/unittests/Dialect/QCO/Transforms/Mapping/mqt-core-mlir-unittest-mapping
    build/release/mlir/unittests/Dialect/QCO/Transforms/NativeSynthesis/mqt-core-mlir-unittest-target-synthesis

Regenerate and test Python bindings, then run repository checks:

    uvx nox -s stubs
    uvx nox -s tests-3.13 -- test/python/test_mlir.py -q
    uvx nox -s cpp-lint
    uvx nox -s docs
    git diff --check
    uvx nox -s lint

## Validation and Acceptance

Compiler-target tests must distinguish unrestricted, explicit, and
explicit-empty applicability and preserve those states through typed MLIR and
Python. QDMI tests must preserve exact one-way and two-way site lists while
keeping calibration sparse.

Mapping tests must show that a three-site one-way chain chooses the native
orientation and inserts exactly the expected SWAP, and that the two-site search
budget can repair an opposite direction. Native synthesis tests must prove
semantic equivalence for reversed CX, exact conformance rejection before
synthesis, runtime RXX operand reordering without a synthesis basis, and safe
failure for ambiguous structured-control-flow sites.

The compiler pipeline test must compile alternating CX directions and verify
that every final two-qubit operation is supported on its exact static sites. The
C++ linter, Python lint, strict documentation, generated-stub check, and
`git diff --check` must pass without LCOV exclusions.

## Idempotence and Recovery

Source edits, formatting, configuration, builds, and tests are repeatable.
Preserve unrelated changes and do not modify another task's worktree. If
generated stubs differ, rerun the repository's `stubs` Nox session instead of
editing them by hand. If a target cannot supply a global synthesis basis, direct
symmetric native reordering may still proceed, but directional mapping falls
back to the ordinary topology because there is no single entangler whose
direction can safely represent every non-native operation.

Revision note (2026-09-03): Retained only Core design, recovery, and validation
information.
