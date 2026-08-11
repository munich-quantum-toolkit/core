# Make QCO DD qubit disposal scalable and semantically complete

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

QCO simulation can deallocate a separable qubit, but currently expands the
entire vector and rejects entangled qubits. The first goal is to detect and
remove separable wires directly on the decision diagram, preserving compact
states. The second goal is to determine and implement the correct representation
for discarding entangled wires, whose result is a mixed quantum state rather
than a statevector.

## Progress

- [x] (2026-08-11 09:30Z) Located vector and matrix DD reduction primitives and
  confirmed that explicit density-matrix support was previously removed.
- [x] (2026-08-11 10:42Z) Prototyped DD-native basis projections with level
  removal and canonical-subgraph separability detection.
- [x] (2026-08-11 10:45Z) Replaced dense statevector deallocation and added a
      compact 41-qubit regression test; focused separable and entangled tests
      pass.
- [x] (2026-08-11 11:20Z) Added an explicit density-simulation path backed by
      `MatrixDD`, without changing the existing functionality-matrix semantics.
- [x] (2026-08-11 11:42Z) Implemented density evolution, physical partial trace,
      measurement, reset, sampling, and Python bindings with mixed-state tests.
- [x] (2026-08-11 12:05Z) Regenerated Python stubs, passed all 148 QCO utility
      tests and all 8 Python DD tests, and passed the full repository lint
      session.

## Surprises & Discoveries

- Observation: `dd::Package` provides `reduceGarbage` for vector and matrix DDs
  and `partialTrace` for matrix DDs, although the repository changelog records
  removal of the former density-matrix subsystem. Evidence:
  `include/mqt-core/dd/Package.hpp` and `src/dd/Package.cpp`.
- Observation: Projecting the removed wire onto zero and one yields vector DDs
  with the same canonical node exactly when both nonzero branches are
  proportional. Rebuilding nodes above the wire with decremented indices then
  removes the dimension without a dense vector. Evidence: the 41-qubit
  allocation/deallocation regression completes in roughly 1 ms while retaining
  the entangled-discard diagnostic.

## Decision Log

- Decision: Never approximate partial trace by choosing one pure-state branch.
  Rationale: that changes observable probabilities and is not a semantics-
  preserving implementation of entangled deallocation. Date/Author: 2026-08-11,
  Codex.
- Decision: Prototype against existing `reduceGarbage` and DD node structure
  before adding a new public state type. Rationale: the existing primitives may
  solve separable removal without dense expansion, while mixed-state support is
  a distinct API decision. Date/Author: 2026-08-11, Codex.
- Decision: Use canonical projected-subgraph identity rather than
  `reduceGarbage` for pure-state deallocation. Rationale: `reduceGarbage`
  intentionally combines measurement magnitudes and retains the garbage DD
  level, whereas deallocation must prove separability, preserve the remaining
  pure state, and renumber higher wires. Date/Author: 2026-08-11, Codex.
- Decision: Expose density simulation through additive `simulateDensity` and
  `sampleDensity` APIs while retaining `MatrixDD` as the storage type.
  Rationale: the existing `buildFunctionality` matrix evolves by left
  multiplication, whereas density matrices require `U rho U†`; an internal
  wrapper keeps those identical storage types semantically distinct.
  Date/Author: 2026-08-11, Codex.

## Outcomes & Retrospective

DD-native pure-state deallocation and additive density simulation are complete.
The density tests cover a partially entangled state, physical partial trace,
subsequent unitary evolution, statistical sampling, reset, and measurement-fed
classical control. The release build, generated Python stubs, 148 native QCO
utility tests, 8 Python DD tests, and repository lint all pass.

## Context and Orientation

`deallocateWire` in `mlir/lib/Dialect/QCO/Utils/DDFunctionality.cpp` calls
`VectorDD::getVector`, tests factorization in dense amplitudes, and rebuilds the
remaining vector. A decision diagram is a graph that shares repeated subgraphs;
expanding it can require exponentially more memory. A pure state is described by
one vector. Discarding half of an entangled state yields a density matrix, which
represents a statistical mixture and cannot in general be converted back to one
vector.

`include/mqt-core/dd/Package.hpp` and `src/dd/Package.cpp` contain graph-level
`reduceGarbage` and matrix `partialTrace` operations. Tests in
`test/dd/test_package.cpp` demonstrate their existing behavior. QCO utility
tests are in `mlir/unittests/Dialect/QCO/Utils/test_dd_functionality.cpp`.

## Plan of Work

First create focused tests for separable deallocation on compact product states
with enough qubits that dense expansion would be impractical. Prototype a
graph-level extraction using current package primitives or a new focused
`Package` operation. It must determine whether the selected wire factors from
the remainder, return the reduced vector DD, preserve normalization, and
renumber higher wires.

Then prototype mixed-state evolution using matrix DDs as density operators.
Verify the full lifecycle: build `|psi><psi|`, partial-trace a selected wire,
apply a unitary as `U rho U†`, measure and sample from the diagonal, and reset.
Only promote this to public QCO APIs if ownership, performance, and result types
are coherent. Because existing `simulate` returns `VectorDD`, mixed-state
support likely requires a separate result/API rather than changing that return
type silently.

## Concrete Steps

From the repository root, inspect and run existing DD package tests:

    ./.agent/run.sh cmake --build --preset release --target mqt-core-dd-test mqt-core-mlir-unittest-qco-utils
    ./build/release/test/dd/mqt-core-dd-test --gtest_filter='Package.*Garbage*:Package.*PartialTrace*'

Run the QCO utility suite after every production change:

    ./build/release/mlir/unittests/Dialect/QCO/Utils/mqt-core-mlir-unittest-qco-utils

Finish with:

    ./.agent/run.sh uvx nox -s lint

## Validation and Acceptance

DD-native deallocation is accepted only if a compact high-qubit product-state
test completes without dense expansion and matches the expected reduced state.
It must continue rejecting an entangled discard unless a mixed-state API is
implemented. Mixed-state support is accepted only with Bell-state partial-trace
tests showing a maximally mixed remaining qubit, subsequent gate evolution,
measurement, reset, and sampling.

## Idempotence and Recovery

Prototypes and tests are local and repeatable. Keep any mixed-state API additive
so the existing vector API remains usable. If the prototype cannot meet the
acceptance criteria, remove prototype-only production code, retain the evidence
in this plan, and preserve the explicit semantic rejection.

## Artifacts and Notes

The current dense implementation begins with `state.getVector()` in
`deallocateWire`. The existing DD package exposes `reduceGarbage(vEdge&, ...)`
and `partialTrace(mEdge, ...)`; their exact normalization contracts must be
verified from their tests before reuse.

## Interfaces and Dependencies

Prefer extending `dd::Package` with a narrowly documented vector factorization
operation if existing primitives are insufficient. Any mixed-state result must
be a distinct documented type or API. Do not reintroduce removed density-matrix
code wholesale or add a dependency without maintainer-level design evidence.

Revision note: Initial plan created for graph-native separable disposal and a
proof-driven mixed-state feasibility assessment.

Revision note (2026-08-11): Recorded the completed additive density API and
final native, Python, stub-generation, build, and lint validation.
