# Prune test- and simulator-specific DD helpers

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core v4 should expose decision-diagram (DD) operations that serve several
production consumers or form an intentional user API. After this change, Core
will no longer install random DD-state generators that were introduced for
tests, and Core will no longer own the recursive unitary-construction strategy
used by MQT DDSIM's `UnitarySimulator`. Sequential circuit-functionality
construction and the Python conveniences for zero, basis, GHZ, W, and dense
vector states remain available.

The result is visible in three ways. C++ code can no longer include the removed
declarations from `dd/StateGeneration.hpp` or call
`dd::buildFunctionalityRecursive`. Python's `build_unitary` and
`build_functionality` functions no longer accept a `recursive` argument. The DD
and Python tests continue to pass for all retained behavior.

## Progress

- [x] (2026-08-26 12:18Z) Read issue #2102, repository policy, the DD
  implementation, bindings, tests, history, and direct consumers.
- [x] (2026-08-26 12:18Z) Check open Core and DDSIM issues and pull requests for
  source conflicts and ownership constraints.
- [x] (2026-08-26 12:28Z) Remove Core's random test-state generators and their
  self-tests.
- [x] (2026-08-26 12:28Z) Remove recursive functionality construction from
      Core's C++ and Python APIs while preserving sequential construction.
- [x] (2026-08-26 12:28Z) Update the upgrade guide and generated Python stubs.
- [x] (2026-08-26 12:28Z) Build and run focused C++ and Python tests, then run
  repository lint.
- [x] (2026-08-26 12:38Z) Review the final diff, complete an independent
  correctness and migration review, and record the downstream DDSIM merge
  prerequisite.
- [x] (2026-08-26 12:53Z) Internalize recursive construction in
      munich-quantum-toolkit/ddsim#975 and validate it against this Core tree.

## Surprises & Discoveries

- Observation: `generateExponentialState`, `generateRandomState`, and
  `GenerationWireStrategy` have no Core consumer other than their own tests. The
  exponential helper previously supplied approximation tests, but Core removed
  approximation support in pull request #2154. Evidence: a repository search on
  Core `main` at `1c8f61ae2` and the history of pull requests #975, #985, and
  #2154.
- Observation: `buildFunctionalityRecursive` has two current production entry
  points: MQT DDSIM's `UnitarySimulator` and Core's untested Python `recursive`
  options. Evidence: `src/UnitarySimulator.cpp` in MQT DDSIM and
  `bindings/dd/register_dd.cpp` in Core.
- Observation: no open Core pull request changes the DD source, headers,
  bindings, or tests in this plan. Several open pull requests change
  `UPGRADING.md`, so that file can require a routine rebase conflict resolution.
- Observation: `makeGHZState` and `makeWState` back tested Python methods on
  `DDPackage`. Removing them would replace an O(n)-sized DD constructor with a
  dense state vector or a simulated circuit for Python users.
- Observation: The release build tree retained a unity source from the branch
  that was checked out before this work. Regenerating it with
  `cmake --preset release` removed the stale source reference. Evidence: the
  first build referred to `src/ir/OpenQASMSerializer.cpp`, which does not exist
  on this branch; the regenerated build completed.
- Observation: The recursive implementation moved to DDSIM had untested edge
  cases. An empty circuit with a nonzero qubit count reached `log2(0)`, while
  the one-operation fast path skipped output-permutation, ancillary-qubit, and
  garbage-qubit correction. The DDSIM migration delegates zero- and
  one-operation circuits to sequential construction and tests the shared
  boundary behavior.
- Observation: The retained Python binding for `build_functionality` did not
  keep its `DDPackage` argument alive. A matrix DD built from a temporary
  package became dangling and aborted when read. The binding now attaches the
  package lifetime to the returned DD, and the retained-path test checks that
  the result acquires a package reference.
- Observation: DDSIM's full C++ binary passes 116 of 117 tests locally. The
  unrelated `StochNoiseSimTest.CheckQubitOrder` deterministically produces 943
  expected shots, below its tolerance cutoff of 950, and fails identically in
  three isolated reruns. All five `UnitarySimTest` tests pass.

## Decision Log

- Decision: Delete the random DD-state generators and their tests instead of
  moving them into Core test support. Rationale: no remaining Core test needs
  these fixtures, so a test-only copy would be dead code. Date/Author:
  2026-08-26 / Codex.
- Decision: Retain `makeGHZState` and `makeWState`, including their Python
  methods and tests. Rationale: these are small, efficient constructors for
  standard states and have intentional user-facing value. Date/Author:
  2026-08-26 / Codex.
- Decision: Retain `buildFunctionality` as the only Core circuit-to-matrix-DD
  constructor and remove both Python `recursive` switches. Rationale: Core's
  documented baseline is sequential construction; the recursive strategy is a
  DDSIM simulator choice. Date/Author: 2026-08-26 / Codex.
- Decision: Treat the DDSIM migration as a merge prerequisite, not as code in
  this repository. Rationale: the cleanup tracker requires the downstream owner
  to be ready before Core removes the API. After explicit authorization, the
  migration was implemented in munich-quantum-toolkit/ddsim#975; it still must
  merge before the Core removal. Date/Author: 2026-08-26 / Codex.
- Decision: Add `nb::keep_alive<0, 2>()` to the retained Python
  `build_functionality` binding. Rationale: every matrix DD depends on the
  package that owns its nodes, and the binding must preserve that owner for the
  result's lifetime. Date/Author: 2026-08-26 / Codex.

## Outcomes & Retrospective

The Core implementation removes 608 lines and adds 32 lines across the public
headers, implementations, tests, Python binding and stub, upgrade guide, and a
retained-API Python regression. No new abstraction or dependency was needed.

The complete DD test binary passes 282 tests. The focused Python DD suite passes
13 tests on each of Python 3.11, 3.12, 3.13, and 3.14. Stub generation and the
full repository lint session pass. DDSIM now owns and tests recursive
construction in munich-quantum-toolkit/ddsim#975, and its production library
builds against this Core tree. The DDSIM pull request must merge before this
Core pull request.

## Context and Orientation

A decision diagram is a graph representation of a vector or matrix that can
share repeated subgraphs. `include/mqt-core/dd/StateGeneration.hpp` and
`src/dd/StateGeneration.cpp` construct vector DDs. Before this change, random
generators at the end of those files manufactured graphs with selected shapes;
they did not model a user-supplied state or circuit. Their only remaining Core
callers were their own tests in `test/dd/test_state_generation.cpp`.

`include/mqt-core/dd/FunctionalityConstruction.hpp` and
`src/dd/FunctionalityConstruction.cpp` convert a `qc::QuantumComputation` into a
matrix DD. `buildFunctionality` multiplies each operation into the result in
circuit order. `buildFunctionalityRecursive` groups operation DDs in a binary
tree. MQT DDSIM selects between these strategies in its `UnitarySimulator`, so
DDSIM must own the binary-tree implementation after the Core v4 boundary.

`bindings/dd/register_dd.cpp` exposes dense `build_unitary` output and the
matrix-DD `build_functionality` result to Python. Both functions previously took
an optional `recursive` Boolean. `python/mqt/core/dd.pyi` is generated from
these bindings and must not be edited by hand.

`bindings/dd/register_dd_package.cpp` exposes `makeGHZState` and `makeWState` as
`DDPackage.ghz_state` and `DDPackage.w_state`. C++ and Python tests cover both
constructors. These files remain unchanged.

## Milestones

### Milestone 1: Establish the ownership boundary

Trace each named API through Core, MQT repositories, current open work, and
repository history. This milestone is complete when each API has a supported
owner or a deletion rationale, and when direct pull-request conflicts and merge
prerequisites are known. The completed review shows that only DDSIM needs the
recursive strategy, the random-shape helpers only test themselves, and GHZ/W
have intentional Python value.

### Milestone 2: Prune the Core surface

Remove the test-shape and recursive APIs from Core while leaving the retained
constructors unchanged. Regenerate the Python stub and document every released
API break. This milestone is complete when the removed names disappear from
production headers, sources, bindings, tests, and stubs.

### Milestone 3: Prove retained behavior and coordinate DDSIM

Build Core's DD library and Python binding, then run the DD and Python tests and
full lint session. This milestone's Core work is complete with the recorded 282
C++ test passes and four 13-test Python runs. The DDSIM migration is implemented
and locally validated; release coordination now consists of merging that branch
before the Core change.

## Plan of Work

First, remove `GenerationWireStrategy` and both overload sets for
`generateExponentialState` and `generateRandomState` from
`include/mqt-core/dd/StateGeneration.hpp`. Remove their implementation and
private random-number helpers from `src/dd/StateGeneration.cpp`. Delete the
corresponding test section and now-unused includes and helpers from
`test/dd/test_state_generation.cpp`. Keep all basic, GHZ, W, and dense-vector
state construction unchanged.

Second, remove the public `buildFunctionalityRecursive` declaration and its
implementation, including the private recursive helper. Remove the now-unused
standard-library includes. Change both Python binding lambdas to call
`buildFunctionality` directly, delete the `recursive` arguments, and adjust the
docstrings.

Third, add one `UPGRADING.md` section that names every removed C++ and Python
surface and directs recursive-unitary users to MQT DDSIM. Generate
`python/mqt/core/dd.pyi` through the repository's stub session. Add the
`CHANGELOG.md` entry only when a pull request number exists because repository
policy requires a pull-request link and this plan does not authorize opening a
pull request.

Finally, build the DD library and bindings, run the focused DD and Python tests,
run the stub session and full lint, and inspect the diff. Search the tree again
for every removed name. Record any check that cannot run and its exact failure.

## Concrete Steps

Run all commands from the repository root.

1. Edit the named headers, sources, tests, binding, and upgrade guide with a
   focused deletion-only implementation.
2. Regenerate the release build graph and build the relevant C++ targets:

       cmake --preset release
       cmake --build build/release --target mqt-core-dd mqt-core-dd-test

3. Run the focused C++ tests:

       ./build/release/test/dd/mqt-core-dd-test --gtest_filter='StateGenerationTest.*:DDFunctionality.*:Parameters/DDFunctionality.*'
4. Regenerate stubs and run focused Python tests:

       uvx nox -s stubs
       uvx nox -s tests -- test/python/dd/test_dd_package.py test/python/dd/test_vector_dds.py

5. Run repository checks:

       uvx nox -s lint
       git diff --check
       git status --short

## Validation and Acceptance

The change is accepted when the DD target and Python bindings compile; every
retained state-generation and sequential-functionality test passes; Python can
still build a unitary, a matrix DD, GHZ states, and W states; and the generated
stub has no `recursive` parameter on `build_unitary` or `build_functionality`.

A repository search must find none of `generateExponentialState`,
`generateRandomState`, `GenerationWireStrategy`, or
`buildFunctionalityRecursive` outside this historical plan and upgrade text. The
installed public headers must expose `makeZeroState`, `makeBasisState`,
`makeGHZState`, `makeWState`, `makeStateFromVector`, and `buildFunctionality`.

Before the Core pull request merges, munich-quantum-toolkit/ddsim#975 must merge
first. It replaces the call to `dd::buildFunctionalityRecursive` with a
DDSIM-owned private implementation and tests sequential-versus-recursive
equivalence for empty and one-operation circuits, virtual swaps,
non-power-of-two operation counts, layouts and output permutations, ancillary
qubits, garbage qubits, and root reference counts.

## Idempotence and Recovery

The edits, searches, build commands, and tests are safe to repeat. Stub
generation is deterministic. The work is isolated on a branch created from
`origin/main`; unrelated branches and user changes must not be reset or
overwritten. If another v4 pull request changes `UPGRADING.md`, rebase and keep
both independent migration sections.

## Artifacts and Notes

The initial source-conflict query found no open pull request that changes the
implementation files in this plan. Relevant coordination items are Core issues
`#2085`, `#2102`, `#2103`, and `#2107`; Core pull requests `#2077` through
`#2080`; and DDSIM issue `#201`. Only Core issue `#2102` directly owns this
removal. Core issue `#2103` owns the wider simulator-API consolidation and must
not be folded into this diff.

Validation completed on 2026-08-26:

    cmake --preset release
    cmake --build build/release --target mqt-core-dd mqt-core-dd-test
    ./build/release/test/dd/mqt-core-dd-test
    # 282 tests passed.

    uvx nox -s stubs
    uvx nox -s tests -- test/python/dd/test_dd_package.py test/python/dd/test_vector_dds.py
    # 13 tests passed on each of Python 3.11, 3.12, 3.13, and 3.14.

    uvx nox -s lint
    git diff --check
    # Both passed.

DDSIM validation completed from the DDSIM repository root, with
`MQT_CORE_SOURCE_DIR` set to the MQT Core repository root:

    cmake --preset release
    cmake --build build/release --target mqt-ddsim-test
    ./build/release/test/mqt-ddsim-test --gtest_filter='UnitarySimTest.*'
    # 5 tests passed.

    cmake -S . -B build/core-2102 -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_MQT_DDSIM_TESTS=OFF -DBUILD_MQT_DDSIM_CLI=OFF \
      -DFETCHCONTENT_SOURCE_DIR_MQT-CORE="${MQT_CORE_SOURCE_DIR}"
    cmake --build build/core-2102 --target mqt-ddsim
    # DDSIM built against the Core tree with the recursive API removed.

    uvx nox -s lint
    # Passed.

## Interfaces and Dependencies

At the end of this plan, the retained C++ interfaces are:

    dd::VectorDD dd::makeZeroState(size_t, dd::Package&, size_t);
    dd::VectorDD dd::makeBasisState(size_t, const std::vector<bool>&, dd::Package&, size_t);
    dd::VectorDD dd::makeBasisState(size_t, const std::vector<dd::BasisStates>&, dd::Package&, size_t);
    dd::VectorDD dd::makeGHZState(size_t, dd::Package&);
    dd::VectorDD dd::makeWState(size_t, dd::Package&);
    dd::VectorDD dd::makeStateFromVector(const dd::CVec&, dd::Package&);
    dd::MatrixDD dd::buildFunctionality(const qc::QuantumComputation&, dd::Package&);

The retained Python functions are `build_unitary(qc)` and
`build_functionality(qc, dd_package)`. No new library or external dependency is
required.

Revision note (2026-08-26): Created the plan from the issue, current source,
repository history, open Core work, and current DDSIM ownership evidence. After
implementation, recorded the exact scope, stale-build recovery, validation
results, and downstream merge prerequisite.
