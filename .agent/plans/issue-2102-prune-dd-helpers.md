# Prune test- and simulator-specific DD helpers

Status: historical implementation record.

## Goal and scope

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

## Constraints

- `generateExponentialState`, `generateRandomState`, and
  `GenerationWireStrategy` have no Core consumer other than their own tests. The
  exponential helper previously supplied approximation tests, but Core removed
  approximation support in pull request #2154. Evidence: a repository search on
  Core `main` at `1c8f61ae2` and the history of pull requests #975, #985, and
  #2154.

- `buildFunctionalityRecursive` has two current production entry points: MQT
  DDSIM's `UnitarySimulator` and Core's untested Python `recursive` options.
  Evidence: `src/UnitarySimulator.cpp` in MQT DDSIM and
  `bindings/dd/register_dd.cpp` in Core.

- `makeGHZState` and `makeWState` back tested Python methods on `DDPackage`.
  Removing them would replace an O(n)-sized DD constructor with a dense state
  vector or a simulated circuit for Python users.

- The release build tree retained a unity source from the branch that was
  checked out before this work. Regenerating it with `cmake --preset release`
  removed the stale source reference. Evidence: the first build referred to
  `src/ir/OpenQASMSerializer.cpp`, which does not exist on this branch; the
  regenerated build completed.

- The recursive implementation moved to DDSIM had untested edge cases. An empty
  circuit with a nonzero qubit count reached `log2(0)`, while the one-operation
  fast path skipped output-permutation, ancillary-qubit, and garbage-qubit
  correction. The DDSIM migration delegates zero- and one-operation circuits to
  sequential construction and tests the shared boundary behavior.

- The retained Python binding for `build_functionality` did not keep its
  `DDPackage` argument alive. A matrix DD built from a temporary package became
  dangling and aborted when read. The binding now attaches the package lifetime
  to the returned DD, and the retained-path test checks that the result acquires
  a package reference.

- Historical downstream validation retained one unrelated stochastic failure
  reproduced without the migration. That run was not a full-suite pass.

## Decisions

- Delete the random DD-state generators and their tests instead of moving them
  into Core test support. Rationale: no remaining Core test needs these
  fixtures, so a test-only copy would be dead code.

- Retain `makeGHZState` and `makeWState`, including their Python methods and
  tests. Rationale: these are small, efficient constructors for standard states
  and have intentional user-facing value.

- Retain `buildFunctionality` as the only Core circuit-to-matrix-DD constructor
  and remove both Python `recursive` switches. Rationale: Core's documented
  baseline is sequential construction; the recursive strategy is a DDSIM
  simulator choice.

- Treat the DDSIM migration as a merge prerequisite, not as code in this
  repository. Rationale: the cleanup tracker requires the downstream owner to be
  ready before Core removes the API. After explicit authorization, the migration
  was implemented in munich-quantum-toolkit/ddsim#975; it still must merge
  before the Core removal.

- Add `nb::keep_alive<0, 2>()` to the retained Python `build_functionality`
  binding. Rationale: every matrix DD depends on the package that owns its
  nodes, and the binding must preserve that owner for the result's lifetime.

## Outcome and validation

The Core implementation removes 608 lines and adds 32 lines across the public
headers, implementations, tests, Python binding and stub, upgrade guide, and a
retained-API Python regression. No new abstraction or dependency was needed.

The complete DD test binary passes 282 tests. The focused Python DD suite passes
13 tests on each of Python 3.11, 3.12, 3.13, and 3.14. Stub generation and the
full repository lint session pass. DDSIM now owns and tests recursive
construction in munich-quantum-toolkit/ddsim#975, and its production library
builds against this Core tree. The DDSIM pull request must merge before this
Core pull request.

## Code and ownership

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

## Acceptance

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

## Interfaces

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
