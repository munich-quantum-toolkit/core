# Remove CircuitOptimizer and move generic transformations to CoreIR

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core currently installs a separate shared circuit-optimizer library for only
three transformations. This change removes that library. The generic in-place
transformations for flattening compound operations and removing final
measurements become `QuantumComputation` member functions in `MQT::CoreIR`. QCEC
and QMAP each receive their own single-qubit gate-fusion function because they
are its only production users and already contain the helper machinery it needs.
Users can verify the result by running the CoreIR tests and by building the five
downstream packages against this Core branch.

## Progress

- [x] (2026-08-26 13:58Z) Refreshed the public API and production-consumer
  census across all repositories listed by the Munich Quantum Toolkit.
- [x] (2026-08-26 13:58Z) Created clean branches from each repository's current
  default branch without modifying unrelated working trees.
- [x] (2026-08-26 15:12Z) Moved QCEC's six equivalence-checking transformations
      and 44 focused tests into a private QCEC component.
- [x] (2026-08-26 15:12Z) Moved QMAP's three mapping transformations and 13
  focused tests into `MQT::QMapDS`, and added its local dependency graph
  builder and regression test.
- [x] (2026-08-26 15:12Z) Replaced QuSAT's use of Core's public dependency graph
      aliases with private local types and construction, covered by a
      multi-qubit regression test.
- [x] (2026-08-26 15:12Z) Reduced the Core header and implementation to the
  three shared transformations and their private helpers.
- [x] (2026-08-26 15:12Z) Documented every removed name and migration owner in
  `CHANGELOG.md` and `UPGRADING.md`.
- [x] (2026-08-26 16:19Z) Ran the final staged Core lint and repeated the 16
  optimizer and 51 DDSIM QDMI-device tests successfully.
- [x] (2026-08-26 16:19Z) Installed complete static and shared Core builds and
  compiled and ran an out-of-tree consumer of `MQT::CoreCircuitOptimizer`
  against each installed package.
- [x] (2026-08-27 10:40Z) Audited the reviewer proposal against every MQT
  repository and confirmed that Core, QCEC, QMAP, DDSIM, Debugger, and QuSAT
  are the complete migration set.
- [x] (2026-08-27 10:47Z) Merged current `origin/main` into the Core pull
  request branch and preserved both sides of the changelog and upgrade-guide
  updates.
- [x] (2026-08-27 11:07Z) Moved flattening and final-measurement removal into
  `MQT::CoreIR`, moved their tests into the IR test target, and removed the
  optimizer library, installed header, export, and wheel dependency.
- [x] (2026-08-27 11:07Z) Removed Core's fusion tests after porting all seven
      structural cases to QCEC and QMAP and all four DD-equivalence cases to
      QCEC.
- [x] (2026-08-27 11:12Z) Built and tested Core and all five downstream
  migrations, ran each repository's lint hooks, and compiled every affected
  production target with warnings as errors.

## Surprises & Discoveries

- Observation: only QuSAT names the public `CircuitOptimizer::DAG` alias; QMAP
  consumes the graph only within mapping code. Four of the five public DAG
  aliases have no production consumer. Evidence: the refreshed organization-wide
  symbol census at the default branches on 2026-08-26.
- Observation: Core production code calls only `removeFinalMeasurements`.
  `singleQubitGateFusion` also has Core DD test consumers. Evidence:
  `rg 'CircuitOptimizer::'` outside `src/circuit_optimizer` and
  `test/circuit_optimizer`.
- Observation: full reduced-Core configurations of QCEC and DDSIM currently stop
  at their unrelated use of the already-removed `MQT::CoreAlgorithms` target,
  while QMAP's hybrid target stops at the already-removed `MQT::CoreNA` target.
  Evidence: explicit FetchContent configurations against this Core checkout. The
  migrated optimizer sources and all unaffected production targets compile
  against the reduced interface.
- Observation: Core's all-files lint needs the deleted test files staged;
  otherwise `prek` asks hooks to process paths that no longer exist. Evidence:
  every individual hook passed before the session reported those missing
  unstaged paths.
- Observation: QCEC and QMAP already contain dependency-graph and identity
  cleanup helpers that match the helpers used by single-qubit gate fusion.
  Evidence: `src/optimizer/EquivalenceCheckingOptimizer.cpp` in QCEC and
  `src/datastructures/CircuitOptimizations.cpp` in QMAP need only the fusion
  body and one declaration.
- Observation: all downstream projects currently pin or accept Core 3.9.x. Their
  migration pull requests must remain drafts until Core v4 is available; local
  validation uses this Core checkout directly.

## Decision Log

- Decision: keep `singleQubitGateFusion`, `removeFinalMeasurements`, and
  `flattenOperations` public in Core. Rationale: each has production consumers
  in at least two repositories, and `removeFinalMeasurements` is also used by
  Core's DDSIM QDMI device. Date/Author: 2026-08-26, Codex.
- Decision: supersede the previous decision and remove `CircuitOptimizer`.
  Rationale: distributing a shared library for three functions costs more than
  placing two generic functions on their owning IR type and placing fusion with
  its two domain owners. The reviewer proposed this ownership split, and the
  complete consumer audit found no dependency cycle or additional owner.
  Date/Author: 2026-08-27, Codex.
- Decision: move `swapReconstruction`, `removeDiagonalGatesBeforeMeasure`,
  `eliminateResets`, `deferMeasurements`, `backpropagateOutputPermutation`, and
  `elidePermutations` to QCEC. Rationale: QCEC is their only current production
  consumer and owns the related equivalence-checking contracts. Date/Author:
  2026-08-26, Codex.
- Decision: move `decomposeSWAP`, `cancelCNOTs`, and `replaceMCXWithMCZ` to
  QMAP. Rationale: QMAP is their only current production consumer and owns the
  mapping behavior. Date/Author: 2026-08-26, Codex.
- Decision: give QMAP and QuSAT local dependency graph builders instead of a new
  shared abstraction. Rationale: the graph contains pointers into a mutable
  `QuantumComputation`, is not a transformation, and has only two small
  consumers with different ownership. Date/Author: 2026-08-26, Codex.
- Decision: remove `collectBlocks`, `collectCliffordBlocks`, and the public
  generic removal helpers. Rationale: the production census found no consumer;
  private identity removal remains only where the retained transformations need
  it. Date/Author: 2026-08-26, Codex.

## Outcomes & Retrospective

The first milestone reduced the public optimizer interface to three methods and
moved domain behavior to its production owners. That migration deleted more than
four thousand lines from Core without adding a new abstraction or dependency.

Core's 16 optimizer tests and 51 DDSIM QDMI-device tests pass in the release
build. Complete static and shared builds install successfully, and an
out-of-tree executable links to `MQT::CoreCircuitOptimizer` and calls all three
retained methods through each installed package. The static and shared libraries
export only those three optimizer methods. QCEC's 44 migrated tests and full
554-test C++ suite pass; QMAP's 27 data structure, 37 swap-circuit, 115 hybrid,
and focused Clifford tests pass; and QuSAT's full 11-test C++ suite passes.
QCEC, QMAP, and QuSAT lint sessions pass. DDSIM's production library also builds
against this reduced Core checkout.

Final staged Core lint and test repetitions passed for that milestone. The final
milestone removes the last optimizer target and validates the member API across
the five downstream packages. Core's warning-clean release build passes all 298
CoreIR tests, including explicit default-constructed empty-circuit contracts,
and all 51 DDSIM QDMI-device tests. Fresh static and shared package installs
contain the two member declarations and no optimizer header, library, or
exported CMake target; a minimal out-of-tree `MQT::CoreIR` consumer compiles,
links, and runs against both packages.

QCEC passes its 20 focused optimizer tests and all 573 C++ tests. QMAP passes 26
circuit-optimization, 40 QMapDS, and 11 NASP tests. QuSAT passes all 11 tests.
DDSIM passes all 28 affected simulator tests and 116 of 117 full-suite tests;
the single stochastic tolerance miss is unrelated and reproducible without this
migration. Debugger passes all 149 tests. Each downstream migration passes lint
and a warnings-as-errors build. Full downstream configurations that still use
the separately removed `MQT::CoreAlgorithms`, `MQT::CoreNA`, or legacy OpenQASM
interfaces use temporary, out-of-tree compatibility shims or a stacked draft;
those migrations remain outside this issue rather than expanding its scope.

## Context and Orientation

`include/mqt-core/circuit_optimizer/CircuitOptimizer.hpp` is the installed C++
interface that this milestone removes.
`src/circuit_optimizer/CircuitOptimizer.cpp` contains the two generic
implementations that move into `src/ir/` and the fusion implementation that
moves to QCEC and QMAP. A dependency graph is a vector indexed by qubit; each
entry lists pointers to the circuit operations that act on that qubit.
`test/circuit_optimizer/` contains the focused tests to move or delete.

`include/mqt-core/ir/QuantumComputation.hpp` declares the central circuit type,
and `src/ir/CMakeLists.txt` already collects every source below `src/ir/` into
`MQT::CoreIR`. The Core pull request can remove the optimizer target before the
downstream changes merge because each downstream default branch remains pinned
to Core 3.9.x. Draft downstream pull requests will document their Core v4
dependency and merge after that release.

## Plan of Work

First declare `QuantumComputation::flattenOperations(bool)` and
`QuantumComputation::removeFinalMeasurements()` in
`include/mqt-core/ir/QuantumComputation.hpp`. Move their exact implementations
and private helpers to a source below `src/ir/`. Move their tests into
`test/ir/`, change calls to member syntax, and update the Core DDSIM QDMI device
to use the member function.

Then delete the fusion implementation and its Core tests after QCEC and QMAP
contain equivalent focused tests. Remove the optimizer header, source target,
test target, CMake subdirectories, installed export, wheel dependency, and all
remaining Core links to `MQT::CoreCircuitOptimizer`. Update the changelog and
upgrade guide to list the member-call replacements and downstream fusion owners.

Finally build the CoreIR and QDMI DDSIM device tests, install static and shared
Core packages, and confirm that no optimizer target or header remains. Build and
test draft QCEC, QMAP, QuSAT, DDSIM, and Debugger migrations against this Core
checkout. Run lint in each changed repository and inspect every final diff for
unrelated files.

## Concrete Steps

From the Core repository root, build and run the focused tests with:

    cmake --build --preset release --target mqt-core-ir-test
    ./build/release/test/ir/mqt-core-ir-test
    cmake --build --preset release --target mqt-core-qdmi-ddsim-device-test
    ./build/release/test/qdmi/devices/dd/mqt-core-qdmi-ddsim-device-test
    uvx nox -s lint

From each changed downstream repository root, configure if necessary and run:

    cmake --preset release
    cmake --build --preset release
    ctest --preset release
    uvx nox -s lint

Configure downstream repositories so their Core dependency resolves to the
reduced Core checkout when validating the final interface. Record exact target
names and concise pass or failure output in this plan after execution.

## Validation and Acceptance

The installed `QuantumComputation` header must declare
`removeFinalMeasurements()` and `flattenOperations(bool)`, and both methods must
preserve all existing behavior. No installed optimizer header, CMake target, or
shared library may remain. The CoreIR and DDSIM QDMI device tests must pass.
QCEC and QMAP must own and test fusion locally. QCEC, QMAP, QuSAT, DDSIM, and
Debugger must compile without naming `CircuitOptimizer` or
`MQT::CoreCircuitOptimizer`.

The migration documentation must list each removed method, state whether it
moved or has no replacement, and name the owning package for moved behavior. No
new dependency or generic cross-package optimizer abstraction is accepted.

## Idempotence and Recovery

All source edits are repeatable. Each repository uses a separate branch from its
current default branch, so an incomplete migration can be discarded without
changing another task's checkout. Build directories are generated artifacts and
are not committed. If downstream configuration has cached another Core checkout,
remove only that repository's generated build directory or pass an explicit Core
source/package path; never modify another working tree.

## Artifacts and Notes

The refreshed census found current production references in Core, QCEC, QMAP,
DDSIM, Debugger, and QuSAT. No other repository from the official MQT list has a
production reference. No open Core pull request changes the optimizer header,
implementation, tests, or CMake target at the start of this work.

## Interfaces and Dependencies

At completion, `qc::QuantumComputation` provides these signatures through
`MQT::CoreIR`:

    void removeFinalMeasurements();
    void flattenOperations(bool customGatesOnly = false);

`qc::CircuitOptimizer` and `MQT::CoreCircuitOptimizer` no longer exist. QCEC
keeps fusion private in its existing optimizer component. QMAP declares
`qmap::singleQubitGateFusion(QuantumComputation&)` in its existing circuit
optimization header because several QMAP translation units use that component.

Revision note: 2026-08-26. Created the initial implementation plan from the
refreshed production census and issue acceptance criteria. Updated it after the
four-repository implementation and cross-repository validation pass.

Revision note: 2026-08-27. Replaced the retained-library outcome with the
reviewer-approved CoreIR member API and coordinated downstream drafts.
