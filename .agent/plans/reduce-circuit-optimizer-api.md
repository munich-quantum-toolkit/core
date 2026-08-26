# Reduce CircuitOptimizer to shared transformations

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core currently exports circuit transformations that only one downstream
package uses, as well as transformations that no production package uses. This
change leaves only three transformations with multiple current production
consumers in MQT Core. QCEC and QMAP receive the transformations that belong to
their domains, while QMAP and QuSAT receive their own small circuit dependency
graph builders. Users can verify the result by building the updated downstream
packages against the reduced Core header and by running each package's focused
tests.

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

## Decision Log

- Decision: keep `singleQubitGateFusion`, `removeFinalMeasurements`, and
  `flattenOperations` public in Core. Rationale: each has production consumers
  in at least two repositories, and `removeFinalMeasurements` is also used by
  Core's DDSIM QDMI device. Date/Author: 2026-08-26, Codex.
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

The public optimizer interface now has exactly three methods, and the removed
domain behavior lives with its only production owner. The migration deletes more
than four thousand lines from Core without adding a new shared abstraction or
dependency.

Core's 16 optimizer tests and 51 DDSIM QDMI-device tests pass in the release
build. Complete static and shared builds install successfully, and an
out-of-tree executable links to `MQT::CoreCircuitOptimizer` and calls all three
retained methods through each installed package. The static and shared libraries
export only those three optimizer methods. QCEC's 44 migrated tests and full
554-test C++ suite pass; QMAP's 27 data structure, 37 swap-circuit, 115 hybrid,
and focused Clifford tests pass; and QuSAT's full 11-test C++ suite passes.
QCEC, QMAP, and QuSAT lint sessions pass. DDSIM's production library also builds
against this reduced Core checkout.

Final staged Core lint and test repetitions pass. Full downstream configurations
that still use the separately removed `MQT::CoreAlgorithms` or `MQT::CoreNA`
targets remain outside this issue and are recorded above rather than expanding
this migration.

## Context and Orientation

`include/mqt-core/circuit_optimizer/CircuitOptimizer.hpp` is the installed C++
interface. `src/circuit_optimizer/CircuitOptimizer.cpp` implements all current
passes. A dependency graph in this code is a vector indexed by qubit; each entry
lists pointers to the circuit operations that act on that qubit.
`test/circuit_optimizer/` contains one focused source file for each exported
transformation.

The `MQT::CoreCircuitOptimizer` CMake target remains installed because Core,
QCEC, QMAP, DDSIM, and Debugger still use the three shared transformations. The
transfer is coordinated across separate QCEC, QMAP, and QuSAT branches. Those
branches must land before or together with the Core removal so no default branch
is knowingly left unable to build.

## Plan of Work

First add private QCEC and QMAP optimizer components by moving the exact
implementations and relevant tests from Core. Replace their production call
sites while leaving calls to the three shared Core methods unchanged. Replace
QMAP and QuSAT uses of `constructDAG` with local graph construction and remove
unneeded `MQT::CoreCircuitOptimizer` links only where no shared method remains.

Then replace Core's public graph aliases and generic removal helpers with
translation-unit-local types and functions. Delete all moved and unused
implementations from `src/circuit_optimizer/CircuitOptimizer.cpp`, leaving the
existing implementations of the three retained methods unchanged. Remove the
test sources whose behavior moved downstream or has no production owner. Keep
the test sources for the three retained methods.

Finally document the complete name-to-owner migration in the unreleased
changelog and upgrade guide. Build the focused Core target and the DDSIM QDMI
device test, then build and test each changed downstream repository against the
reduced Core checkout. Run the lint session in each changed repository and
inspect every final diff for unrelated files.

## Concrete Steps

From the Core repository root, build and run the focused tests with:

    cmake --build --preset release --target mqt-core-circuit-optimizer-test
    ./build/release/test/circuit_optimizer/mqt-core-circuit-optimizer-test
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

The installed Core header must declare exactly
`singleQubitGateFusion(QuantumComputation&)`,
`removeFinalMeasurements(QuantumComputation&)`, and
`flattenOperations(QuantumComputation&, bool)`. The Core optimizer and DDSIM
QDMI device tests must pass. QCEC, QMAP, and QuSAT must compile without naming
any removed Core method or DAG alias. DDSIM and Debugger must compile against
the reduced Core API because they use only retained methods.

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

At completion, `qc::CircuitOptimizer` retains these signatures:

    static void singleQubitGateFusion(QuantumComputation& qc);
    static void removeFinalMeasurements(QuantumComputation& qc);
    static void flattenOperations(QuantumComputation& qc,
                                  bool customGatesOnly = false);

`MQT::CoreCircuitOptimizer` continues to depend publicly on `MQT::CoreIR` and
adds no dependency. QCEC and QMAP may expose no new public API unless an
existing repository pattern requires it; internal ownership is preferred.

Revision note: 2026-08-26. Created the initial implementation plan from the
refreshed production census and issue acceptance criteria. Updated it after the
four-repository implementation and cross-repository validation pass.
