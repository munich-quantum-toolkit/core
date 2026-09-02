# Consolidate DD gate semantics with QCO

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core currently implements every supported named quantum gate twice: once as
a QCO operation matrix and once in the low-level decision-diagram (DD) package.
After this change, QCO is the sole source of named-gate semantics. The QCO
interpreter and QIR runtime obtain canonical QCO matrices through one small
adapter and turn those matrices into DD operations. The exported `MQT::CoreDD`
library remains an MLIR-independent collection of backend-neutral DD data
structures and matrix-to-DD primitives, including in builds configured with
`BUILD_MQT_CORE_MLIR=OFF`.

Users can observe the result by running the existing QCO DD simulation and QIR
runtime tests: fixed, parameterized, controlled, and multi-qubit gates retain
their behavior. A DD-only build still configures and tests successfully, but the
obsolete public `dd::GateType`, `dd::opTo*GateMatrix`, and `dd::getGateDD`
convenience API no longer exists in the v4 C++ surface.

## Progress

- [x] (2026-09-02 23:10 CEST) Read issue #2331, its two originating #2288 review
      discussions, repository policy, and the DD/QCO/QIR build and source
      surfaces.
- [x] (2026-09-02 23:10 CEST) Chose a one-way QCO-to-DD adapter while retaining
      an independent and exported CoreDD target.
- [x] (2026-09-02 23:45 CEST) Added the shared adapter and migrated QCO DD
      execution to canonical QCO matrices.
- [x] (2026-09-02 23:45 CEST) Migrated the QIR runtime and central gate registry
      away from `dd::GateType`.
- [x] (2026-09-02 23:45 CEST) Removed DD-owned named-gate semantics and
      preserved DD-only package tests with raw test matrices.
- [x] (2026-09-03 00:30 CEST) Added independent adapter tests for asymmetric
      one-, two-, three-, and four-target matrices, including operand
      permutations, sparse controls, and noncontiguous embedding.
- [x] (2026-09-03 02:05 CEST) Updated the existing changelog entry for #2335 and
      documented the v4 API migration.
- [x] (2026-09-03 02:05 CEST) Completed focused, DD-only, full build, test,
      formatting, and lint validation and inspected the final diff.

## Surprises & Discoveries

- Observation: The QCO DD interpreter cannot rely solely on
  `UnitaryOpInterface::getUnitaryMatrix()`. That interface only sees constants
  present in IR, while `DDArgumentBindings` also supplies concrete values for
  runtime SSA parameters. The standard-gate path must continue resolving those
  bindings before calling each operation's static QCO matrix factory.
- Observation: Controlled QCO operations must keep a base matrix plus sparse
  `dd::Controls`. Materializing `CtrlOp`'s full dense matrix would grow
  exponentially with the number of controls and lose the existing DD fast path.
- Observation: QIR Runtime and JIT are internal build-tree targets, whereas
  CoreDD is installed and exported. Making CoreDD depend on a QCO target would
  give the installed static library an unexported dependency and reverse the
  intended dependency direction.
- Observation: DD-only builds and tests are intentional and documented. Moving
  CoreDD under `mlir/` would also make it unavailable when bindings and the
  current DD test subtree are configured earlier in the top-level build.
- Observation: `dd::applyGlobalPhase` does not duplicate named-gate semantics;
  it is a backend-neutral operation that mutates and returns a `VectorDD`.
  Keeping it also avoids an unrelated v4 API removal.
- Observation: `llvm::DenseMap<dd::Qubit, ...>` cannot key the two largest valid
  qubit indices because LLVM reserves those values as sentinels. The
  arbitrary-target adapter now scans its small target list instead; its
  recursion follows target levels only and adds intervening identity levels
  iteratively.
- Observation: Integration tests that build their expected DD through the new
  adapter can hide conversion and operand-order defects. Direct adapter tests
  therefore compare the specialized paths with raw DD constructors and the
  arbitrary-target path with an independently embedded dense matrix.

## Decision Log

- Decision: Keep `MQT::CoreDD` unconditional, MLIR-independent, and exported.
  Rationale: Matrix-to-DD construction is backend-neutral and useful without a
  program dialect; QCO should depend on this primitive layer, never the reverse.
  Date/Author: 2026-09-02 / Codex.
- Decision: Add one internal `MLIRQCODDAdapter` shared by the QCO interpreter
  and QIR runtime. Rationale: These are the two real consumers that translate
  canonical QCO matrices to DD nodes; sharing only this narrow conversion avoids
  linking QIR to the complete interpreter. Date/Author: 2026-09-02 / Codex.
- Decision: Remove the DD-specific operation column from `GateTable.def` and
  generate QIR dispatch directly from the existing QCO operation key. Rationale:
  A second named-gate enum is redundant once QCO owns the matrices. Date/Author:
  2026-09-02 / Codex.
- Decision: Keep measurement projector matrices private to `Package.cpp` and
  retain raw DD matrix aliases and constructors. Rationale: Projectors are DD
  implementation details, while raw matrices are the backend-neutral boundary.
  Date/Author: 2026-09-02 / Codex.
- Decision: Retain `dd::applyGlobalPhase` in `dd/Operations.hpp` while removing
  `dd::getGateDD`. Rationale: Applying a scalar phase is a raw DD primitive,
  unlike the removed enum-to-named-matrix dispatch. Date/Author: 2026-09-02 /
  Codex.
- Decision: Treat qubit-range, uniqueness, and control/target disjointness as
  verified-IR invariants at the internal adapter boundary. Rationale: QCO
  execution and generated QIR may assume valid IR; duplicating their verifiers
  in the matrix adapter would add guardrails for unreachable inputs. The adapter
  retains the existing matrix-arity check needed by dynamic custom unitaries.
  Date/Author: 2026-09-03 / Codex.

## Outcomes & Retrospective

QCO now owns every named-gate matrix used by MQT Core. One internal adapter
turns those matrices into DDs for both direct QCO interpretation and QIR
execution. CoreDD remains an installed, MLIR-independent primitive library; its
raw matrix constructors and global-phase operation remain available, while the
duplicate `dd::GateType`, named matrix formulas, and dispatch are gone.

The release build and all 3,781 discovered tests passed, with one test skipped.
Focused validation passed 173 QCO utility tests, 72 QIR runtime tests, and 155
DD tests. A separate MLIR-disabled build also passed all 155 DD tests. The
repository lint and C++ lint passed with zero format or tidy findings.

No behavior or ABI changes were needed in the QIR runtime. The deliberately
retained boundary is raw matrix-to-DD construction; moving CoreDD under MLIR or
removing backend-neutral primitives would make standalone DD builds impossible
and was therefore not part of this consolidation.

## Context and Orientation

The exported low-level library is built in `src/dd` and exposed as
`MQT::CoreDD`. `include/mqt-core/dd/GateMatrixDefinitions.hpp` and
`src/dd/GateMatrixDefinitions.cpp` currently define a DD-specific gate enum and
29 named-gate formulas. `include/mqt-core/dd/Operations.hpp` and
`src/dd/Operations.cpp` dispatch those formulas into raw constructors on
`dd::Package`.

The canonical formulas already exist on QCO standard operations declared in
`mlir/include/mlir/Dialect/QCO/IR/QCOOps.td` and implemented below
`mlir/lib/Dialect/QCO/IR/Operations/StandardGates`. Fixed gates expose static
`getUnitaryMatrix()` factories; parameterized gates expose static
`unitaryMatrix(...)` factories. Their matrix storage types live in
`mlir/Dialect/QCO/Utils/Matrix.h`.

`mlir/lib/Dialect/QCO/Utils/DDFunctionality.cpp` interprets QCO directly into a
DD. `mlir/lib/Dialect/QIR/Execution/Runtime` implements QIR quantum instruction
entry points on the same package. Both currently route named operations through
`dd::GateType`. `mlir/include/mlir/Conversion/GateTable.def` contains an `OP`
column solely to spell those DD enumerators; all other conversion consumers
ignore it.

The change is limited to the gate registry, the QCO-to-DD adapter and
interpreter, QIR runtime, DD implementation details, their direct tests, and v4
documentation. It must not expose the internal MLIR adapter in the installed
CoreDD target or change the behavior and ordering of existing gate matrices.

## Plan of Work

First, add `DDAdapter.h` and `DDAdapter.cpp` beside the existing QCO utilities
and build them as a small `MLIRQCODDAdapter` library. The adapter accepts QCO
matrix objects plus DD controls and targets, validates matrix arity, and calls
the raw `dd::Package::make*GateDD` constructors. It must support one-, two-,
three-, and arbitrary-target matrices. Sparse controls remain on the raw DD
constructors used for the one-, two-, and three-target standard operations;
arbitrary-target custom unitaries continue to use their existing uncontrolled
embedding path.

Second, migrate `DDFunctionality.cpp`. Resolve standard-operation parameters
through the existing classical environment, call the corresponding static QCO
matrix factory, and pass the result to the adapter. Reuse the adapter for
compile-time custom unitary matrices. Preserve global phase, barriers, reset,
wire remapping, and sparse controls.

Third, migrate QIR Runtime templates and generated QIS bodies to QCO operation
types. The existing gate table already contains the QCO key and target/parameter
counts, so generated bodies can instantiate operation-specific matrix factories
without another enum. Preserve the uncontrolled SWAP permutation shortcut and
generic controlled tuple ABI. Remove the obsolete DD operation column from the
gate table and mechanically update consumers that currently ignore it.

Fourth, delete DD-owned named-gate definitions and dispatch. Make measurement
projectors private package constants while retaining the backend-neutral DD
global-phase helper. Remove tests that only test deleted dispatch, migrate
integration references to the QCO adapter, and give DD-only package tests small
raw matrix fixtures where a matrix is merely input data. Delete redundant tests
that compare one production implementation of a gate formula with the other.

Finally, document the removed v4 C++ conveniences and fold the implementation
note into the existing unreleased changelog entry. Validate focused QCO DD and
QIR behavior, the DD-only configuration, the complete release build and test
suite, formatting, C++ lint, and repository lint.

## Concrete Steps

Run all commands from the repository root.

After each migration stage, search for obsolete production usage with:

    rg -n "GateType|GateMatrixDefinitions|opTo(Single|Two|Three)QubitGateMatrix|getGateDD" \
      include src mlir bindings test

At completion, matches may exist only in migration prose. Configure and test a
normal release build using the repository preset and the available MLIR 23.1
package when discovery requires it:

    cmake --preset release -DMLIR_DIR=/private/tmp/mqt-core-llvm-23.1.0/lib/cmake/mlir
    cmake --build --preset release
    ctest --preset release --output-on-failure

Configure a separate DD-only build and run its DD tests:

    cmake -S . -B build/release-no-mlir -GNinja \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_MQT_CORE_MLIR=OFF \
      -DBUILD_MQT_CORE_BINDINGS=OFF -DBUILD_MQT_CORE_TESTS=ON
    cmake --build build/release-no-mlir --target mqt-core-dd-test
    ctest --test-dir build/release-no-mlir -R mqt-core-dd-test --output-on-failure

Run final static checks with:

    git diff --check
    uvx nox -s cpp-lint
    uvx nox -s lint

## Validation and Acceptance

All QCO DD functionality tests must pass, including fixed, parameterized,
runtime-bound, controlled, RCCX, custom-unitary, reset, and sampling cases. All
QIR runtime tests must pass, including direct, fixed-control, generic-control,
parameterized, SWAP, global-phase, reset, and state ownership cases. DDSIM QDMI
tests must continue accepting and executing supported OpenQASM and QIR jobs.

The DD-only configuration must build `MQT::CoreDD` and pass its package tests
without LLVM or MLIR targets. The generated installed CoreDD interface must not
mention an MLIR target. No production source may define a second named-gate
formula or `dd::GateType`. QCO matrices must remain the only formula source and
the central gate table must no longer carry a DD operation spelling.

`git diff --check`, C++ lint, and `uvx nox -s lint` must pass, unless an
external tool failure is recorded with its exact diagnostic and does not
originate in the change.

## Idempotence and Recovery

Configuration, builds, tests, searches, formatting, and lint are repeatable. Use
separate build directories for MLIR-enabled and DD-only validation. If a build
directory is stale, rerun CMake configuration; do not delete user files or reset
the worktree. All source edits remain recoverable in Git. Inspect
`git status --short` before broad edits and preserve changes outside this plan.

## Artifacts and Notes

Issue #2331 follows the review discussions on #2288 that identified duplicate DD
gate matrices and the QIR runtime as their second consumer. Those discussions
also explicitly defer the architectural cleanup from #2288 to this follow-up.

## Interfaces and Dependencies

`MQT::CoreDD` remains the installed backend-neutral library and publishes raw
matrix aliases plus `dd::Package` constructors. It has no MLIR dependency.

`MLIRQCODDAdapter` is an internal MLIR library with public build dependencies on
`MLIRQCODialect`, `MLIRQCOMatrix`, and `MQT::CoreDD`. It is consumed by
`MLIRQCODDFunctionality` and `MQT::CoreQIRRuntime`; it is not appended to
`MQT_CORE_TARGETS` or exported as part of the installed CMake package.

The QIR C ABI remains unchanged. The C++ DD convenience surface removes
`dd::GateType`, `dd::opToSingleQubitGateMatrix`, `dd::opToTwoQubitGateMatrix`,
`dd::opToThreeQubitGateMatrix`, and `dd::getGateDD` for v4. Existing callers
construct DDs from raw matrices or use the QCO-owned adapter when compiling
inside the MLIR stack.
