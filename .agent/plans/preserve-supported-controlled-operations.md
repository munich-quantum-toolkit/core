# Preserve supported controlled operations

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

The compiler currently decomposes every supported multi-controlled X, Z, phase,
SWAP, and RCCX operation before it checks whether an all-to-all target can
execute that operation directly. After this change, target compilation keeps
controlled operations that the `CompilerTarget` reports as native. The
standalone `decompose-multi-controlled` pass remains target-independent, and
targets with explicit couplings still receive the existing decomposition before
mapping.

The change is visible by compiling a circuit for the QDMI DDSIM device. Broad
controlled gates, including controlled single-, two-, and three-target base
gates, remain in QCO until QIR lowering and execute with the same complex
statevector as the source circuit.

## Progress

- [x] (2026-09-01 00:00Z) Read the compiler target and multi-controlled
      decomposition contracts.
- [x] (2026-09-01 00:20Z) Added a target-aware construction path to the existing
      decomposition pass.
- [x] (2026-09-01 00:25Z) Used the target-aware pass in target compilation
      without changing the command-line pass.
- [x] (2026-09-01 00:45Z) Added focused compiler and DDSIM execution
      regressions.
- [x] (2026-09-01 19:14Z) Ran focused and full compiler tests, the focused DDSIM
      execution test, C++ lint, repository lint, and inspected the final diff.
- [x] (2026-09-01 21:22Z) Restacked on the final compiler-target change and
      verified the Qiskit regression in current and minimum dependency sessions.

## Surprises & Discoveries

- Observation: The existing decomposition pass only rewrites controlled X, Z,
  constant phase, SWAP, and bare RCCX operations. Controlled H, RX, RXX, and
  RCCX bodies already pass through unchanged, so the regression must include X,
  phase, SWAP, and bare RCCX to exercise the new skip path as well as broader
  bodies to protect end-to-end target support.
- Observation: Qiskit 2.5 emits a deprecation warning when `Gate.control()`
  receives its former implicit `annotated=None` value, and the test suite treats
  that warning as an error. Evidence: The first integration run failed in
  `HGate.control(2)` before compilation. Passing `annotated=False` constructs
  the concrete controlled gate that the importer needs.
- Observation: The minimum dependency session installs Qiskit 1.1, while the
  compiler translation supports Qiskit 2.5.x. The end-to-end regression must use
  the same version guard as the existing Qiskit translation tests.

## Decision Log

- Decision: Reuse `DecomposeMultiControlled` with an optional immutable
  `CompilerTarget`, exposed by one overloaded factory. The target-independent
  generated factory keeps its current behavior. Rationale: This puts the support
  check at the only rewrite point and avoids a second pass or operation filter.
  Date/Author: 2026-09-01 / Codex.
- Decision: Preserve target-supported operations only for all-to-all
  connectivity. Rationale: The current mapper accepts at most two-qubit
  operations, so explicit topologies must retain the established decomposition
  before mapping. Date/Author: 2026-09-01 / Codex.

## Outcomes & Retrospective

The target pipeline now retains native controlled operations on all-to-all
targets and keeps the established decomposition on explicit topologies. The
compiler regression covers controlled H, RX, RXX, SWAP, RCCX, X, and phase, plus
bare RCCX and global phase. The QDMI regression executes the resulting QIR on
DDSIM and matches Qiskit's complex statevector exactly. The minimum dependency
session skips this Qiskit translation regression. All focused tests, the full
compiler test binary, repository lint, and C++ lint pass.

## Context and Orientation

`mlir/lib/Dialect/QCO/Transforms/Decomposition/DecomposeMultiControlled.cpp`
implements the existing target-independent decomposition pass. Generated pass
factories come from `mlir/include/mlir/Dialect/QCO/Transforms/Passes.td`, while
hand-written factories are declared in
`mlir/include/mlir/Dialect/QCO/Transforms/Passes.h`.

`mlir/lib/Compiler/TargetCompilation.cpp` assembles the canonical target
pipeline. It currently invokes the generic decomposition pipeline before
choosing placement for all-to-all connectivity or mapping for explicit
couplings. `CompilerTarget::supports(Operation*)`, declared in
`mlir/include/mlir/Compiler/Target.h`, recognizes structurally controlled base
operations with variadic target arity.

`mlir/unittests/Compiler/test_compiler_pipeline.cpp` owns compiler-pipeline
behavior. `test/python/qdmi/test_qdmi.py` can compile Qiskit circuits against
the registered DDSIM target, submit QIR with zero shots, and read the exact
dense statevector.

## Plan of Work

Extend the existing pass class with an optional compiler target. Give the two
rewrite patterns access to that target. Before an eligible rewrite, return
without changing the operation when the target is all-to-all and reports the
operation as supported. Apply the same rule to bare RCCX because it is a native
DDSIM operation that this pass otherwise expands.

Declare and implement one overload of `createDecomposeMultiControlled` that
accepts a compiler target and the existing minimum-qubit threshold. Replace the
generic decomposition step in `populateTargetCompilationPipeline` with this
overload. The pass itself enforces the all-to-all condition, so an explicit
target still takes the old rewrite path.

Add a compiler test that derives the target from the registered DDSIM device,
compiles controlled H, RX, RXX, SWAP, RCCX, X, and phase operations, and checks
that each supported controlled body remains. Keep the existing explicit-target
test as the regression that unsupported higher-arity controls decompose before
mapping. Add one Python QDMI test that executes a small circuit containing the
same gate families and a nonzero global phase, then compares every complex
amplitude with Qiskit's reference statevector.

## Concrete Steps

From the repository root, edit the pass, factory declaration, target pipeline,
and focused tests described above. Build and run:

    cmake --build --preset release --target mqt-core-mlir-unittests-compiler
    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler --gtest_filter='CompilerPipelineTest.*Controlled*'
    uvx nox -s tests-3.12 -- test/python/qdmi/test_qdmi.py -k 'controlled or qir' -q
    uvx nox -s minimums-3.12 -- test/python/qdmi/test_qdmi.py -k 'controlled or qir' -q
    uvx nox -s cpp-lint -- origin/codex/generalize-compiler-target
    uvx nox -s lint

The focused C++ and Python tests must pass. The Python execution test must
report equal complex statevectors, including their global phase.

## Validation and Acceptance

Acceptance requires three behaviors. Target compilation for the DDSIM all-to-all
target keeps every controlled operation that the target reports as supported.
Compilation for an explicit-coupling target still has no operation wider than
two qubits before mapping completes. Zero-shot QIR execution on DDSIM produces
the same complex statevector as Qiskit for a circuit containing broad controlled
gates and a nonzero global phase.

The final tree must also pass `git diff --check`,
`uvx nox -s cpp-lint -- origin/codex/generalize-compiler-target`, and
`uvx nox -s lint`.

## Idempotence and Recovery

All edits and test commands are repeatable. The change does not alter external
state, create recovery branches, or modify another worktree. If the Python
integration test exposes an unrelated importer limitation, retain the focused
C++ regression and record the exact limitation here instead of adding a large
workaround.

## Artifacts and Notes

The compiler binary passed 144 tests. The current QDMI integration test passed:

    1 passed in 1.14s

The minimum dependency session installed Qiskit 1.1 and skipped the unsupported
translation as expected:

    1 skipped in 1.94s

`uvx nox -s lint`, focused compiler tests, `git diff --check`, and manual
clang-tidy on all three changed C++ sources passed. The aggregate C++ lint
selected all three changed sources and passed without findings.

## Interfaces and Dependencies

At completion, `mlir/include/mlir/Dialect/QCO/Transforms/Passes.h` declares:

    std::unique_ptr<Pass>
    createDecomposeMultiControlled(const CompilerTarget& target,
                                   uint64_t minQubits = 3);

The implementation reuses `CompilerTarget::supports(Operation*)` and adds no
dependency. Existing generated factories and command-line options remain
unchanged.

Revision note: Created this plan on 2026-09-01 for the independent target-aware
controlled-operation workstream.

Revision note: Updated progress, discoveries, outcomes, and validation evidence
after implementation on 2026-09-01.

Revision note: Restacked on the final compiler-target commit and verified the
minimum dependency session on 2026-09-01.
