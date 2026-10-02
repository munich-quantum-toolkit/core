# Preserve supported controlled operations

Status: historical implementation record.

## Goal and scope

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

## Constraints

- The existing decomposition pass only rewrites controlled X, Z, constant phase,
  SWAP, and bare RCCX operations. Controlled H, RX, RXX, and RCCX bodies already
  pass through unchanged, so the regression must include X, phase, SWAP, and
  bare RCCX to exercise the new skip path as well as broader bodies to protect
  end-to-end target support.

- Qiskit 2.5 emits a deprecation warning when `Gate.control()` receives its
  former implicit `annotated=None` value, and the test suite treats that warning
  as an error. Evidence: The first integration run failed in `HGate.control(2)`
  before compilation. Passing `annotated=False` constructs the concrete
  controlled gate that the importer needs.

- The minimum dependency session installs Qiskit 1.1, while the compiler
  translation supports Qiskit 2.5.x. The end-to-end regression must use the same
  version guard as the existing Qiskit translation tests.

## Decisions

- Reuse `DecomposeMultiControlled` with an optional immutable `CompilerTarget`,
  exposed by one overloaded factory. The target-independent generated factory
  keeps its current behavior. Rationale: This puts the support check at the only
  rewrite point and avoids a second pass or operation filter.

- Preserve target-supported operations only for all-to-all connectivity.
  Rationale: The current mapper accepts at most two-qubit operations, so
  explicit topologies must retain the established decomposition before mapping.

## Outcome and validation

The target pipeline now retains native controlled operations on all-to-all
targets and keeps the established decomposition on explicit topologies. The
compiler regression covers controlled H, RX, RXX, SWAP, RCCX, X, and phase, plus
bare RCCX and global phase. The QDMI regression executes the resulting QIR on
DDSIM and matches Qiskit's complex statevector exactly. The minimum dependency
session skips this Qiskit translation regression. All focused tests, the full
compiler test binary, repository lint, and C++ lint pass.

## Code and ownership

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

## Acceptance

Acceptance requires three behaviors. Target compilation for the DDSIM all-to-all
target keeps every controlled operation that the target reports as supported.
Compilation for an explicit-coupling target still has no operation wider than
two qubits before mapping completes. Zero-shot QIR execution on DDSIM produces
the same complex statevector as Qiskit for a circuit containing broad controlled
gates and a nonzero global phase.

The final tree must also pass `git diff --check`,
`uvx nox -s cpp-lint -- origin/codex/generalize-compiler-target`, and
`uvx nox -s lint`.

## Interfaces

At completion, `mlir/include/mlir/Dialect/QCO/Transforms/Passes.h` declares:

    std::unique_ptr<Pass>
    createDecomposeMultiControlled(const CompilerTarget& target,
                                   uint64_t minQubits = 3);

The implementation reuses `CompilerTarget::supports(Operation*)` and adds no
dependency. Existing generated factories and command-line options remain
unchanged.
