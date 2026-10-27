# Separate cleanup from quantum optimization

One end-to-end PR separates structural cleanup, modifier lowering and optional
quantum optimization across import, binding, synthesis and serialization.

## Correctness contracts

| Entry point                        | Contract                                                                                                                                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| General compilation                | Preserve quantum/classical semantics, controlled relative phase, wire identity, output order and target constraints. Optimization and synthesis may change the gate sequence.                           |
| Ideal simulation and functionality | Preserve the documented amplitudes, matrices or measurement observables. Coherent APIs retain global phase.                                                                                             |
| Already-compiled native submission | Preserve supplied unitary gate events, physical operands and bound parameters on each executed path, together with measurement and feedback dependencies. Input must fit the representation and target. |

Ideal-unitary equivalence does not establish gate-sequence preservation. Both
exact-zero and tolerance-zero rotation removal belong to explicit optimization.
The absolute parameter tolerance of `1e-15` is not a circuit error bound. Dense
identity matrices are checked exactly.

Empty modifier shells are structural. A zero-target modifier containing a phase
is not empty. Shell removal preserves classical evaluation, diagnostics and
positional wire yields. Global-phase normalization is safe during cleanup and
export preparation: controlled relative phases remain conditional operations.

## Supported subsets and assumptions

- A quantum program uses QC or QCO. QCO input is valid linear IR; boundary
  verification owns one-use and positional wire constraints.
- Parameter binding supports named f64 scalars and static rank-one f64 arrays,
  including partial binding. Formal arguments stay symbolic until binding or
  inlining exposes constants. Runtime expressions remain unresolved.
- Classical evaluation and static branch selection are allowed during cleanup.
  LLVM/MLIR folding and canonicalization own this work.
- Modifier lowering supports fixed single-qubit QCO gates with runtime powers
  and guarded constant powers of other gates. Unsupported powers remain for
  conversion diagnostics. Numerical and branch-cut guards define supported
  powers, rather than an optional accuracy mode.
- Native input must be representation-compatible. A provenance attribute cannot
  repair inappropriate optimization scheduling. Stored `CompiledProgram`
  submission remains direct.
- OpenQASM preserves barriers. Formats without barriers omit them. Within IR,
  duplicate barriers on the same ordered wires may merge; overlapping partial
  barriers retain their dependencies.
- Simulation and DD functionality retain their existing control-flow and result
  restrictions. Unsupported timing, assertions and custom operations retain
  their boundary diagnostics.

## Transformation ownership

| Transformation                                                                                                    | Owner                                                                                          |
| ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Constant arithmetic/math, casts, bound array extracts, classical DCE and static branches                          | LLVM/MLIR folding and canonicalization; importer-local `createOrFold` for constant expressions |
| Empty modifiers, unused shell wires, resource scaffolding and duplicate barriers                                  | Structural canonicalization                                                                    |
| Reset identities                                                                                                  | Existing reset fold and trait                                                                  |
| Identity gates/matrices, zero rotations, gate specialization, inverse cancellation and local rotation composition | `simplify-quantum`                                                                             |
| Nested controls, inverse and supported power lowering                                                             | `lower-quantum-modifiers`                                                                      |
| Rotation fusion and controlled-gate combination                                                                   | Existing dedicated optimization passes                                                         |
| Phase normalization                                                                                               | Existing normalization pass in cleanup and export preparation                                  |
| Phase dropping, dead-gate removal, terminal swaps and measurement lifting                                         | Existing passes under their output and layout contracts                                        |

QC and QCO expose optimization and lowering patterns separately from structural
canonicalization. QIR retains unused-resource cleanup and LLVM simplification.
QCO linear values keep gates live; phase operations carry an effect.

## Compiler flow and affected callers

- `Dialect/{QC,QCO}/IR/` owns gate rules and modifier rules. Shared `QCOUtils.h`
  zero-rotation handling serves direct synthesis consumers too.
- `Dialect/MQT/Transforms/SimplifyQuantum.cpp` owns the explicit quantum passes.
  They fold generated arithmetic and leave CFG simplification to cleanup.
- `MergeSingleQubitRotationGates.cpp` and `DecomposeMultiControlled.cpp`
  populate the rules needed by synthesis directly. Changing canonicalizer
  registration alone does not update these consumers.
- `Support/Passes.cpp` and `Compiler/TargetCompilation.cpp` schedule
  optimization for general compilation and target synthesis. The leading
  simplifier exposes modifier bodies to fusion; the trailing simplifier reduces
  generated gates.
- `Compiler/Pipeline.cpp` and OpenQASM 2 preparation lower modifiers where the
  output representation requires it. Cleanup and export normalize phases.
- `ReplaceClassicalControls.cpp` inlines the body directly when all quantum
  controls have been removed.
- Binding, classical-memory promotion and inlining use canonicalization. Cleanup
  after these mutations is necessary even when import folded constants.
- Qiskit native serialization uses the shared circuit validator and the output
  converters. It does not schedule optional quantum optimization.

Modifier unrolling also performs selective call inlining and phase
normalization. It has a different contract from local modifier lowering.
Constant recognition remains local and nonmutating; read-only matrix queries
must not invoke mutating folds on live IR. DD evaluation retains clone-based
folding.

## Evidence and acceptance gates

Tests concentrate each contract at its owner:

- Cleanup retains H/H, exact-zero and tiny rotations. Explicit optimization
  simplifies eligible gates. Barrier tests retain overlapping dependencies.
- Captured native OpenQASM and QIR payloads check gate order, parameters,
  physical sites and asymmetric readout. IQM tests check exported instructions.
- Parameterized controlled-U uses an independent trigonometric full-matrix
  oracle. Partial binding, runtime branches and powers exercise public APIs.
- Synthesis, decomposition, DD functionality and format round trips check phase,
  ordering and supported subsets. Importer regressions from Core #2879 remain.

No original Catalyst program or transformation trace is available. The ZNE
example is explicitly synthetic. Its sequence oracle detects H/H cancellation
and rotation removal; deleting the controlled phase independently fails the
matrix oracle (maximum error `0.234`, compared with `2.8e-17` for correct
output).

Profiles separate import, cleanup, optimization, QC/QIR lowering and extraction.
The public functionality call combines DD construction and matrix extraction.
Reachable IR size and peak RSS do not establish allocation volume or occupied DD
storage. No memory-saving claim is made.

The baseline for this feedback round passes 4,159 native tests and 1,546
selected Python integration tests. Isolated expensive rewrite checks pass 1,288
selected tests. A dependency-only SCF reproducer exposes an upstream
rewrite-notification defect; no Core workaround or installed dependency patch is
included. Large-loop expensive checks exceed the bounded test budget. These
checks are not a full expensive-check LLVM build.

Executable prototypes compare existing indexed pattern dispatch,
dialect-specific collection and centralized matching before choosing a
production change. Quantum linearity, classical dependencies, side effects,
region boundaries and rewrite notifications are required regardless of dispatch
strategy. Measurements and reproducers live outside the repository in
`/home/nvidia/.codex/experiments/cleanup-separation-550d/`.

Acceptance requires the native suite, Python integration tests, generated stubs,
repository lint and whole-file C++ lint on the final change. Windows, hosted CI
and a full expensive-check LLVM/ASAN build are separate validation modes.

## Compatibility

Callers that need cancellation, fusion or gate specialization request
`simplify-quantum` or an optimizing pipeline. Direct C++ pattern-population
callers use `getOptimizationPatterns` or `getLoweringPatterns`. General
compilation retains optimization; native submission preserves gate events.
