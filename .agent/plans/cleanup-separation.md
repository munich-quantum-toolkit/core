# Separate cleanup from quantum optimization

Status: implemented and locally validated; hosted validation is separate.

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
  LLVM/MLIR folding and canonicalization own this work. Phase normalization runs
  before the cleanup canonicalizer, whose interleaved CSE folds generated
  arithmetic and expressions exposed by deduplication. Cleanup preserves phase
  semantics; explicit normalization provides phase aggregation. Known-invalid
  phase expressions remain in their original region for dead-code removal or
  diagnostics.
- Register compaction matches supported static registers with already-folded
  indices. Standalone callers canonicalize size and index expressions first.
  Candidate worklists handle sibling rewrites without sweeping unrelated IR.
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

Native validation passes 4,164 tests; Python compiler and submission integration
passes 1,246 tests. Repository lint and whole-file C++ lint pass. Compaction
tests check sparse width, repeated-wire mapping, source metadata, nested
deallocation and the folded-index precondition. A public cleanup regression
checks QC and QCO compaction without assuming shared SSA constants. CSE-exposed
control flow and symbolic phase are checked together, including phase-sensitive
binding results. Reachable and unreachable invalid phase expressions check
diagnostics under control modifiers.

The shrinking passes use the upstream operation worklist with module scope;
patterns may replace allocations outside a nested deallocation's region. Only
candidate and newly created operations enter the worklist. Phase normalization
retains its local folding utility; cleanup's canonicalizer owns general folding
and CSE. A separate trailing canonicalizer was rejected because it increased
pattern-matching work and compilation time.

Representative target workloads cover Core's GHZ, QFT, QAOA, Bernstein–Vazirani
and Grover generators through native IQM and IBM output. Against pre-change
commit `76590d7b`, 51 workload/target pairs used the same Clang 23 ThinLTO
toolchain and dependency archives, CPU 9 affinity, three rounds and five
repetitions per round (first discarded). The harness asserts each loaded
extension path and alternates variant order per workload. All final payload
hashes match across variants and repetitions. Independent NumPy simulation of
eight small payloads matches analytic benchmark probabilities within `1.2e-15`.

Paired geometric-mean time ratios are `0.979` for QC cleanup, `0.979` for QCO
cleanup, `0.994` through target QC and `0.982` through final output. Aggregate
end-to-end round ratios are `0.997/1.037/0.949`; the large QFT/Emerald case is
variable. A focused follow-up of six alternating paired rounds on QFT48/Emerald
(three repetitions, first discarded) retained identical payloads and measured a
median paired ratio of `0.975`. These shared-host results support removing
repeated cleanup work, not a uniform or isolated speedup claim. Benchmark
harnesses and raw measurements remain outside the repository.

Earlier isolated expensive rewrite checks passed 1,288 selected quantum tests;
these do not validate the changed compaction driver under a full expensive-check
LLVM build. A dependency-only SCF reproducer exposes an upstream
rewrite-notification defect. No Core workaround or dependency patch is included.

Acceptance requires the native suite, Python integration tests, generated stubs,
repository lint and whole-file C++ lint on the final change. Windows, hosted CI
and a full expensive-check LLVM/ASAN build are separate validation modes.

## Compatibility

Callers that need cancellation, fusion or gate specialization request
`simplify-quantum` or an optimizing pipeline. Direct C++ pattern-population
callers use `getOptimizationPatterns` or `getLoweringPatterns`. General
compilation retains optimization; native submission preserves gate events.
