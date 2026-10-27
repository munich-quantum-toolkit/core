# Separate cleanup from quantum optimization

This change targets upstream main `4f09df3267` as one end-to-end PR. The unit of
review includes pattern ownership, compilation and serialization callers, and
observable behavior. Further splitting is optional after reviewing that result.

## Required contracts

| Entry point                        | Contract                                                                                                                                                                                                                                                 |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| General compilation                | Preserve quantum/classical semantics, controlled relative phase, wire identity, output order and target constraints. Explicit optimization and synthesis may change gates.                                                                               |
| Ideal simulation and functionality | Preserve the documented amplitudes, matrices or measurement observables. Coherent APIs retain global phase. Ideal agreement alone does not establish native sequence preservation.                                                                       |
| Already-compiled native submission | Preserve supplied unitary events, physical operands and bound parameters on each executed path, plus measurement/feedback dependencies and supported ordering boundaries. Reject an unrepresentable native circuit rather than synthesize a replacement. |

Reset is state initialization. Repeated resets and reset after a
zero-initialized allocation remain foldable. Keep the existing reset fold and
`Idempotent` trait; reset count is not a native sequence oracle. This does not
permit commuting a reset through measurement or treating a static device qubit
as freshly allocated.

Both exact-zero and tolerance-zero rotation removal change the supplied unitary
sequence. They require explicit optimization. The existing absolute parameter
tolerance of `1e-15` remains an optimization policy, not a circuit error bound.
Identity matrices are checked exactly. Preserve controlled relative phase even
when a final-program phase may be discarded by a particular output format.

An empty modifier is structural scaffolding. A zero-target modifier containing
`gphase` is not empty. Removing shells must preserve classical evaluation,
diagnostics and positional wire yields. A barrier constrains ordering across its
wires: exact duplicate boundaries may merge, but overlapping partial barriers
must retain their dependencies. A barrier on no wires has no constraint;
OpenQASM expands its operand-free all-qubit syntax before this distinction.

## Supported subsets

These limits belong to existing boundaries; they are not additional semantics:

- Parameter binding supports named f64 scalars and static rank-one f64 arrays,
  including partial binding. Formal gate arguments stay symbolic until binding
  or inlining exposes constants. Runtime values remain unresolved.
- Static-unitary DD functionality and simulation retain their existing dynamic
  control-flow and result restrictions.
- Native QIR and IQM serializers reject nonempty barriers, including nested
  barriers. Direct IQM export also rejects them. General QIR conversion retains
  its weaker, now documented contract: barriers are omitted. No runtime barrier
  intrinsic or provenance attribute is introduced.
- OpenQASM retains barriers and native device indices. Other formats retain
  their existing supported-operation checks; unsupported timing, assertions or
  custom operations do not acquire support through cleanup.
- Lowering supports the existing modifier shapes and guarded constant powers.
  Unsupported powers remain for conversion diagnostics. It does not introduce
  runtime parameter support on backends requiring constant parameters.

## Transformation ownership

| Transformation                                                                               | Owner                                                                                                                                |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Constant arithmetic/math, casts, bound array extracts, classical DCE and static branches     | Existing LLVM/MLIR folding and canonicalization. Importer-local `createOrFold` remains for already-constant expressions.             |
| Empty modifier shells, unused shell wires, resource scaffolding and exact duplicate barriers | Structural canonicalization, preserving side effects and positional wires.                                                           |
| Repeated reset and reset after known-zero allocation                                         | Existing reset trait/fold.                                                                                                           |
| Identity gates/matrices, zero rotations, trivial phases and gate specialization              | Explicit `simplify-quantum`, using the existing patterns and numerical guards.                                                       |
| Inverse-pair cancellation, rotation composition and adjacent control simplification          | Existing patterns exposed to explicit optimization; existing merge/fusion/controlled-combination passes retain their separate roles. |
| Nested controls, inverse and supported power lowering                                        | `lower-quantum-modifiers`, reusing modifier patterns without adjacent-gate cancellation.                                             |
| Phase normalization/drop, dead-gate removal, terminal swaps and measurement lifting          | Existing dedicated passes under their phase, boundary and output-layout contracts.                                                   |

QC unitary and modifier canonicalizers and QCO unitary fold hooks/registrations
are separated accordingly. QIR continues to use its existing unused-resource
cleanup and LLVM simplification. Linear QCO values keep ordinary gates live;
`gphase` already has an effect. No broad change to gate purity is needed.

## Implementation and affected callers

Paths are relative to `mlir/` unless otherwise stated.

- `Dialect/{QC,QCO}/IR/` exposes quantum patterns separately from structural
  canonicalization. `QCOUtils.h` shares zero-rotation removal with direct
  synthesis consumers. Reset behavior is unchanged.
- `Dialect/MQT/Transforms/SimplifyQuantum.cpp` houses the two explicit passes.
  Both use the existing greedy driver. There is no separate classical evaluator,
  whitelist, interpreter, mutable traversal algorithm or constant cache.
- `MergeSingleQubitRotationGates.cpp` and `DecomposeMultiControlled.cpp`
  populate the required optimization/lowering patterns directly. Moving
  registrations alone would leave these consumers incomplete.
- `Support/Passes.cpp` and `Compiler/TargetCompilation.cpp` explicitly schedule
  quantum simplification for general optimization and target synthesis.
  `Compiler/Pipeline.cpp` and OpenQASM 2 preparation request modifier lowering
  where the representation needs it. Cleanup no longer normalizes global phases.
- `ReplaceClassicalControls.cpp` directly inlines the body when no quantum
  controls remain. It no longer creates a wrapper that another pass must remove.
- Parameter binding, classical-memory promotion and inliner cleanup retain
  ordinary canonicalization. Import-time folding cannot replace these calls.
  `valueToDouble` recognition remains unchanged for its optimization consumers.
- Qiskit's existing circuit validator owns native format checks, and serializers
  pass their selected format to it. Direct IQM export rejects barriers before
  QIR lowering could erase them. Stored `CompiledProgram` submission stays
  direct.
- Tests comparing optimized representations request optimization explicitly. Raw
  conversion tests still exercise raw converters and retain their negative
  diagnostics. Public preparation tests cover modifier lowering separately.

## Assumptions that remove work

1. Valid linear QCO is the internal input contract. Existing boundary checks own
   exactly-one-use and positional wire correspondence; individual rewrites need
   no duplicate use-count scans.
2. Classical evaluation and static branch selection are allowed during cleanup.
   Existing MLIR machinery therefore replaces the experimental classical-only
   pass, including its operation whitelist and extra traversal.
3. Already-compiled native input must be representation-compatible. Validation
   rejects unsupported input; a provenance attribute cannot repair an
   incorrectly scheduled optimizer.
4. Constant recognition remains local and nonmutating. Do not invoke potentially
   mutating folds on live IR from read-only matrix queries. The existing DD
   evaluator's clone-based approach remains unchanged.
5. Preparation is reusable only until a relevant mutation. Keep cleanup after
   binding/inlining and required phase normalization at representation
   boundaries; do not remove it merely because the importer already folded
   constants.

Pattern packaging and worklist order are implementation choices. Quantum
linearity, classical dependencies, side effects and region boundaries are
requirements. `QCO/Utils/Drivers.h` requires nonmutating callbacks and tracks
quantum-wire dependencies; it is unsuitable as a general mutable rewrite driver.

## Evidence and acceptance

The corpus concentrates each contract at its owner:

- Cleanup retains H/H, exact-zero and tiny rotations; explicit optimization
  simplifies eligible gates. Overlapping barriers retain their shared
  dependency.
- Native QASM and both QIR profiles capture the submitted payload, including
  unitary order, rotation parameters, physical sites and asymmetric readout.
  Native QIR/IQM reject barriers, including nested ones.
- Parameterized controlled-U uses an independent trigonometric full-matrix
  oracle. Partial scalar/array binding, live/unreachable nonfinite expressions,
  runtime branches and bound `pow(p + 1)` exercise existing public APIs.
- Synthesis, decomposition, DD functionality and format round trips retain their
  phase-sensitive, ordering and supported-subset checks. Importer regressions
  from merged Core #2879 remain intact. No original Catalyst program or
  transformation trace is available; the folded circuit is explicitly synthetic.

The baseline cleanup removes the synthetic H/H and both rotation events. The new
sequence oracle detects that behavior. Removing the controlled phase from the
parameterized example separately fails its full-matrix oracle. These checks
establish different contracts and are not interchangeable.

Profiling separates IR import, cleanup, explicit optimization, QC lowering and
OpenQASM serialization. The public functionality call combines DD construction
and matrix extraction, so its timing is reported jointly. Reachable IR operation
counts and process peak RSS are recorded separately; neither establishes
allocation volume or occupied DD storage. No allocation or memory-saving claim
is made. No rewrite algorithm change is justified by the measurements.

The separate classical evaluator added no useful measured improvement when
followed by necessary canonicalization on the same arithmetic corpus. Its
standalone output retained duplicated constants. Deleting it is justified by
redundancy, not by comparing unequal outputs or promising a compiler speedup.

## Compatibility and delivery gates

`cleanup()` and generic canonicalization no longer cancel, fuse or specialize
unitary gates. Callers relying on that behavior must request `simplify-quantum`
or an existing optimizing pipeline. Direct C++ pattern-population callers use
`getOptimizationPatterns` or `getLoweringPatterns` as appropriate. Default
compilation retains explicit optimization. Native QIR/IQM barrier rejection is a
visible compatibility change and must appear in the PR description.

Implementation slices are parts of this one PR: pattern ownership and shared
consumers; pipeline and serialization integration; semantic regression
migration; installed-artifact validation and profiling. Acceptance requires
native tests, Python integration tests, regenerated stubs, repository lint and
whole-file C++ lint on the complete change. Expensive MLIR rewrite-check builds,
Windows and hosted CI are separate validation modes and must not be claimed
without running them. No further traversal or compiler redesign is included.

## Local acceptance results

- Optimized Clang 23/ThinLTO build and all 4,157 CTest entries pass.
- The installed wheel passes 1,546 selected MLIR, Qiskit and PennyLane tests.
  Stubs are regenerated, and whole-file C++ lint passes.
- The independent controlled-U matrix differs by at most `2.8e-17`; deleting its
  controlled phase produces maximum error `0.234`, detected by the oracle.
- Nine-sample diagnostic medians on the 4,096-gate arithmetic/barrier corpus are
  13.89 ms for baseline canonicalization, 13.76 ms for separated cleanup and
  15.10 ms for the dedicated quantum pass. Each retains one shared constant,
  4,096 rotations and 4,096 barriers. Adding canonicalization after the quantum
  pass costs 18.25 ms total on this corpus and provides no further reduction.
  These shared-host measurements do not establish a general speedup.

Raw profiles, emitted IR and QIR, mutation results, installed wheel identity and
check logs are retained outside the repository in
`/home/nvidia/.codex/experiments/cleanup-separation-550d/full-pr/`. Windows,
hosted CI and an expensive MLIR rewrite-check build were not run.
