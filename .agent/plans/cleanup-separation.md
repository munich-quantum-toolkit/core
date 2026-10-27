# Canonical cleanup, classical evaluation, and quantum optimization

Status: refreshed on upstream main on 27 October 2026. The first implementation
slice exposes explicit classical folding. Production-wide migration remains
proposed; automatic scheduling and parameter binding are unchanged.

## Decision

Separate classical folding and structural normalization from quantum
simplification. Make the latter explicit in compilation and optimization
pipelines. Native submission must preserve supplied physical operations,
including mathematical identities; unitary equivalence does not establish that
contract. Do not attach provenance attributes to compensate for pass scheduling.

Keep the existing rewrite machinery. There is no correctness argument for using
the quantum wire traversal as a general mutable classical/quantum rewrite
driver, and the measurements below do not justify replacing it. This agrees with
[MLIR’s canonicalization guidance](https://mlir.llvm.org/docs/Canonicalization/):
required correctness must not depend on optional canonicalization.

The executable prototype adds `fold-classical-values`, an explicitly callable
pass. It uses MLIR's selected-operation greedy driver with an empty pattern set,
`ExistingOps` strictness, and constant CSE disabled. Only regionless numeric
arith/math operations and f64 tensor extracts enter the worklist. Numeric
operand/result checks exclude `arith.select` of quantum handles. Existing fold
hooks and classical dead-code elimination run with driver mutation
notifications; quantum users and enclosing regions cannot enter the worklist. No
interpreter, new constant-recognition API, traversal, or production pipeline
switch is added.

## Evidence and revisions

Current implementation base is `4f09df3267e5b293d13ae6f412d6c6bd178965a6`.
[Core #2878](https://github.com/munich-quantum-toolkit/core/pull/2878) is
merged. [Core #2879](https://github.com/munich-quantum-toolkit/core/pull/2879)
is now merged as `f2d4378151`. Its importer changes and all seven regressions
are in the base and are no longer part of this diff.
[Core #2849](https://github.com/munich-quantum-toolkit/core/pull/2849) adds
exact DD expectation evaluation and a Qiskit estimator. Its QCO binding calls
the DD preparation path directly and preserves the input program. Add its tests
to the simulation acceptance gate; it does not change the cleanup ownership
decision.

The measurements below remain historical evidence on `afd4aefcc8` plus #2879;
they are not performance measurements of the refreshed main branch.

The pre-existing installed #2879 experiment artifact reported
`4.1.2.dev110+gafd4aefcc.d20261027`; its extension SHA-256 begins
`27e903eb530eb32c9`. That version string alone does not identify its
dirty-source fix. The newly built prototype wheel uses the same package version;
its extension hash begins `b14f30461327ed0b`. Full hashes, original QASM,
imported/emitted IR, submitted payloads, harnesses, and raw samples are in the
companion evidence bundle. The original native and installed-wheel checks cover
the prototype source; refreshed validation is recorded separately below.
Profiling used the preceding artifact; subsequent changes only add a
null-dialect guard and correct C++ access visibility, without changing the
profiled inputs. Full hashes distinguish those artifacts.

A bounded search of current source, local task records and related GitHub issues
recovered no original Catalyst/ZNE program or transformation trace. Historical
Catalyst attribution remains **unverified**. Every folded-circuit example here
is explicitly synthetic; no Catalyst execution is claimed.

## Correctness contracts

| Use                                | Required equivalence                                                                                                 | Permitted changes and limits                                                                                                                                                                                                                                           |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| General compilation                | Quantum semantics, classical dependencies/outputs, resource identity and declared target/payload constraints         | Explicit gate synthesis, cancellation, routing and allocation changes; preserve controlled relative phase and layout/output permutations. Approximation needs a stated numerical policy.                                                                               |
| Ideal simulation / functionality   | Full amplitudes or matrix including phase for coherent APIs; classical result order and branch behavior for sampling | Optional ideal optimization is legal only under that API's numerical/phase contract. Dropping entry-point global phase is a distinct probability-only action. Simulation does not establish native sequence preservation.                                              |
| Already-compiled device submission | Ordered operation names, ordered physical operands, bound parameters, modifiers and supported observable boundaries  | Classical evaluation and representation lowering may change spelling, not deliberately supplied gate events. Preserve zero rotations, identities, resets and folded pairs. Diagnose a boundary the payload cannot represent, or explicitly document a weaker contract. |

For submission, sequence means the executed path's events, not unreachable
branches in the source text. Structural branch selection must retain classical
side effects, measurement/reset order, output register order, assertions and
wire correspondence. Do not move operations across unknown regions, calls,
barriers or feedback merely because their ideal matrices commute.

Exact zero and tolerance zero are different policies. Even exact-zero gate
elimination changes a physical sequence. Tolerance-zero elimination also changes
the ideal unitary: the current QCO rotation folds compare against
`PARAMETER_COMPARISON_TOLERANCE` (1e-15). Move both out of unrestricted fold
hooks if canonical cleanup is to preserve native sequences. Do not silently
replace the existing numerical policy with exact equality throughout synthesis;
first retain it in explicit optimization and decide its error contract
separately.

A modifier with no unitary body can be structural scaffolding. A modifier with
no target wires is not necessarily empty: controlled `gphase` is observable
relative phase. Preserve eager classical computations and their diagnostics when
removing empty shells. Power zero/one, inverse normalization and gate
specialization are not all interchangeable with shell removal.

Current QIR lowering erases `qc.barrier` in
`Conversion/QCToQIR/QIRCommon/QIRCommon.cpp`; the captured payloads confirm
this. Thus today's native QIR serializer preserves the demonstrated gate list
but not a strict barrier-boundary contract. OpenQASM preserves those barriers.
IQM JSON explicitly normalizes and drops final global phase and has a narrower
instruction set. These are format compatibility decisions, not reasons to weaken
the common native contract silently.

## Transformation inventory and ownership

Paths in this section are relative to `mlir/`.

| Class                                 | Current transformations                                                                                                                                  | Proposed destination                                                                                                                                                                      |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Constant evaluation                   | arith/math folding, integer-to-float conversion, U-family phase arithmetic, bound f64 tensor extraction                                                  | Importer `createOrFold` where values exist; explicit classical folding after substitution/inlining. Keep direct constant recognition separate from recursive queries.                     |
| Structural cleanup                    | Static branches, dead classical values, static-qubit normalization, genuinely empty modifier shells, unused shell wires, allocation/consumption pairs    | Canonicalization or shared structural patterns, after proving resource/side-effect and positional-wire contracts. “Unused allocation” is not blanket permission to delete physical reset. |
| Single-operation identity elimination | QCO `IdOp`, exact identity `UnitaryOp`, reset after allocation, tolerance-zero GPhase/P/RX/RY/RZ/RXX/RYY/RZZ/RZX/XXPlusYY/XXMinusYY; QC identity unitary | Explicit quantum simplification. Exact and approximate policies remain distinguishable.                                                                                                   |
| Single-operation resynthesis          | U/U2/R specializations, powered-gate replacements                                                                                                        | Explicit quantum simplification/decomposition. This necessary category does not fit structural cleanup or identity elimination.                                                           |
| Multi-operation optimization          | H/X/Y/Z and phase/SX inverse pairs, SWAP/DCX/ECR/RCCX cancellation, rotation composition, adjacent control grouping                                      | Existing merge/fusion/controlled-combination passes plus shared cancellation helpers; no duplicated pattern implementations.                                                              |
| Observable-boundary rewrites          | Consecutive barrier merging, phase movement, terminal swaps, measurement lifting, dead gates before discard/reset                                        | Explicit passes with their own observable, layout and phase contracts.                                                                                                                    |

QCO fold implementations occur in `lib/Dialect/QCO/IR/Operations/` and its
`StandardGates/` directory. QC has no operation fold implementations in the
inspected source; its canonicalizers cover modifiers, allocation/static
management and identity unitary removal. QIR has no quantum operation fold or
canonicalization registrations: it represents calls in LLVM IR. Its explicit
`QIRCleanup.cpp` removes unused qubit-array allocation/release pairs and
external declarations; generic LLVM folding remains active in its greedy driver.

Existing dedicated passes already provide rotation fusion, single-/two-qubit
unitary fusion, controlled grouping, Hadamard/measurement lifting, classical
control replacement, dead-gate removal, qubit reuse, terminal-swap elimination,
phase normalization/drop, decomposition and target synthesis. Reuse these. The
only missing quantum pass is an explicit home for the formerly implicit
identity/specialization/cancellation patterns; do not create one pass per gate.

## Affected callers and hidden dependencies

| Path                                                                              | Current contract/dependency                                                                                                                                         | Required migration                                                                                                                                                          |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Target/OpenQASM/OpenQASMToQCEmitter.cpp`                                         | Importer-generated U phases/casts previously needed later folding                                                                                                   | Keep #2879's `createOrFold` and seven regressions. Formal arguments remain unresolved at import.                                                                            |
| `Compiler/ParameterBinding.cpp`                                                   | Replaces named scalar/array inputs, then canonicalizes the entry                                                                                                    | Fold classical values **and** resolve newly static branches before final verification. Preserve partial binding, array shape, metadata and transactional validation.        |
| `Support/Passes.cpp`                                                              | QC/QCO cleanup, default optimization, OpenQASM2/QIR preparation and export interleave canonicalization with phases/CSE/liveness                                     | Separate documented cleanup and optimization schedules; retain format-required structural legalization. An empty custom pipeline currently still gets cleanup before/after. |
| `Compiler/Pipeline.cpp`, `TargetCompilation.cpp`                                  | General compilation deliberately optimizes; target inliner already uses an empty callable pipeline                                                                  | Schedule explicit quantum simplification at the former implicit optimization points. Default MLIR inlining otherwise runs canonicalization inside callables.                |
| `QCO/Transforms/Optimizations/MergeSingleQubitRotationGates.cpp`                  | Native symbolic synthesis directly populates RX/RY/RZ/P canonicalizers                                                                                              | Extract shared composition helpers and retarget these callers before removing registrations.                                                                                |
| `QCO/Transforms/Decomposition/DecomposeMultiControlled.cpp`                       | Direct Ctrl/Inv/Pow canonicalizer population obtains decomposition normal forms                                                                                     | Split structural and quantum helpers; explicitly request the required ones.                                                                                                 |
| `Conversion/QCToQCO/QCToQCO.cpp`                                                  | Direct QC static-qubit patterns                                                                                                                                     | Retain representation normalization and linearity verification.                                                                                                             |
| `QCO/Transforms/LegalizePayloadControlFlow.cpp`                                   | Selected static-branch/tensor patterns run through folding-enabled drivers                                                                                          | Pattern selection alone does not restrict quantum folds. Preserve loop budgets, capture checks and positional quantum state.                                                |
| `QC/Builder/QCProgramBuilder.cpp`                                                 | Selected SCF/arith patterns restructure loops using a greedy worklist                                                                                               | Preserve CFG/SCF and source-order contracts; examine implicit folds independently.                                                                                          |
| `MQT/Transforms/PromoteClassicalMemory.cpp`                                       | Canonicalization folds access indices before SROA/Mem2Reg; comparison driver also folds                                                                             | Replace only with an index/alias/branch-capable classical pipeline, not the prototype alone.                                                                                |
| `Conversion/QCOToJeff/QCOToJeff.cpp`, `Target/OpenQASM/TranslateQCToOpenQASM.cpp` | Alias/comparison patterns and array normalization use generic greedy folding                                                                                        | Restrict folding or remove quantum folds globally; keep array normalization, safety diagnostics and export support.                                                         |
| `Conversion/QCToQIR/`                                                             | Profile-specific preparation, phase normalization, modifier lowering and final LLVM canonicalization                                                                | Distinguish required representation lowering from optional quantum optimization; preserve calling conventions and output order.                                             |
| `bindings/mlir/register_mlir.cpp`, `QCO/Utils/DDFunctionality.cpp`                | Raw simulation requests raw QCO, not optimized QCO; unitary matrix readers require constant matrices                                                                | Explicit classical preparation can resolve inlined helpers. Do not silently optimize native programs to make simulation work.                                               |
| `Compiler/QDMIAdapter.cpp`, Python Qiskit serializers                             | `CompiledProgram` validates compatibility and submits stored payload; uncompiled input invokes compilation. Qiskit serializers use `native=True` and direct exports | Keep these public contracts distinct. No provenance attribute is necessary. Audit format-specific helpers rather than adding an optimizer to serialization.                 |

`valueToDouble` recognizes direct constants. Broadening it globally would also
change rotation identity folds, matrix fusion, phase handling, gate power and
native synthesis. `valueToConstantAttr` recursively invokes fold hooks on live
IR: MLIR allows in-place folding with empty results, so purity alone does not
make that query mutation-free. This is a source/API-contract hazard, not a
reproduced failure in this task. DD execution already folds a clone in
`DDFunctionality.cpp::foldClassicalOp`; reuse established facilities rather than
adding another evaluator. The prototype's driver owns mutations and revisits
selected arithmetic consumers; symbolic/runtime values remain unresolved.

## Executable findings and rejected integration

The prototype runs through `QCOProgram.run_pass_pipeline`:

```python
program.run_pass_pipeline("inline{default-pipeline=fold-classical-values},fold-classical-values")
```

For a custom gate whose formal arguments feed controlled-U phase arithmetic,
classical folding alone retains the symbolic arithmetic and the existing
unsupported-matrix diagnostic. Explicit inlining with that callable pipeline
then materializes the phase without cancelling the supplied H/H pair. A full
4-by-4 matrix agrees with an independent trigonometric calculation at absolute
error tolerance 1e-12, including controlled relative phase.

Mutation probes demonstrate distinct oracles: deleting the controlled phase
produces maximum entry error **0.2342320850**; replacing the preparation with
quantum cleanup changes the H count **2 to 0**. The latter preserves the ideal
matrix but fails the sequence contract. These probes are not hardware-noise
experiments.

The installed artifact's native serializers submitted synthetic
`H,H,X,RX(0),RX(1e-16),barrier,measure` to the local DDSIM device in OpenQASM3,
QIR Base string and QIR Adaptive string formats. All returned `{'1': 16}`.
Captured ordered payload events retain both H gates and both rotations in all
three formats. Only OpenQASM retains the barrier. This validates local
serializer and submission inputs, not a remote provider's compiler or physical
execution. Existing tests cover register order, parameter binding and profile
feedback.

An attempted replacement of binding's canonicalizer by the prototype failed
`ParameterBinding.RejectsReachableNonFiniteGateParameters`: binding theta=0 in
`if (theta != 0) { ry(1/theta); } else { x; }` made the unreachable gate's
infinity visible before the dead branch disappeared. This production integration
was reverted, not hidden by weakening the test. Binding remains unchanged. The
same experiment confirmed that selecting only arith/math would miss bound array
extracts; the standalone pass therefore includes f64 `tensor.extract`. A
specialist also identified quantum-valued `arith.select`; numeric type checks
and a regression now exclude it.

The tests are consolidated by contract: standalone C++ checks protect sequence,
quantum-handle exclusion and array/cast/runtime folding; the Python helper test
owns the full controlled-phase matrix. #2879 retains its importer regression
matrix and sample test. Existing binding tests remain the migration gate.

## Profiling and driver decision

Seven serial samples per workload on the shared ARM64 Spark, Python 3.14.10,
Clang 23.1.5, LLVM/MLIR 23.1.0, Release/ThinLTO/mold. Each synthetic six-qubit
workload repeats H/H, RX(+0.125)/RX(-0.125), and CZ/CZ; cases contain 192, 1536
and 12288 source gates. Copies and metrics are outside timed regions. Pass order
alternates. Other work was active on the host: these are diagnostic profiles,
not isolated performance estimates or end-to-end speedup claims.

Median milliseconds:

| Stage                             | 192 gates | 1536 gates | 12288 gates |
| --------------------------------- | --------: | ---------: | ----------: |
| Import                            |     1.125 |      6.518 |      65.197 |
| QC to QCO                         |     0.239 |      4.645 |      39.406 |
| Classical-only prototype          |     0.178 |      1.137 |      63.300 |
| Generic canonicalize              |     0.377 |      4.634 |      26.298 |
| Existing dedicated rotation merge |     0.303 |      5.028 |      69.377 |
| Current full cleanup              |     0.413 |      5.365 |      49.633 |
| QCO to QC                         |     0.533 |      7.241 |     101.981 |
| QC to QIR                         |     8.339 |     21.494 |     173.824 |
| LLVM text serialization           |     0.555 |      6.172 |      72.172 |
| Build matrix DD                   |     0.272 |      0.870 |      10.502 |
| Extract dense matrix              |     0.020 |      0.018 |       0.021 |

These passes do different work. At the largest size, raw/prototype IR contains
24591 operations; canonicalize/full cleanup leave 3, and rotation merging leaves
12303. The prototype is not consistently faster. Each resulting functionality
has one reachable DD node (identity), although its source has thousands of
operations. Process peak RSS across the complete harness was 102240 KiB; this
includes libraries, contexts, copies and allocator retention. It is neither
allocation volume nor occupied live IR/DD storage. Those two quantities were not
instrumented, so no allocator/storage savings are claimed.

## 2878's `QCO/Utils/Drivers.h` traversal requires verified linear scalar QCO

inlined quantum calls, supported scalar arities and a nonmutating callback. It
ignores classical dependencies and side effects and excludes tensor handles. Its
contract disqualifies a general mutating cleanup traversal. The reusable

## 2878 change is the existing greedy configuration disabling region

simplification, folding and constant CSE in gate-only synthesis sweeps. A future
extracted quantum pattern pass should first use that configuration and compare
identical inputs/outputs against the ordinary greedy driver. No traversal change
is proposed or benchmarked here.

### Ordered implementation slices and acceptance gates

1. **Preserve #2879 (merged).** Keep its independent importer fix/regressions.
   Gate: all seven fail on the old importer and pass on the fixed artifact;
   native H/H and controlled matrices remain observable. The prototype does not
   replace it.
2. **Classical preparation and structural branch selection.** Use the bounded
   folding pass as evidence, then share existing static-if/switch patterns with
   folding restricted to classical operations. Resolve dead branches before
   final verification; preserve diagnostics on reachable non-finite values.
   Gate: existing scalar/array/partial binding, dead-branch, dynamic-loop,
   side-effect, formal-helper and wire-correspondence regressions. Only then
   replace binding/inliner/classical-promotion scheduling.
3. **Extract quantum pattern ownership.** Move identity/approximate-zero folds,
   gate specialization and pair/composition patterns into shared population
   helpers and one explicit simplification pass. Preserve structural shells.
   Update direct synthesis/decomposition users in the same slice. Gate: explicit
   optimization still simplifies; canonical cleanup preserves native events;
   controlled phase, modifier numerical guards and target conformance pass.
4. **Migrate pipelines and serialization.** Explicitly schedule quantum
   simplification for general compilation; use required classical/structural
   preparation for simulation and native exports. Gate: exact ordered payload
   sequences, parameter/site/output order, arrays and profile diagnostics across
   OpenQASM, QIR, IQM and jeff. Decide barrier loss and unsupported
   timing/custom boundaries per format; reject unsupported strict contracts.
5. **Delete duplication; measure only an identified cost.** Remove obsolete
   registrations and redundant sweeps after the consumers above are migrated.
   Gate: phase-sensitive corpus and paired equal-output profiles. Instrument
   allocations/occupied storage only if a concrete memory claim warrants it.

Compatibility decisions still needed: whether public `cleanup()` changes its
historical optimizing behavior immediately or through a documented migration;
which APIs promise strict device-event preservation; exact versus approximate
optimization policy; and whether QIR barrier loss becomes a diagnostic or a
stated format limitation. Existing exported pattern-population APIs and external
canonicalize users need migration notes when ownership changes. A new native
provenance attribute is neither needed nor recommended.

### Original prototype validation

- Installed prototype wheel: 461 focused Python tests passed, including the
  complete MLIR, parameter-binding, Qiskit backend and serializer files selected
  for this investigation; #2879 regressions and formal-helper matrix included.
- Native validation: 335 compiler tests, 211 OpenQASM frontend tests and 246
  OpenQASM translation tests passed (792 total). Repository lint passed;
  whole-file C++ lint covered all five changed translation units and passed.
- Submitted payload/mutation probes passed against the installed artifact.
- No remote device jobs, production-wide pass migration, or rewrite-driver
  algorithm change. No claim of physical ZNE validation or recovered Catalyst
  history.

## First implementation slice after the upstream refresh

Deliver the explicit `fold-classical-values` pass, registration, generated pass
reference, and the sequence/formal-argument regressions. No implicit pipeline or
binding behavior changes in this slice. This is useful through the public
`run_pass_pipeline` entry point and as an explicit inliner default pipeline for
known straight-line gate bodies. It is not the complete native-submission fix.

A second specialist review rejected the proposed simple numeric-sweep then
branch-sweep binding integration: MLIR's expensive rewrite checks can verify
intermediate IR before the second sweep removes a dead branch containing a
non-finite gate parameter. Pass-manager verification ordering alone is not a
correctness argument. Before migrating binding, design preparation that prunes
known unreachable regions before evaluating their contents and validate it with
expensive rewrite checks enabled. Do not add another interpreter to achieve it.

The full SCF `IfOp` canonicalization set also contains non-static control-flow
rewrites; it is not a static-branch-only helper. Recollect operation selections
after every mutating rewrite phase, because removing a region destroys pointers
to its descendants. Reuse the focused QCO static-if helper where applicable.

Refreshed validation on `4f09df3267` plus this slice:

- 337 compiler, 211 OpenQASM frontend and 246 translation tests passed (794).
- Rebuilt and installed wheel: 482 selected Python tests passed,
  including #2879, parameter binding, backend/serializers and the new exact DD
  estimator. After adding the non-finite diagnostic regression, all 18 tests in
  the binding file passed (483 distinct tests across the two runs).
- Whole-file C++ lint passed for all three changed translation units. Generated
  MLIR pass documentation built successfully. Repository lint passed.
- No refreshed performance claim, expensive-rewrite-checks build, full
  repository suite, remote device job, or hosted CI run. No implicit pipeline
  migration.

The review-ready branch is `codex/classical-value-folding`. The prepared PR body
and raw refreshed validation logs live beside the original evidence bundle in
`/home/nvidia/.codex/experiments/cleanup-separation-550d/main-refresh/`.
