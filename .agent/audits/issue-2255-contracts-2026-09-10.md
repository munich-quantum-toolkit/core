🤖 *AI text below* 🤖 <!-- rumdl-disable-line MD041 -->

# MLIR audit resolution — issue #2255

Status: accepted findings resolved, except C1, owned by
[PR #2502](https://github.com/munich-quantum-toolkit/core/pull/2502).
Date: 2026-09-10. Rebased onto main at `7d9061796`; implementation originally
started from `4faf68e3a`. Scope:
[issue #2255](https://github.com/munich-quantum-toolkit/core/issues/2255),
covering QC/QCO modifiers, conversions, optimization and mapping passes, QTensor
cleanup, and QIR metadata.

The original counterexamples were measured on `d994fe683`; the
[report and raw evidence at `b31e80e66`](https://github.com/munich-quantum-toolkit/core/tree/b31e80e66b01bea9e052962de6d65c951220d3b5/.agent/audits/issue-2255-evidence)
remain available in history. Durable regression tests now replace the temporary
probe harness and copied inputs.

## Resolved findings

| ID  | Change and retained evidence                                                                                                                                                                                                                                                        |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| F1  | Mapping diagnoses opaque classical effects before mutation instead of silently reversing calls. `RejectOpaqueClassicalEffectsBeforeMutation` checks valid input, the diagnostic, and unchanged output. Existing CBit ordering tests remain.                                         |
| F2  | Standalone QCO-to-jeff diagnoses modifiers outside its single-unitary, full-width, argument-order normal form before mutation. `RejectsNonNormalizedModifiersBeforeMutation` covers untouched targets and reordered gate operands. Existing normalized modifier round trips remain. |
| F3  | XX±YY matrices use an explicit −i factor and negative conjugate instead of subtracting π/2 from large angles. `XXPlusMinusYYRemainUnitaryForLargeBeta` covers ±10¹⁶ and the largest finite double; ordinary-angle matrix oracles remain.                                            |
| F4  | Measurement/Hadamard lifting uses nullable typed producer queries. Both `MeasuresBlockArguments` regressions verify IR and linearity before/after the pass and retain the actual measured wire.                                                                                     |
| F5  | Full quantum unrolling uses the existing empty-body workaround so LLVM remaps terminator-only permutations. `PreservesYieldOnlyPermutation` checks the actual wires after 1, 2, 3, and 4 iterations.                                                                                |
| F6  | Placement discovery follows only a flat extract/insert/deallocate chain. `RejectTensorControlFlowBeforeMutation` checks valid tensor control flow, diagnostics, and unchanged IR in both placement and mapping.                                                                     |
| F7  | QIR capacity follows known QIS operand roles and all pointer uses. Metadata tests establish shared-ID capacity 8, result-read qubit capacity 1, and safe handling of dynamic pointer arguments; sparse-ID and overflow tests remain.                                                |
| F8  | Resolved by main in #2495. `ClassifiesUnconditionalBackEdge` covers unconditional loops; this PR retains `MetadataIncludesMeasurementLatch` for measurement-dependent exits. Duplicate production code and the unconditional-loop test were dropped.                                |
| F9  | `qco.index_switch` prints the bare attribute dictionary its parser accepts. `DefaultOnlyIndexSwitchParser` now round-trips and retains a discardable attribute.                                                                                                                     |
| F10 | QC modifier body arguments must match targets; QCO control results must match control inputs. Compact owning-verifier tests cover both signature gaps and positional yields for all three modifiers.                                                                                |
| C2  | Deleted the unused QTensor helper header and tests that only exercised those helpers. Actual alias/reset and index-boundary tests remain.                                                                                                                                           |
| C3  | Removed ancestor-modifier checks from QTensor extract/insert verifiers. The nested-register-access test now checks the owning modifier; local index checks remain.                                                                                                                  |
| C4  | Removed the impossible deallocation/map-failure branches and duplicate mapped-index vector from register shrinking. Existing QTensor transform tests remain.                                                                                                                        |

Regressions live in `mlir/unittests/Conversion/{JeffRoundTrip,QCOToQC}` and
`mlir/unittests/Dialect/{QC,QCO,QIR,QTensor}`. The phase-extraction fixture in
`mlir/unittests/Dialect/MQT/Transforms` retains its original purpose with valid
positional yields.

## Contract decisions and duplicate work

- The user accepted positional correspondence between modifier arguments and
  yielded qubits. The QCO region verifier follows unitary wire ties after nested
  operations have been verified. Real SWAP gates and reordered gate operands
  remain valid; a bare yield permutation does not.
- This removes 13 repeated canonicalizer guards and redundant modifier checks in
  QCO-to-QC. The old permutation-only canonicalization test file and three
  invalid conversion cases were replaced by owning-verifier coverage. Tests for
  valid gate operand reordering, modifiers, and control-flow permutations
  remain.
- Modifier input uniqueness is checked once in the shared verifier. SSA result
  identity and positional yields make the old control-output uniqueness checks
  redundant. The regression checks duplicate inputs and yields for Ctrl/Inv/Pow.
- Mapping retains its deterministic scheduler and CBit effect model. A trial
  that serialized all effects broke routing dominance repair; it was discarded.
  Other classical side effects and tensor control flow need lowering first.
- QIR metadata recognizes the compiler's known QIS calls. The result-read
  counterexample uses a QIS spelling, while this compiler emits
  `__quantum__rt__read_result`; the loop regression uses the latter. This change
  does not add a runtime alias or claim support for arbitrary QIS ABIs.
- C1 duplicates finding 8 in #2502. Rechecked that PR at `da67d7fc1`: it now
  implements removal of the QC-to-QCO modifier-verification walk and relocates
  its tests. That patch is deliberately not duplicated here. Its QIR output
  ordering, comparator, diagnostics, and lifetime fixes are distinct.
- F8 is now implemented in main by #2495. The rebase keeps main's implementation
  and unconditional-back-edge test, dropping the duplicate fix and regression
  from this PR. The distinct measurement-latch regression remains here.

## Validation and limits

The affected GoogleTest binaries are built with assertions enabled against
LLVM/MLIR 23.1.0. Post-rebase validation at `d0bcfa697`: 2,687 tests pass across
15 binaries (QC/QCO/QTensor/QIR IR, QCO utilities and optimizations, mapping,
decomposition, target synthesis, phase normalization, QTensor transforms,
QC↔QCO, jeff round trips, and compiler pipelines). Run each under
`build/release/mlir/unittests` with `--gtest_brief=1`.

`uvx nox -s lint` and full changed-file `uvx nox -s cpp-lint` pass. The latter
checked all 32 changed C++ files with zero findings at `d0bcfa697`. Context-only
test setup now uses constructors without naming suppressions. No sanitizer,
hardware execution, or measured speedup is claimed.

Remaining candidates are not accepted findings: QC-to-QIR entry arguments and
measurement-bearing helper support; permissive contexts with unregistered
operations at import; and unsupported multi-target/zero-control Hadamard matcher
shapes. These need a support decision or corrected semantic reproducers. The old
controlled-SWAP yield-permutation concern is superseded by the new modifier
contract. No blanket invalid-IR pass checks or previously closed broad
resource-management work were revived.
