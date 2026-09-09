# Evidence for the remaining canonicalization fixes

This appendix retains reproducible evidence and historical measurements. The
[main audit](dialect-canonicalization.md) owns current scope, decisions, and
validation status. It links the comprehensive record at `79e9c347b` and the
three extracted changes now merged into main.

## Matrix and U-power regressions

The
[QCO numeric suite](../../mlir/unittests/Dialect/QCO/IR/test_qco_numeric_canonicalization.cpp)
retains five tests:

- `RMatrixPreservesLargeAxisAngles` checks an independent axis matrix and
  unitarity at `phi=1e16` and `1e308`. The former `exp(i*(+/-phi-pi/2))` formula
  loses the quarter turn. Keeping `-i` outside the exponential avoids the loss.
- `RMergePreservesLargeAxisAngles` and `RPowerPreservesLargeAxisAngles` compare
  untouched inputs with rewritten outputs, including global phase.
- `UMatricesPreserveLargeEulerAngles` uses independent
  `P(phi) * RY(theta) * P(lambda)` products and checks unitarity. Fixed offsets
  such as `lambda+pi` disappear at `lambda=1e16`; same-sign phase angles near
  `1e308` overflow when added. Separate phase factors avoid both failures.
- `UToU2PreservesLargeEulerAngles` tests a consumer of the shared U/U2 matrix.

The
[QC](../../mlir/unittests/Dialect/QC/IR/test_qc_modifier_canonicalization.cpp)
and
[QCO](../../mlir/unittests/Dialect/QCO/IR/test_qco_modifier_canonicalization.cpp)
modifier suites retain `InverseU2PreservesLargeAngles`,
`InverseU2PreservesDynamicAngles`, `InverseU2PreservesFullUnitary`, and
`InverseControlledU2PreservesFullUnitary`. They cover sign-only U parameters,
dynamic values, and ordinary or controlled full-matrix equivalence. QC uses
QC-to-QCO conversion only to obtain the matrix oracle. No conversion pattern is
changed.

[Gate-power tests](../../mlir/unittests/Dialect/MQT/Utils/test_gate_powering.cpp)
check `PositiveIntegerPowersPreserveFullMatrix`,
`DiagonalAndAntiDiagonalPowersPreserveGlobalPhase`,
`NearGimbalPowersPreserveFullMatrix`, and
`LargeEulerPhasePowersPreserveFullMatrix`. Their oracle uses sequential matrix
products; the implementation uses binary powering. Unsupported exponents and
nonfinite angles retain separate rejection tests. Positive integral exponents up
to 1024 remain subject to the `5e-13` reconstruction bound.

### Reconstruction rejection and phase-factor reuse

At the rebased baseline `84b5cfd32`, removing the final reconstruction/check
block in a disposable GatePowering.cpp copy still passed all ten gate-power
utility tests. The finite input
`(0.615926832310562, -2.7139721469341298, -2.7602783230969417, 1024)` was
rejected by the unmodified helper under both Clang 23 and GCC 13 on DGX Spark.
Without the guard, the reconstructed matrix had maximum entry error
`3.6334658662220525e-12` against a binary matrix power, over seven times the
`5e-13` bound. The new `RejectsFinitePowerBeyondReconstructionBound` test
requires rejection; `RejectedPowURemainsUnchanged` checks the controlled QCO
consumer. The new utility test fails when the reconstruction guard is removed
from a disposable helper copy. Exact behavior on other math libraries remains
subject to hosted validation.

For R's matrix, real finite parameters imply `m01 = -conj(m10)`. Reusing the
lower-left entry removes one exponential. An isolated Clang 23 `-O3` benchmark
used one million varying-axis evaluations per sample, seven samples, noinline
functions, and a consumed checksum. Median time was 28.548 ms with two
exponentials and 17.630 ms with conjugation, 38.2% lower on this host. Zero,
negative, ordinary, `1e16`, and `1e308` cases agreed within `1e-15`. This is a
helper microbenchmark, not a compiler or simulation speedup. Existing
independent axis-matrix, unitarity, merge, and power tests remain the semantic
oracles.

## Identity and local-analysis regressions

[QCO gate tests](../../mlir/unittests/Dialect/QCO/IR/test_qco_canonicalization.cpp)
check H/X/Y/Z pair cancellation through `PatternApplicator`, then immediately
verify IR and QCO linearity. This catches the rejected `Involution` migration: a
folder can forward through a producer but cannot erase that producer, leaving an
extra use until DCE. Pipeline-only tests had hidden this violation. The same
suite retains unlike-gate, identity operand-order, `-I`, and nonzero-phase
cases, including a phase of `1e-16` that an approximate identity predicate would
erase.

[QTensor pair tests](../../mlir/unittests/Dialect/QTensor/IR/test_qtensor_pair_canonicalization.cpp)
retain `SameDynamicIndexCancellationPreservesLinearityBeforeDCE` and
`DifferentDynamicIndicesKeepExtractAndInsert`. The first establishes the same
immediate-linearity contract for Insert cancellation; the second preserves the
uncertain-alias barrier.

The baseline live-snapshot CBit probe stored true, read the whole register into
a live result, then loaded the stored bit. It retained one unnecessary Load
(exit 1). The permanent
[CBit tests](../../mlir/unittests/Dialect/CBit/IR/test_cbit_canonicalization.cpp)
retain `ForwardsStoredBitAcrossReadOnlySnapshot` and
`DoesNotForwardAcrossWholeRegisterWrite`. Their fixture loads Arith explicitly;
runtime dialect dependencies are covered by the extracted change.

The
[QCO If test](../../mlir/unittests/Dialect/QCO/IR/test_qco_control_flow_canonicalization.cpp),
`SharesEarliestClassicalResultAndPreservesLinearSuffix`, checks ordered yield
pairs, the earliest representative, distinct reversed pairs, unused and
branch-independent results, and the quantum suffix. Source analysis establishes
the replacement of quadratic pair comparisons with expected linear map lookup.

The existing
[QTensor IR tests](../../mlir/unittests/Dialect/QTensor/IR/test_qtensor_ir.cpp),
`ResetAfterExtractThroughCommutingInsertIsEliminated` and
`ResetAfterExtractThroughSameIndexInsertIsNotEliminated`, retain the match-set
oracle for decoding each provenance constant once.

## Historical QTensor scaling

These measurements describe the investigation baseline, not a post-fix speedup.
Adjacent commuting remains deferred; main already batches fresh-slot resets.
Input and output passed ordinary verification and QCO linearity. Times are
single local runs without thresholds.

| Accesses N | Successful adjacent commutations |               Time |
| ---------- | -------------------------------: | -----------------: |
| 16         |                              120 |   202 microseconds |
| 64         |                            2,016 |   640 microseconds |
| 256        |                           32,640 | 9,206 microseconds |

The input is a tensor with N successive distinct constant-index
`extract; H; insert` accesses. Only `qtensor::InsertOp` canonicalization
patterns are registered. Each success moves one insert across one extraction, so
the required all-extracts-before-inserts order takes exactly N(N-1)/2 successes.

For reset provenance, allocate N qubits, extract each index in order, apply
`reset; H` to every scalar, insert the results in reverse index order, and
return the tensor. Register only `qtensor::ExtractOp` canonicalization patterns.
The historical runs removed exactly N resets:

| N     | Successful rewrites |                Time |
| ----- | ------------------: | ------------------: |
| 64    |                  64 |    232 microseconds |
| 256   |                 256 |  2,404 microseconds |
| 1,024 |               1,024 | 34,883 microseconds |

The historical reset pattern made N(N-1)/2 prior-access visits across this
family. Main now batches these fresh-slot resets using one allocation proof and
a forward linear-chain walk. These old times do not characterize that
implementation. They also do not establish a bound for arbitrary tensor graphs.
No mutable provenance cache is needed for the supported batch.

The
[archived probe appendix](https://github.com/munich-quantum-toolkit/core/blob/79e9c347b30b6c6639b6422a206fdc78dba6bd16/.agent/audits/dialect-canonicalization-probes.md)
retains the complete counted input generator and reproduction procedure.

## Reproduction

The scaling measurements used baseline `b75b02fa9` and LLVM/MLIR 23.1.0. The
archived appendix above retains the generators and commands. The new regression
sources preserve the numerical inputs. Rebuild before running the tests; use the
native release preset and exact suite/test names. An empty selection is not a
passing probe. See the [main audit](dialect-canonicalization.md#validation) for
current validation rather than duplicating results here.
