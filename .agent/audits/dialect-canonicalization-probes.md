# Evidence for the remaining canonicalization fixes

Status: narrowed diff rebuilt and tested locally. Base: `7e2a2679f`. The
[main audit](dialect-canonicalization.md) links the comprehensive record at
`79e9c347b` and the three extracted PRs. This appendix retains only matrix,
identity-folding, local-analysis, and QTensor scaling evidence.

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

These permanent regressions passed after rebuilding the narrowed diff. The
[main audit](dialect-canonicalization.md#validation) records the current
results.

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
The large rewrite/cache redesign is deferred. Input and output passed ordinary
verification and QCO linearity. Times are single local runs without thresholds.

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

The reset pattern's source establishes N(N-1)/2 prior-access visits across this
family. Caching each decoded constant reduces repeated decoding but does not
change that asymptotic bound. Measurements alone do not justify mutable cache
state without invalidation rules.

The
[archived probe appendix](https://github.com/munich-quantum-toolkit/core/blob/79e9c347b30b6c6639b6422a206fdc78dba6bd16/.agent/audits/dialect-canonicalization-probes.md)
retains the complete counted input generator and reproduction procedure.

## Reproduction and validation

The scaling measurements above used the investigation baseline `b75b02fa9` and
LLVM/MLIR 23.1.0. They establish a cost to revisit, not a current speedup. The
permanent tests name the owning contract; their source must be rebuilt before
claiming a result.

Use the release preset and pinned dependencies. The relevant targets are
`mqt-core-mlir-unittest-qc-ir`, `mqt-core-mlir-unittest-qco-ir`,
`mqt-core-mlir-unittest-cbit-ir`, `mqt-core-mlir-unittest-qtensor-ir`, and
`mqt-core-mlir-unittests-mqt-utils`.

Use exact suite and test names when selecting permanent regressions. A missing
or empty selection is not a passing probe. The narrowed diff passed 1,150 tests
in six focused binaries and 3,224 configured C++ tests, with one optional QDMI
test skipped. The main audit records build and lint results. Results from the
combined change at `79e9c347b` remain historical.
