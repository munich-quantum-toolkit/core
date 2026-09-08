# Controlled synthesis test runtime

Status: implemented and validated. Baseline:
`1f25f40667079f909685992653549b12ce0e0be6` with the pending MCY support, shared
Pauli tests, and rotation CX budgets. The open PR is
[#2467](https://github.com/munich-quantum-toolkit/core/pull/2467).

## Scope and contract

Speed up `test_multi_controlled_decomposition.cpp` without removing control
widths, numeric or runtime angles, gate-count bounds, exact-phase comparisons,
borrowed-control restoration, or input/output IR and linearity verification.
Production synthesis and the Python test matrix are unchanged.

## Findings

The single-pass helper verified its output twice: once in `PassManager::run` and
again in an explicit `verify` call. Sampling the rotation resource test
attributed 658 samples to the first check and 677 to the second, compared with
712 in greedy rewriting. The helper now explicitly enables the pass manager's
verifier and retains the separate output linearity check. Input verification is
unchanged. Both output checks covered the same module after the same pass.

The large Pauli full-operator tests spend most of their time in DD matrix
multiplication. Reusing packages solely to avoid allocations is not justified by
this profile. Switching the three matrix-only helpers to the existing
unitary-simulation DD configuration showed no benefit: the three-run median for
the 25 Pauli-at-eight-controls and numeric/runtime rotation tests changed from
5.276 to 5.294 seconds. This experiment was reverted.

## Validation

The baseline Debug binary passed all 297 decomposition tests in 42.4 seconds on
macOS ARM64 with AppleClang 21 and LLVM/MLIR 23.1. The run included a short
sampling interval, so it is diagnostic rather than a controlled speed ratio.

A subsequent unprofiled three-run comparison of
`MultiControlledDecompositionTest.RotationsUseLinearResourcesWithoutExtraQubits`
used the original binary and the rebuilt single-verifier binary serially, with
no concurrent build. The median fell from 2.035 to 1.477 seconds (27%). All six
runs passed with all 96 axis, width, and angle-kind combinations retained. The
34-test MCY and rotation-resource filter also passed three times per binary. Its
median fell from 10.528 to 8.910 seconds (15%). Compilation and profiling time
are excluded from test durations.

The final rebuilt binary passes all 297 tests. The full changed-file
`uvx nox -s cpp-lint -- ec799daa09f855bd0edcbc5592a5fedd90836516` check reports
zero findings, and `uvx nox -s lint` passes. The final full-suite run overlapped
with C++ lint and is not used for a whole-suite speedup claim.
