# Native implementation validation

Baseline: Core `1d384a5d23aa1f02b26fd5684dab586ef26167fe`, with the uncommitted
`feat/native-equivalence` implementation in
`/home/nvidia/.codex/experiments/native-equivalence-20261025/core`. Validation
used the normal Release preset, GCC 13, and LLVM/MLIR 23.1.0 on 2026-10-25.
These are correctness checks, not performance measurements.

The final successful result for each affected binary is recorded in
[native-tests.json](native-tests.json). Exact GoogleTest summaries and focused
trace regressions are retained in
[native-test-results.txt](native-test-results.txt). The JSON also identifies the
full local `/tmp` logs, which are not copied into this artifact directory.

| Binary                                    | Expected        | Observed      |
| ----------------------------------------- | --------------- | ------------- |
| `mqt-core-mlir-unittest-qco-utils`        | All tests pass  | 300/300       |
| `mqt-core-mlir-unittests-compiler`        | All tests pass  | 321/321       |
| `mqt-core-mlir-unittests-mqt-transforms`  | All tests pass  | 31/31         |
| `mqt-core-mlir-unittest-qc-ir`            | All tests pass  | 369/369       |
| `mqt-core-mlir-unittest-qco-ir`           | All tests pass  | 586/586       |
| `mqt-core-mlir-unittest-target-synthesis` | All tests pass  | 99/99         |
| `mqt-core-mlir-unittest-optimizations`    | All tests pass  | 207/207       |
| `mqt-core-dd-test`                        | All tests pass  | 220/220       |
| **Total**                                 | **No failures** | **2133/2133** |

QCO Utils includes all 109 previous runtime tests, ten new runtime/trace tests,
six independent dense verification tests, and 15 interface tests. The migrated
compiler, canonicalization, synthesis, and optimization suites exercise all 53
replaced assertion call sites across 12 `.cpp` files. The assertion still uses
phase-preserving construction and each caller's original maximum-entry
tolerance. Independent literal/dense matrix references remain in the tests.

Build command, from the worktree root:

```sh
cmake --build --preset release --target \
  mqt-core-mlir-unittest-qco-utils \
  mqt-core-mlir-unittests-compiler \
  mqt-core-mlir-unittests-mqt-transforms \
  mqt-core-mlir-unittest-qc-ir \
  mqt-core-mlir-unittest-qco-ir \
  mqt-core-mlir-unittest-target-synthesis \
  mqt-core-mlir-unittest-optimizations \
  mqt-core-dd-test -j4
```

The build passed; its local log is
`/tmp/native-equivalence-final-native-build.log`. Run each binary listed in the
JSON as `build/release/<binary>` without a GoogleTest filter. The initial sweep
used three concurrent processes and a 180-second limit per process. After the
corrections below, QCO Utils and Compiler were rebuilt and rerun in full; the
last fixture-only correction required only a Compiler rebuild/rerun. Final
correction build logs are `/tmp/native-equivalence-final-corrections-build.log`
and `/tmp/native-equivalence-final-fixtures-build.log`. The final zero-threshold
numerical-guard correction was rebuilt with
`cmake --build --preset release --target mqt-core-mlir-unittest-qco-utils -j4`
and rerun through all 300 QCO Utils tests. Its logs are
`/tmp/native-equivalence-zero-boundary-build.log` and
`/tmp/native-equivalence-final-zero-boundary-qco-utils.log`. The last production
correction restricted simulation eligibility when private workspace must be
proven clean for all quantum inputs. The same QCO Utils build command and all
300 tests passed again; the latest logs are
`/tmp/native-equivalence-workspace-eligibility-build.log` and
`/tmp/native-equivalence-final-workspace-eligibility-qco-utils.log`.

The failing checks resolved during validation were:

- A copied deferred measurement token was initially lost by CBit load, and a
  later concrete overwrite could leave a stale token. The shared interpreter's
  trace path now preserves/cancels those tokens; copy and loop-overwrite tests
  pass. Existing sampling behavior and all previous runtime tests pass.
- The strict process-residual upper bound rejected the analytically orthogonal
  dense corpus case because binary64 accumulation produced a value just above
  two. The final guard admits only the smaller of the configured numerical
  tolerance and `64 * epsilon`, clamps that admitted overshoot, and rejects
  larger/nonfinite values. The unchanged independent dense reference now passes.
- Approximate distance zero at acceptance threshold zero now remains
  inconclusive when the configured numerical guard is positive, consistently
  with other values inside the numerical uncertainty band. The existing
  approximation boundary regression passes with this final behavior; exact
  checking is unchanged.
- Automatic checking could let a simulation mismatch cancel construction before
  construction checked the required all-input clean-workspace premise. Such
  contracts now require construction: simulation is eligible for exact checking
  only when there are no free quantum inputs or neither program has a private
  output cleanliness obligation. The existing workspace regression now also
  requires automatic dirty-workspace checking and forced simulation on a clean
  all-input workspace program to report unsupported. See
  [the observed failure and regression](workspace-eligibility.md).
- A dense matrix accepted by the IR tolerance can become nonunitary after a
  large power. Resolved execution checks the resulting matrix's Gram entries
  against the existing `DENSE_UNITARY_TOLERANCE`, polls cancellation/deadline,
  and allocates no second dense matrix. The tests reject exponents `1e10`,
  `1e12`, and `1e14` for `diag(1.00000000004, 1.00000000004)` before DD replay.
  Exact/process and construction/alternating/automatic verification all return
  unsupported. See the separate [power probe](pow-amplification/README.md).
- The new native interface exposed previously implicit physical wire identities
  in compiler fixtures. This was corrected without relaxing the decoder or the
  phase/entry-error contract, as detailed below.

Three compiler tests start with already-physical `qco.static` inputs:
`TargetSynthesisResynthesizesTwoQubitBlocks`,
`TargetCompilationFusesOnlyWithUsableNativeBasis`, and
`TargetCompilationCancelsInteractionsBeforeRouting`. Their fixture-defined
identity is literally `[0,1]` or `[0,1,2,3,4]`, fixed on an independent
reference before compilation. `PrepareTargetCompilationPass` rejects prior
layout metadata and leaves raw physical input without source provenance. In
`Mapping.cpp`, `needsPlacement` returns false for entry static roots; both
`PlacementPass` and `MappingPass` return immediately in that case. Consequently,
these fixtures supply their fixed physical identity after compilation, only
after asserting that no compiler-produced layout exists. They never infer an
identity from transformed operations or overwrite a compiler permutation.
Production provenance for this raw-physical input mode remains a separate
producer API candidate; the logical/allocation input path is covered by the
native interface and compilation tests.

The combined-controls pipeline test also retains all three declared inputs until
the target pipeline records their identity. Its redundant preliminary cleanup
previously erased an idle input before the baseline clone; both pipelines still
run their own cleanup. Native-synthesis pass fixtures declare the known static
input identity before the transformation, including the synthesis-only
checkpoint after mapping.

No test tolerance was increased, no phase-sensitive assertion was removed, and
the old standalone test unitary-construction engine is deleted.
`git diff --check` passed. C++ lint, Python/stub validation, and QCEC
differential validation are reported separately by their owning checks.
