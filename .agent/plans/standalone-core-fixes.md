# Standalone fixes from the exception-free work

Status: validated. Nine independent commits based on main after #2730.

## Scope

Base: `4e3e14380` (`main` after #2730). Source: `40b54e698` in #2545. Preserve
the throwing APIs and optional LLVM/MLIR build. Keep the larger dependency,
result-type, and diagnostic migrations in #2545.

## Changes

- [x] Validate native QIR calling conventions, ABI attributes, and call types.
- [x] Return output-write failures without an LLVM stream-destructor abort.
- [x] Resolve provider resources beside relatively loaded libraries.
- [x] Release scalar DD roots when functionality construction fails.
- [x] Validate serialized DDs in the source file introduced by #2730.
- [x] Remove failure propagation from infallible compiler helpers.
- [x] Remove repeated benchmark validation, probability work, and parsing.
- [x] Include pass-pipeline parser details in the owning MLIR diagnostic.
- [x] Review minimality and pass native, Python, and lint checks.

## Validation

Use focused regressions against the unchanged implementation where practical,
then run native release tests, affected Python tests, the no-MLIR build, and
whole-file C++ and repository lint. Test malformed input alongside valid input;
retain numerical limits, empty outcomes, ownership, and external API behavior.
Each standalone commit includes its tests. Hosted CI is separate from local
validation.

## Evidence

The new regressions reproduced the ABI, stream-write, provider-path, scalar
root, and malformed-DD defects on the unchanged implementation. Native release
tests passed (4,004 runnable tests), as did a fresh no-MLIR build (770 tests)
and the Python DD suite (60 tests). Each native configuration skipped the
existing unsupported job-ID test. Repository lint and whole-file C++ lint
passed; the latter checked all 23 selected files. The nine serialization tests
passed again after the test-only lint fixes.

Review retained the existing DD parser, optional LLVM dependency, and throwing
APIs. It also caught and corrected rejection of function-level `alignstack`. DD
validation covers record structure and non-finite weights; broader numeric
normalization changes remain outside this extraction.
