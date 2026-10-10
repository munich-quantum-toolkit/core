# Compiler Program results

Status: implemented, reviewed, and validated locally.

## Scope

Use LLVM `FailureOr<T>` and `LogicalResult` for failing Program creation,
conversion, pass, and output APIs. Pass results through the benchmark
generators, compiler bindings, QDMI adapter, DDSIM worker, and tests. Preserve
ownership, MLIR diagnostics, Python exception categories, successful optional
absence, and ordinary predicates. Remove the private boolean linearity wrapper
so its callers use the verifier's result directly.

Keep native exception handling and Target/TargetEnvironment error ownership on
main's contracts. Do not introduce the native diagnostic bridge from #2545.
Include the requested QDMI configuration paragraph removal and the XDG-specific
HOME fallback for an unset or empty `XDG_CONFIG_HOME`.

## Completion

- [x] Update the APIs and all callers without changing supported behavior.
- [x] Run the ponytail review and address its findings: no actionable findings.
- [x] Validate native and Python consumers, regenerate stubs, and run lint.

## Validation

Use the release build and CTest, the rebuilt wheel's Python tests, stub
generation, repository lint, and whole-file C++ lint. Existing compiler tests
cover ownership, diagnostics, failed passes, binding, serialization, and output
errors; the registry regression covers the XDG fallback. Hosted CI and platform
results must be reported separately from local Linux checks.

The release build and CTest passed all 4,006 runnable tests; one existing SC
test was skipped. All 1,960 Python tests passed against a rebuilt wheel. Stub
generation left the Python APIs unchanged. Repository lint, whole-file C++ lint
on 31 files, and executable documentation with local link checks passed.
Documentation required clearing generated pages left over from #2545. Local
validation used Linux aarch64, GCC 13.3, LLVM/MLIR 23.1, and Python 3.14.
Windows, macOS, and hosted CI remain outside the local validation.
