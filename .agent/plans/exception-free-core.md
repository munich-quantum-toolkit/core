# Explicit errors across Core and compiler APIs

Status: complete.

## Goal and scope

Replace exception-based recoverable failures in QDMI, benchmark, DD, and QIR
execution APIs with explicit results. Keep C++20 and standalone builds without
LLVM. Python keeps its normal exception interface. The baseline is PR #2545 at
`f7553f5a5`.

## Decisions

- Standalone APIs use C++20 value/error variants and optional errors for
  operations without a result. Errors retain their category and message.
- LLVM services use `llvm::Expected` and `llvm::Error`. MLIR operations use
  `LogicalResult` or `FailureOr` with diagnostics.
- Each operation has one canonical fallible interface. Remove throwing/try
  pairs; translate errors only at LLVM, Python, CLI, and QDMI C boundaries.
- Fallible construction uses factories; RAII continues to own partial resources.
- Python bindings translate explicit errors with native nanobind primitives.
  Python call signatures remain stable where practical.
- Preserve unsupported optional properties, warning behavior, strict JSON
  validation and canonical output, numerical checks, job states, and cleanup.
- Runtime ABI callbacks require safe propagation through generated code. An
  error flag alone is insufficient: execution must stop before dependent work.
- Recoverable input and operational failures must remain recoverable. Allocation
  exhaustion and unexpected dependency exceptions are nonrecoverable in the
  native algorithms. QIR ABI functions are noexcept, so exceptions cannot unwind
  through generated frames. Input checks continue to return errors.

## Progress

- [x] Canonical QDMI result APIs, driver/configuration propagation, and callers.
- [x] Benchmark factories, parsing/evaluation results, and diagnostic transport.
- [x] LLVM-native JIT setup and DD results through MLIR consumers.
- [x] Safe QIR runtime failure propagation and exception build policy.
- [x] Binding adaptation, stubs, migration documentation, and final validation.

## Validation

Use the release and release-no-mlir presets. Run subsystem tests while
migrating, then native and Python regression suites, stub regeneration, C++ and
repository lint. Verify actual compile commands for migrated targets. Retain
provider failure/recovery, malformed input, partial initialization, runtime
failure, Python exception category, and CLI file-safety checks.

Final checks:

- Release build and CTest: 3557 pass; the existing unsupported SC job-ID test
  skips. The suite includes runtime error propagation, session reuse, provider
  status preservation, strict JSON validation, and CLI regression checks.
- Standalone build without LLVM and CTest: 579 pass; the same SC check skips.
- Python 3.14 suite with both packaged providers: 1420 pass.
- Stub generation passes and leaves Python signatures unchanged.
- Complete executable documentation and local documentation links pass.
- Repository lint and the standard C++ lint session pass. A supplementary
  whole-file clang-tidy scan covers every changed C++ source; affected files
  were checked again after fixes.
- Actual compile commands confirm C++20 and disabled exceptions in the migrated
  algorithms. Python and C ABI exception barriers retain exception handling.

The error categories intentionally correct four Python test expectations:
invalid DD arguments raise `ValueError`, and unsupported QDMI submissions raise
`RuntimeError` through both the direct and compiled-program interfaces.

Local release builds disable IPO to avoid duplicate LLVM TypeID definitions from
GCC LTO and the installed LLVM static libraries. This is a local configure
option, not a repository policy change. The standard C++ linter omitted most
uncommitted modified files, so the supplementary scan verifies its selection
gap.

The follow-up complexity review removes 32 implementation lines: optional SC
fields reuse required-field validation, QIR instrumentation shares attribute
cleanup and keeps one current failure block, DD deserialization drops an
obsolete array reset, and registry discovery uses one ordered loop.

The native and standalone suites and the Python tests pass after this cleanup.
The update belongs to PR #2545.
