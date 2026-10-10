# Exception-free API reassessment

Rebased PR #2545 on main `da9ce44fc` after #2731. Validation of the rebased
implementation is recorded in the
[execution plan](../plans/exception-free-core.md).

## Next separations

1. **Require LLVM/MLIR for native development.** Remove the compiler-disabled
   option, presets, CI row, and conditional build graph. Replace the private
   SHA-256 implementation and QDMI unreachable helper with LLVM facilities. Keep
   existing native throwing APIs and exception settings. Link the targets that
   use LLVM directly; do not add CoreSupport as a dependency wrapper. Installed
   Development consumers need SDK discovery, while wheel Runtime consumers must
   remain SDK-free. This is a small build-contract change with independent
   value.

2. **Use upstream results for compiler Program APIs.** Change failing
   `optional<T>` and status `bool` returns to `FailureOr<T>` and `LogicalResult`
   in Programs, Pipeline, and ParameterBinding, then adapt their consumers.
   Retain existing MLIR diagnostics, Python exception classes, successful
   absence, predicates, and infallible accessors. These targets already require
   LLVM/MLIR, so this split does not depend on the first one. Most of its file
   count comes from existing tests and bindings.

Do not extract the compiler directory wholesale. Target and TargetEnvironment
currently own errors through `llvm::Expected`; converting them to status-only
results changes diagnostic ownership. QDMI compilation adapters also depend on
native exception boundaries. Both belong with the remaining error migration.

## Remaining PR scope

- Native DD, benchmark, QDMI client, and private driver result APIs, including
  fallible factories and explicitly infallible owned-state operations.
- Scoped diagnostics, original QDMI statuses, shared Python invocation, and test
  capture.
- Private JSON exception boundaries and exception-disabled algorithm targets.
- QCO DD and QIR integration, allocation-output recovery, and fatal unhandled
  QIR runtime errors.
- Structured diagnostic transport through the existing DDSIM worker boundary.
  Worker isolation, reuse, cancellation, and packaging already exist on main.

The merged fixes retain their coverage after adaptation: malformed and skipped
DD levels, zero-count invalid benchmark outcomes, scalar DD cleanup, compiler
stream failures, pass-parser details, relative module lookup, and QIR native ABI
checks. DD deserialization remains out of line. The benchmark generator no
longer needs exception unwinding because JSON failures return at their boundary.

## Open compatibility and acceptance work

- `submit_program` changes an unsupported QIR-output request from `ValueError`
  to `RuntimeError` in `test/python/qdmi/test_compilation.py`. This conflicts
  with the intended Python compatibility contract. Preserve the existing
  category before accepting #2545; do not carry this change into the Program
  result PR.
- Matched success-path performance measurements, including cold and reused DDSIM
  workers and state transfer, remain outstanding for the final revision.
- Windows/macOS packaging and hosted checks must validate the published head.
  Local Linux results do not establish those platform contracts.
