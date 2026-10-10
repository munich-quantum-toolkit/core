# Exception-free API audit

Status: findings applied. Baseline: PR #2545 at `1c131aade`, based on main
`b35d3334a` after #2733.

The audit traces native DD and JSON inputs, compiler diagnostics, Python
exceptions, QIR execution, and QDMI status boundaries. An independent design
review and LLVM/QIR and native/test specialists checked the remaining design. No
public input restrictions are needed for the accepted simplifications.

## Correctness fixes

- **QIR ownership:** `Runtime::reset` resets quantum state without freeing live
  classical allocations. Reference counts and runtime destruction own those
  allocations. A tuple retained in a global across two shots reproduced heap
  corruption before the fix; the regression now passes.
- **JIT lifetime:** static constructors and destructors bind the session
  runtime, like entry-point execution. This keeps their allocations out of a
  temporary thread's fallback runtime. The lifecycle test creates the session on
  another thread, reads its allocation, and checks destructor output.
  Worker-local output streams outlive their JIT sessions.
- **QDMI recovery:** the driver remains exception-enabled through session
  allocation and its C boundary catches. Its separate CAPI translation unit is
  unnecessary and has been merged back into `Driver.cpp`. An isolated Linux test
  injects allocation failure into the production shared library, checks
  `QDMI_ERROR_OUTOFMEM` and a null handle, then retries successfully. The former
  exception-disabled allocation path aborted or leaked the exception.
- **Status and exception categories:** a missing required provider symbol
  returns `QDMI_ERROR_FATAL`. Compiler target queries and submission use the
  shared binding adapter with their existing `ValueError` policy. Device lookup
  and direct QDMI bindings retain their own categories. The unsupported-output
  regression rejects the previous accidental change to `RuntimeError`.
- **Parallel diagnostics:** QCO sampling workers capture all diagnostic
  severities and replay them in worker order after joining. Serial and parallel
  division-by-zero sampling now retain the same category and message without
  leaking diagnostics to stderr. Previously only the parallel path raised a
  generic `RuntimeError`.

## Simplifications

- `evaluateJSON` parses its manifest once, dispatches its validated parameters
  through the existing registry, and evaluates the resulting variant. Typed
  parsing and evaluation share the manifest consistency check. Numeric-kind,
  case-ID, source-location, and diagnostic-precedence checks remain intact.
- Delete the unused private `LoadedDeviceAPI::create` factory; provider loading
  already belongs to the cached loader.
- Remove duplicate MSVC exception configuration from MLIR CMake. Keep the common
  exception policy and the existing LLVM flag normalization.
- Remove redundant DD extern-template declarations. Definitions and explicit
  instantiations remain out of line; Doxygen can resolve the public templates.
- Consolidate DDSIM isolation guidance into multi-program execution. Failures
  discard incomplete program results, while completed siblings keep theirs.
  Native API documentation describes current construction and error behavior.

## Test reductions and surviving checks

The benchmark/DD scope removes 581 net lines, including production dispatch
simplification. These deletions were checked against the surviving oracles:

- Family JSON tests retain round trips, semantic case IDs, resolved parameters,
  schemas, and count evaluation. The central suite retains envelope/registry
  checks, nested duplicate keys, field diagnostics, malformed counts, numeric
  limits, and altered manifests. Its manifest rejection cases now also exercise
  `evaluateJSON` directly.
- The shared invalid-JSON helper checks both failed status and diagnostic
  metadata; its old `void` callback discarded the returned status.
- Existing DD deserialization tests retain malformed and truncated text/binary
  input, nonfinite weights, references, and qubit limits. Complex phase round
  trips moved into the general serialization test. The unique throwing-stream
  test remains.

No numerical, input-boundary, or ABI validation was removed merely because it
looked defensive. In particular, mutable QIR globals and native ABI restrictions
remain supported and tested rather than being replaced with narrower inputs.

## Validation and limits

See the [execution plan](../plans/exception-free-core.md) for final commands and
results. Focused pre-fix reproducers covered QIR heap corruption, session memory
exhaustion, Python exception compatibility, and lost parallel diagnostics.

Matched whole-PR success-path measurements, including cold/reused DDSIM workers
and state transfer, remain acceptance work. Windows/macOS packaging and hosted
checks must validate the published revision; Linux checks do not prove those
platform contracts.
