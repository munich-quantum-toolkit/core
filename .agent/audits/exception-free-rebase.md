# Exception-free API reassessment

Baseline: PR #2545 at `cfc90e2b8`, rebased on main `b35d3334a` after #2733. The
findings below remain open; no audit fixes have been applied. Validation is
recorded in the [execution plan](../plans/exception-free-core.md).

The audit follows native inputs through DD and JSON processing, compiler and
Python diagnostics, QIR execution, and QDMI status boundaries. The assumed load
is local library use and concurrent DDSIM jobs. The LLVM dependency, utility
reuse, and compiler Program result APIs are now on main.

## Findings

### 1. Preserve live classical QIR allocations between shots

`Runtime::reset` clears `allocations_`; `JitSession::sampleWithRuntime` invokes
it before each ordinary shot. Arrays and tuples now register in this ownership
map. A shot that retains an array or tuple in a mutable LLVM global leaves a
dangling pointer for the next shot, even while its reference count is positive.

Persistent classical state is supported: `QIRBatchSampling`'s
`ExecutesClassicalSideEffectsOnEveryShot` test relies on a global counter, and
the QIR sampling documentation retains classical memory accesses. Keep classical
allocations until explicit reference-count release or runtime destruction;
remove their cleanup from the quantum reset. Add a sampling regression retaining
one classical object across shots. This finding follows the ownership paths; no
invalid-memory-access probe was run.

### 2. Keep allocation recovery inside an exception-enabled boundary

`Driver.cpp` allocates session state while holding `stateMutex_`, but the shared
driver compiles it with `-fno-exceptions`. `CAPI.cpp` catches `std::bad_alloc`
outside this code. That catch cannot provide the required allocation recovery.
The QDMI session allocation contract requires `QDMI_ERROR_OUTOFMEM`, a null
handle, and a usable driver for a retry.

A standalone C++ caller replaced global `operator new` with a one-shot failure.
After allocating and freeing one session, it armed the failure and called
`QDMI_session_alloc` again. The rebased release library and wheel both aborted
with `std::bad_alloc` (exit 134). The merged #2733 wheel returned `-2`
(`QDMI_ERROR_OUTOFMEM`), left the handle null, and accepted another allocation.

Keep this allocation and ownership path exception-enabled through its catch,
with cleanup intact. Test the exported shared library: the private driver test
target currently uses exception support and misses the production build policy.
Audit the other C ABI catches for the same mismatch when applying the fix.

### 3. Preserve the compiler Python exception category

`submit_program` now raises `RuntimeError` for unsupported output capture where
main raises `ValueError`. The shared binding adapter maps the preserved
`NotSupported` diagnostic to `RuntimeError`; the compiler's former LLVM-error
adapter used `ValueError`. The changed assertion in
`test_qir_output_capture_requires_qir_sampling` accepts this regression.

Both behaviors were reproduced with `custom2=True` for QASM3 with one shot and
QIR Base with zero shots, using the merged #2733 and rebased wheels. Preserve
the compiler binding's category while retaining native QDMI status metadata.
Keep direct `device.submit_job` errors as `RuntimeError`; a global remapping of
`NotSupported` would break that API.

### 4. Remove redundant DD template declarations that break documentation

Doxygen 1.17 rejects all four added `extern template` declarations for
`Package::deserialize` in `include/mqt-core/dd/Package.hpp`: it cannot match
their concrete `FailureOr` return types to the member templates. Consequently,
`uvx nox --non-interactive -s docs` fails before building the Sphinx pages.

Remove these eight lines. Main already keeps the template definitions out of
line without these declarations, and `Serialization.cpp` supplies the explicit
instantiations. A copied-header probe with only these declarations removed
passes the same Doxygen configuration. Native compilation after that deletion
and the rest of the documentation build remain to be checked when applying it.

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

## Acceptance work

All four findings also affect the saved pre-rebase PR head; none comes from
replaying #2733. The rebase retains all 16 commits and reduces the remaining
diff from 26,708 to 25,004 changed lines across 273 files, before this audit
record update. Compiler Target and QDMI adapter conversions still change
diagnostic ownership and belong with the native migration.

- Matched success-path performance measurements, including cold and reused DDSIM
  workers and state transfer, remain outstanding for the final revision.
- Windows/macOS packaging and hosted checks must validate the published head.
  Local Linux results do not establish those platform contracts.
