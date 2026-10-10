# Exception-free Core APIs

Status: implementation and audit fixes applied on main `b35d3334a` after #2733.
The [audit](../audits/exception-free-rebase.md) records the boundary fixes,
independent design review, and test reductions. Platform and whole-PR
performance acceptance remain open.

## Work remaining

- [ ] Compare matched successful workloads against upstream, including cold and
  reused DDSIM workers and state transfer.
- [ ] Validate the published revision's Windows/macOS packages and hosted
      checks.

## Contract

All Core libraries require LLVM/MLIR 23.1 or newer. Native fallible APIs use
`llvm::FailureOr<T>` and `llvm::LogicalResult`; MLIR-facing APIs use the same
types through MLIR aliases. Infallible APIs return ordinary values or `void`.
Successful absence remains `optional<T>`, and borrowed results use pointers.
Scoped thread-local diagnostics preserve severity, category, message, and QDMI
status. Binding invocation and test capture each share one implementation. See
[native error handling](../../docs/cpp_api.md#handle-native-errors).

Standalone driver and device builds embed diagnostic support. QDMI C interfaces
carry status codes and use local logging. Private driver APIs and their static
tests use scoped diagnostics without separate error-output parameters. Keep
C-status conversion at each runtime boundary. Do not pass `llvm::Error` between
independently embedded, hidden LLVM support archives: their error class
identities differ, and consuming such an error can abort. The driver keeps
exception unwinding enabled through its C ABI allocation-recovery boundary.

Preserve C++20, Python APIs, QDMI statuses, and result semantics. Private JSON
translation units catch dependency exceptions and parse valid input once,
retaining schema, duplicate-key, encoding, and numerical checks. DD kernels
remain exception-free; MSVC retains normal STL configuration and exception ABI.

Balanced DD reference ownership and valid measurement targets are native
preconditions. Python validates them at its binding boundary. QCO qubit mappings
and QIR address resolution establish valid targets internally. Keep fallibility
for public input, provider, numerical, compiler, and I/O errors. Typed gate
construction and operations on already-validated owned state need no repeated
checks. Benchmark evaluation validates outcomes once through its probability
callback; hashing uses LLVM SHA-256.

The JIT validates C calling conventions, native ABI attributes, and runtime call
signatures before binding host functions. ORC errors use scoped diagnostics. QIR
resource-allocation failures with explicit error outputs remain recoverable and
release partial allocations. Host allocation failures and other unhandled
runtime failures diagnose and terminate. The thin `noexcept` ABI facade retains
exception support; algorithms do not. Setup and compilation errors remain
recoverable. Session-owned classical allocations survive quantum resets until
reference-count release or session destruction. JIT constructors and destructors
bind the session runtime, as entry-point execution does.

DDSIM worker isolation and concurrency are supplied by the parent multi-program
change. This layer adapts worker compilation and compact DD reconstruction to
explicit results and carries structured diagnostics over the existing framed
transport. Crashes fail only their assigned programs without replay; completed
siblings retain their results. Cancellation, worker reuse, indexed results,
stable IDs, and standard-interface driver replacement keep their parent
contracts.

## Platform constraints

LLVM 23's Windows socket stream releases its Winsock guard before the base
stream closes. Both endpoints close explicitly and clear handled errors through
one shared deleter. LLVM's crash-dialog suppression keeps fatal jobs unattended.
Core normalizes MSVC exception flags at the common target boundary. POSIX socket
sends suppress `SIGPIPE` per call: Darwin delivers that signal to the process,
so a thread-local signal mask cannot protect the host. A write may succeed
locally after the peer has disconnected.

Completed workers receive an empty protocol frame and a bounded wait so normal
cleanup can write coverage counters. Active or unresponsive workers terminate.
Poll only the assigned child: LLVM 23's POSIX timed wait changes the host
SIGALRM handler and may reap another child on timeout.

## Validation

Validation on Linux aarch64 with GCC 13.3 and LLVM/MLIR 23.1.0:

- `cmake --preset release`, the release build with LTO, and
  `ctest --preset release` pass: 4,043 runnable tests and one expected
  `ScQDMIJobSpecificationTest.QueryJobId` skip.
- The rebuilt wheel passes all 1,976 Python tests with
  `uv run --no-sync pytest`.
- `uvx nox -s stubs` passes with no generated API changes.
- `uvx nox --non-interactive -s docs` passes Doxygen, Sphinx, executable
  examples, and internal links.
- Repository lint passes. Whole-file C++ lint checked all 179 selected source
  files against main. Its trailing-comma finding is fixed; the affected file
  passes the complete-file follow-up, and all 127 DD package tests pass again.

Earlier installed GCC Development/default consumers built and ran, missing-SDK
lookups behaved as specified, and wheel libraries exported no LLVM-owned
symbols. Those installed-consumer and symbol checks were not repeated for these
fixes.

Windows/macOS packaging and hosted checks for the published revision remain
unverified. No new packaging test framework is introduced here.

Performance acceptance requires matched upstream and revised builds using the
external harness. Report successful workloads separately, including cold and
reused DDSIM workers and state transfer. Earlier measurements do not establish
performance of the current implementation.

## JSON evaluation measurement

A matched microbenchmark compares the benchmark code at `1c131aade` with the
single-parse implementation. Both link the Release benchmark and support
archives with GCC 13.3 at `-O3`, on Linux aarch64 with LLVM/MLIR 23.1.0. It
calls `evaluateJSON` with each GHZ manifest and a valid 100-shot all-zero counts
histogram. Each of three alternating runs has one warmup batch and nine measured
batches of 5,000 successful calls, pinned to CPU 0. The table reports the median
of each run's median in microseconds per call.

| Workload               | Before | After |
| ---------------------- | -----: | ----: |
| GHZ 2 qubits, Z basis  |  41.63 | 24.76 |
| GHZ 64 qubits, Z basis |  42.53 | 25.52 |
| GHZ 20 qubits, X basis |  42.15 | 25.12 |

The single-parse change reduced these medians by about 40%. Baseline run medians
varied by up to 3%; one batch had a larger scheduling outlier. The shared host
and these small JSON workloads limit the result: it measures this refactor, not
whole-PR or DDSIM latency.
