# Exception-free Core APIs

Status: rebased on main `715da0ee6` after #2732. The
[current reassessment](../audits/exception-free-rebase.md) defines the compiler
Program API extraction and the remaining acceptance work. Python exception
compatibility still needs the identified submission-category fix.

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
identities differ, and consuming such an error can abort.

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
recoverable.

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

- Release build with LTO passes. CTest passes 4,045 runnable tests, with the
  expected `ScQDMIJobSpecificationTest.QueryJobId` skip.
- The rebuilt Python package passes all 1,974 tests. The final wheel passes all
  433 QDMI tests after the XDG fallback fix.
- Stub generation passes with no generated API changes.
- Repository lint passes. Whole-file C++ lint covered all 183 changed source
  files. Both unused-include findings are fixed; complete-file checks and all
  450 affected QDMI native tests pass after the cleanup.
- Installed GCC Development and default consumers build and run. Runtime and
  optional Development lookups work without the LLVM SDK; required Development
  and default lookups report the missing SDK. The wheel's native shared
  libraries, including CoreSupport, export no LLVM-owned definitions.

Documentation execution was not repeated for this rebase.

Windows/macOS packaging and hosted checks for the published revision remain
unverified. No new packaging test framework is introduced here.

Performance acceptance requires matched upstream and revised builds using the
external harness. Report successful workloads separately, including cold and
reused DDSIM workers and state transfer. Earlier measurements do not establish
performance of the current implementation.
