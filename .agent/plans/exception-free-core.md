# Exception-free Core APIs

Status: restacked and locally validated. Hosted platform checks remain pending.

## Contract

All Core libraries require LLVM/MLIR 23.1 or newer. Native fallible APIs use
upstream `FailureOr<T>` and `LogicalResult`; infallible APIs return ordinary
values or `void`. Successful absence remains `optional<T>`, and borrowed results
use pointers. Scoped thread-local diagnostics preserve severity, category,
message, and QDMI status. Binding invocation and test capture each share one
implementation. See
[native error handling](../../docs/cpp_api.md#handle-native-errors).

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

QIR allocation failures with explicit error outputs remain recoverable and
release partial allocations. Other runtime failures diagnose and terminate. The
thin `noexcept` ABI facade retains exception support; algorithms do not. Setup
and compilation errors remain recoverable.

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

The restack passes all 3,666 release cases with one expected SC query skip,
1,545 Python tests, generated stubs, executable documentation and its link
checks. The parent change provides installed dual-driver consumer, relocation,
controlled concurrency, and throughput evidence. The complete GCC 13 suite uses
`ENABLE_IPO=OFF`: linking the support-test target with GCC LTO and the local
prebuilt LLVM 23 archives reports duplicate MLIR TypeID symbols. The same target
builds with Clang 23 and default IPO, and its 23 tests pass. No source
workaround is applied. Windows, macOS, and packaging results remain subject to
hosted CI.

Performance acceptance requires matched upstream and revised builds using the
external harness. Report successful workloads separately, including cold and
reused DDSIM workers and state transfer. Earlier measurements do not establish
performance of the current implementation.
