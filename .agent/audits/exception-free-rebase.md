# Exception-free API audit

Baseline: PR #2545 on main `9ca32dd3e`, LLVM/MLIR 23.1.0, C++20. The boundary
decisions and four independent audit fixes are retained. Local integration
checks pass; hosted checks, packaging, and performance acceptance remain open.

## Findings addressed

- **DD traversal:** main introduced a throwing width check in `traverseVector`.
  Convert it to `LogicalResult`, retaining the check before callbacks run.
  Preserve ordered, thresholded traversal and direct probability writes. DDSIM
  checks representable result sizes before traversal, so it needs no additional
  failure branch. Existing width and traversal tests cover both outcomes.
- **Upstream integration:** preserve explicit target compilation, consuming
  Qiskit exports with `copy=True` support, GPI synthesis, scoped classical
  storage, and main's compiler and DD optimizations. Adapt fallible callers to
  native results. Qualify MLIR's `mqt` namespace where native diagnostic headers
  also introduce the global namespace.
- **Diagnostic forwarding:** retain the context forwarder alongside main's
  shared pool. MLIR's parallel helpers buffer and replay diagnostics on the
  initiating thread. Extend the existing forwarding test to check ordered worker
  warnings reach its native handler. QCO sampling and DDSIM execution install
  their own worker handlers explicitly.
- **Obsolete code and wording:** remove the duplicate support-library static
  definition and the unreachable JSON parser exception catch. Make the shared
  binding fallback message apply to any operation. Correct the runtime's
  classical-allocation lifetime comment; allocations survive quantum resets.
- **Python diagnostic ownership:** bindings require shared Core libraries so
  extensions and the DD library share one handler stack. CMake rejects static
  Core with bindings; the lint preset and source-build guide use shared mode.
  Standalone native builds still support static libraries.
- **Stream reads:** the SC JSON boundary converts `ios_base::failure` from
  stream-buffer reads to a source-labelled `QDMI_ERROR_FATAL`. The regression
  test covers a throwing buffer alongside bad-stream and missing-file errors.
- **Python compatibility:** gate operand errors, matrix capacity errors, and
  skipped vector levels retain their `RuntimeError` category. Correct the owning
  native diagnostics and restore the Python test expectations.
- **Successful QDMI queries:** custom-property helpers accept synchronous
  `Twine` descriptions. They render descriptions only on failure, retaining
  successful absence and existing status/message semantics.

## Design decisions

| Decision                           | Assessment                                                                                                                                                                                                                                                                          |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FailureOr<T>` and `LogicalResult` | Keep upstream LLVM types and MLIR aliases. Successful results carry no diagnostic. Consume ORC errors at their integration point; the callback returning `llvm::Error` is required by LLVM's API.                                                                                   |
| Fallible factories                 | Keep factories for caller-controlled configuration, capacities, and program setup. Ordinary constructors remain for validated values. Default internal package creation uses a known valid configuration; another checked wrapper would add no recovery.                            |
| Internal preconditions             | Keep balanced DD references and valid measurement targets as native preconditions. Python checks user inputs; QCO mapping and QIR address resolution establish internal targets. Public gate, dimension, numeric, stream, and traversal limits remain checked.                      |
| Scoped native diagnostics          | Keep the small synchronous thread-local stack. A context-owned MLIR diagnostic engine would introduce context ownership into DD and QDMI APIs. Handlers must not throw; re-emission starts at the preceding handler.                                                                |
| MLIR and worker forwarding         | Keep severity, source location, notes, category, and original QDMI status. MLIR parallel replay preserves capture on the initiating thread. Native worker tasks explicitly install capture and publish after completion.                                                            |
| Python and C boundaries            | Keep the single pre-invocation Python adapter and centralized status mapping. Preserve compiler `ValueError` compatibility. Independently embedded QDMI runtimes exchange C statuses, not LLVM error objects or handler state.                                                      |
| JSON                               | Keep normal nlohmann exceptions in private translation units and one DOM parse. Schema/type/range checks precede access; benchmark serialization uses validated values. Parse failures are consumed once by the shared helper. No `JSON_NOEXCEPTION` or STL ABI override is needed. |
| Compiler settings                  | Keep exceptions disabled in native algorithms, enabled in dependency/language boundaries and along driver allocation-recovery paths. MSVC retains `/EHsc` and the normal STL configuration. Disabling unwinding inside a recoverable allocation path would break cleanup.           |
| QIR                                | Keep the `noexcept` facade and result-returning runtime algorithms. Explicit error outputs recover resource-limit failures and clean up partial allocations. ABI validation remains independent of error instrumentation. Runtime calls must not unwind through JIT frames.         |
| DDSIM                              | Keep upstream process isolation, worker reuse, cancellation, and atomic result publication. This PR changes diagnostic transport and result consumption. It does not need a second supervisor or another protocol. Preserve host-side lazy dense/sparse materialization.            |
| Success-path cost                  | Keep main's allocation and copy reductions. Diagnostics are constructed only on failure; QDMI success avoids formatting. No new branch hints or speculative performance changes are justified without matched measurements.                                                         |

The diagnostic conventions follow
[MLIR's diagnostic model](https://mlir.llvm.org/docs/Diagnostics/). The JSON
configuration follows
[nlohmann's exception policy](https://json.nlohmann.me/home/exceptions/).

## Explicit limits

- Host-memory exhaustion is fatal in native algorithms, including QIR resource
  bookkeeping. The
  [QIR allocation contract](https://github.com/qir-alliance/qir-spec/blob/main/specification/Memory_Management.md)
  defines recovery through error outputs; Core provides that recovery for
  invalid requests and runtime resource limits, not host allocator exhaustion.
  The public QIR guide states this limitation. Supporting host exhaustion would
  require allocation recovery throughout the affected runtime and DD stack.
- Direct QIR execution can terminate the process. DDSIM contains payload aborts,
  traps, and runtime termination; it is not a hostile-code sandbox and does not
  isolate the host's own allocations or arbitrary third-party providers.
- Diagnostic handlers do not propagate implicitly across threads or separately
  embedded library runtimes. Consumers must install capture before invocation.
- Exceptions remain at documented dependency, Python, and allocation-recovery
  boundaries. This is result-based native error handling, not an assertion that
  the entire dependency stack is compiled without exceptions.

## Validation and retained coverage

The rebase retains JSON malformed input, `1e400`, duplicate keys, UTF-8,
source/token diagnostics, schemas, and serialization cases. DD coverage retains
numeric limits, invalid configuration, malformed/truncated serialization,
throwing streams, reference ownership, and ordered traversal. QIR tests cover
allocation error outputs, partial cleanup, fatal execution, ABI rejection,
sampling, state extraction, and constructor/destructor lifetime. DDSIM tests
cover concurrent aborting/trapping and valid jobs, later reuse/replacement,
unavailable partial results, cancellation, shutdown, and lazy state results.

See the [execution plan](../plans/exception-free-core.md) for final check
results. Matched successful-call performance and final-head Windows/macOS
packaging remain acceptance work; old-head hosted success is not evidence for
this head.
