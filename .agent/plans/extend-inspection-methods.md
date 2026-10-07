# Quantum program inspection

Status: implementation complete.

## Scope and decisions

QC and QCO share gate-count traversal and Python bindings. `inspect()` returns
resource information and gate, control-flow, and full operation histograms in
`QuantumProgramInfo`. Individual counting methods compute only the requested
metric. Every MLIR `Program` exposes the full operation histogram.

Gate counts visit the entry-point IR once. Unitary operations, measurements, and
resets count atomically; modifier and call bodies are not expanded. Measurements
and resets have arity one, matching Qiskit's default `size()` semantics.
Barriers appear only in the full operation histogram. Single primitive controls
add a `c` per control, while other single-gate modifiers retain structural names
such as `inv(h)` and `ctrl(inv(x))`. Composite modifiers and wrappers with
unused targets retain their modifier name. Parameters do not split buckets;
calls use callee names.

Control-flow counts visit entry-point branches and region-control operations
once, excluding region terminators. Full operation counts match the scope of
MLIR's `print-op-stats`: the root module, helpers, modifier bodies, terminators,
and nested modules. The upstream pass exposes printed text or JSON rather than
its private count map; the API uses a direct MLIR walk.

Qiskit import omits numeric zero circuit phases in every recursive context and
preserves nonzero and symbolic phases.

Resource inspection reports declared allocated width, distinct physical site
IDs, and control-flow presence. It includes helper functions and excludes nested
modules. Dynamic widths, quantum entry-point inputs, a missing entry point, and
size overflow produce an unknown width. Allocation placement and mutually
exclusive static/dynamic modes are enforced by the dialect verifier. Inspection
does not estimate peak live width or recover source-layout width.

The count tests share expected histograms across QC and QCO while retaining
separate resource-boundary cases. Python checks cover the snapshot, individual
queries, consumed programs, and conversion to Python containers and `None`.

Circuit-depth semantics remain tracked in
[#2682](https://github.com/munich-quantum-toolkit/core/issues/2682).

## Validation

Validation covers compiler inspection, Python bindings, recursive Qiskit phase
import, generated stubs, repository lint, whole-file C++ lint, and executable
documentation with local link checks.
