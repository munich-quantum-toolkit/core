# Quantum program inspection

Status: complete; QC and QCO share consolidated inspection results.

## Scope and decisions

QC and QCO share gate-count traversal and Python bindings. `inspect()` returns
resource information and all gate counts in `QuantumProgramInfo`. Individual
counting methods compute only the requested metric.

Gate counts visit the entry-point IR once. Each unitary operation counts
atomically; modifier and call bodies are not expanded. Barriers, measurements,
and resets are excluded. Histograms use operation base symbols, with `ctrl`,
`inv`, and `pow` for modifiers and callee names for unitary calls.

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

All 272 compiler tests and the six Python tests selected by
`pytest test/python/test_mlir.py -k program_inspection` passed. The overflow
regression reproduced an abort before the checked shape query was applied. Stubs
were regenerated; repository lint, whole-file C++ lint, and executable
documentation with local link checks passed.
