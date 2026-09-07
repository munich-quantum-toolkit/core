# Native DD matrix construction

Status: complete.

## Goal and scope

The DD package owns matrix construction for nested C++ matrices, strided NumPy
arrays, and QCO local matrices on ordered physical targets. QCO retains gate
matrix extraction and wire mapping. Preserve scalar matrices, implicit identity
levels, operand order, complex weights, and existing sparse controls for one to
three targets. Do not expand a local matrix to the surrounding state dimension.

## Decisions

- Keep the specialized gate constructors and dispatch to them for one to three
  targets. Move general embedding and its dimension checks into `dd::Package`.
- Share the four-quadrant recursion through compile-time element and level
  accessors. Nested vectors and NumPy strides must require neither a matrix copy
  nor an indirect call per entry.
- Validate dimensions and package capacity before recursion. Reject duplicate or
  out-of-range targets and unsupported sparse controls before constructing
  nodes.

## Validation

Local checks passed: 170 release DD tests, 185 QCO utility tests, 3,980
configured assertion-enabled CTest cases (one expected skip), and 54 Python
DD/QCO tests. Stub regeneration, general lint, and full-file C++ lint on the
committed diff passed. The final style corrections were followed by a fresh
native DD test run.

The focused entry points are `mqt-core-dd-test`,
`mqt-core-mlir-unittest-qco-utils`, and
`pytest test/python/dd test/python/test_qco_dd.py`. Root agent guidance
documents build presets and required lint sessions.

A release microbenchmark compared the merged baseline with the shared recursion
on dense complex matrices of one, two, three, four, six, and eight qubits. Three
alternating process pairs, each reporting the median of seven warmed samples,
measured general embedding at 4.73 vs 5.09 microseconds (four targets), 82.9 vs
89.4 microseconds (six), and 4.74 vs 4.93 milliseconds (eight). Native dense
construction at two to eight qubits and two/three-target gates stayed within
about 3%. Single-target measurements varied between 66 and 140 nanoseconds for
both binaries; no speedup is claimed there. These are construction
microbenchmarks, not end-to-end compiler or CI speedups. Matrices used sin(row *
dimension + column) as the real part and cos(row + 2 * column) as the imaginary
part, on targets [k, ..., 1].

## Outcome and limits

`Package` owns one matrix recursion, dimension validation, and target embedding.
The QCO adapter delegates construction, and the NumPy binding supplies a const
strided view. Generated stubs describe read-only input support. New tests cover
complex matrices, target order, controls, scalar and empty matrices, shape and
capacity failures, negative strides, transposes, offsets, and broadcasts. Sparse
controls remain limited to one to three targets. No dense expansion to the
surrounding state width or new dependency is required.
