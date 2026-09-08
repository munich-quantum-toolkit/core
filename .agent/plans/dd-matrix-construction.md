# Native DD construction

Status: complete.

## Outcome and ownership

Native matrix and vector construction serve C++ containers and NumPy views.
`Package` owns matrix construction and physical target embedding; state
factories in `StateGeneration.hpp` own vector construction and retain the
returned roots. QCO retains gate matrix extraction and wire mapping, then
delegates construction. No new dependency or dense expansion to the surrounding
state width is needed.

## Decisions

- Share recursive construction through compile-time entry accessors. NumPy uses
  const strided views, preserving offsets, negative strides, and broadcasts
  without copying complex input storage.
- Keep specialized one-, two-, and three-target matrix constructors. General
  matrix embedding sorts bounded stack storage in DD level order while
  preserving the matrix's most-significant-bit operand order.
- Validate vector dimensions and qubit counts before reading entries. Retain
  nonconstant scalar vector roots across garbage collection, as for larger
  states.
- Check state intervals with subtraction so offsets cannot overflow validation.
  Zero-state creation validates capacity before construction.
- Reject conflicting polarities on one control qubit before constructing DD
  nodes. Such input previously produced repeated levels and could abort export.
- Compare the at most three small-gate targets directly. Their validator needs
  no temporary sorted vector or heap allocation.

## Validation

Local checks passed: 173 release DD tests, 3,983 configured assertion-enabled
CTest cases with one expected skip, and 57 Python DD/QCO tests. Full-file C++
lint, general lint, stub regeneration, and the strict documentation build
passed.

Focused entry points are `mqt-core-dd-test`, `mqt-core-mlir-unittest-qco-utils`,
and `pytest test/python/dd test/python/test_qco_dd.py`. The root agent guide
documents build presets and required lint sessions. New regressions cover
dimensions, capacity, scalar ownership, complex entries, target order, control
polarity, state offsets, and NumPy view layouts.

## Performance evidence

Release construction probes use warmed medians and alternate the compared paths.
These are construction measurements, not end-to-end compiler or CI speedups.

The matrix comparison against the merged baseline measured general embedding at
4.73 vs 5.09 microseconds for four targets, 82.9 vs 89.4 microseconds for six,
and 4.74 vs 4.93 milliseconds for eight. Three process pairs each reported seven
warmed samples. Matrices used sin(row * dimension + column) as the real part and
cos(row + 2 * column) as the imaginary part, on targets [k, ..., 1]. Native
dense construction at two to eight qubits stayed within about 3% of baseline.

The vector comparison ran the old and new constructors in one process, checked
that their canonical roots matched, and balanced their reference counts. Three
processes each alternated seven warmed samples on vectors with 2, 8, 64, 1,024,
and 65,536 entries. Amplitudes used sin(index) and cos(2 * index). Median
construction time fell by 8–16%, including 33.7 to 28.4 milliseconds at 65,536
entries.

A warmed 1,000-call probe counted 1,000 allocations before and zero after for
each small-gate constructor, with and without controls. Two- and three-target
medians improved by about 1–5%. Single-target timings varied substantially
between processes, so no single-target latency improvement is claimed.

## Limits

Sparse controls remain limited to one to three matrix targets. The adjacent
audit covered state factories, input bindings, root ownership, gate
construction, and traversal consumers; it does not establish correctness of
unrelated DD algorithms.
