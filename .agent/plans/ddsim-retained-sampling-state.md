# Retain simulator states after sampling

Status: complete.

## Contract

DDSIM sampling jobs expose statevector and probability results when the existing
QCO or QIR terminal-sampling path retained an uncollapsed state. The job owns
the DD package until destruction; vector materialization stays lazy.
Repeated-shot execution does not expose a trajectory as a statevector. Queries
do not execute the program or draw random numbers. Zero-shot extraction remains
supported, including eligible Adaptive programs outside the sampling fast path.
Zero-qubit states retain their scalar amplitude. Sampling eligibility is
unchanged.

## Validation and delivery

Local native suites passed: 72 DDSIM tests, 50 QIR JIT tests, and 192 QCO
utility tests. Coverage includes text/bitcode, unavailable states, repeated
queries, phase, wire order, lifetime, seeded samples, and zero-qubit states.
Python QDMI regressions also pass in the combined 330-test focused suite.
Repository lint and whole-file clang-tidy checks passed for the changed C++
sources.

The independent change is exported as
`build/patches/01-ddsim-retained-state.patch`. The compiler convenience change
can be applied after it.
