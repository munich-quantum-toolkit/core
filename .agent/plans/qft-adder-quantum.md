# Register-input QFT adder benchmark

Status: complete; configurable inputs and family consolidation remain proposals.

## Goal and scope

Expose `qft-adder-quantum` through C++, Python, JSON, the CLI, and MLIR.
`src/bench/QFTAdderQuantum.cpp` owns its analytic reference;
`mlir/bench/programs/QFTAdderQuantum.cpp` emits Draper's controlled-phase adder.
The addend starts in `|+>^n`, the accumulator in `|1>`, and the result is the
big-endian concatenation `addend || sum`, with `sum = addend + 1 mod 2^n`.

## Decisions

Measure both registers: the sum alone is uniform and cannot check addition. The
shared no-swap QFT helpers in `mlir/bench/programs/QFTUtils.*` also serve QFT
and QPE. Keep controlled phase gates; controlled RZ changes relative phases.
Halve angles from the largest rotation to avoid premature underflow. The
1024-qubit register limit keeps the reference probability representable.

The DD interpreter reads dense rank-one f64 phase tables with checked indices;
QPE and adder tests can sample structured programs without loop unrolling. The
name distinguishes a register-held addend from a classical constant, not quantum
computation from classical computation. Configurable operands and a shared
family need an agreed overflow contract before changing the public API.

## Validation

Run `mqt-core-bench-test` and `mqt-core-mlir-unittests-benchmark` from their
build directories, and `uv run --no-sync pytest test/python/test_bench.py`.
Prior local checks passed for reference/JSON behavior, QC/jeff generation, phase
structure, and DD sampling of the width-three correlated distribution. Sampling
this fixed input does not certify arbitrary accumulators or relative phases;
those require broader inputs and statevector or functionality checks.
