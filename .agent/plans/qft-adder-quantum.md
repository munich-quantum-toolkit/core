# Add a quantum-input QFT adder benchmark

Status: in progress. The implementation is validated. The draft pull request and
its changelog reference remain to be added.

## Goal and scope

Add the `qft-adder-quantum` structured benchmark from Draper's
[Addition on a Quantum Computer](https://arxiv.org/abs/quant-ph/0008033). The
benchmark must be available through the typed C++, JSON, command-line, Python,
and MLIR generation interfaces. It must generate the full no-swap QFT, Draper
addition, and inverse-QFT circuit rather than a circuit with the same output
distribution.

The benchmark parameter is the width `n` of each quantum register. The source
register is prepared as |+>^n and the accumulator as |1>. The one logical
`result` output has width `2n` and is written as the big-endian concatenation
`addend || sum`. Its ideal distribution has probability `2^-n` exactly when
`sum = addend + 1 mod 2^n`. Measuring both registers keeps this correlation
observable; measuring the sum alone would produce an uninformative uniform
distribution.

## Decisions

Register index zero is the least-significant bit. The forward QFT uses no swaps
and visits targets from most to least significant. For target `t`, it applies H
and then `CP(pi / 2^(t-c))` from every lower control `c`. The addition block
applies the same controlled-phase gate from source control `c <= t` to
accumulator target `t`, including each `CP(pi)` gate. The inverse QFT reverses
the complete gate order and negates each phase. `CP` cannot be replaced with a
controlled RZ because their relative phases differ.

The width is limited to 1024 qubits per register. This keeps the smallest
required binary phase and the ideal probability representable as `double`. The
implementation does not add swaps, carry qubits, approximate rotations, or an
alternative QFT convention. A private MLIR helper may own the shared forward and
inverse no-swap transforms; it must not change the existing QFT benchmark.

## Work remaining

- [ ] Create the draft stacked pull request and fold its number into the
      existing unreleased structured-benchmark changelog entry.

## Validation

The release build, all 50 native benchmark tests, all 15 MLIR benchmark tests,
the benchmark CLI test, and 23 focused Python benchmark and CLI tests pass. The
Python test samples the width-three circuit and compares the result with the
analytic correlation. Stub generation, the general repository lint session, and
`git diff --check` pass. The separate C++ lint session was not run.
