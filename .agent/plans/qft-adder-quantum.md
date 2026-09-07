# Add a quantum-input QFT adder benchmark

Status: complete.

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
accumulator target `t`, including each `CP(pi)` gate. The inverse QFT visits
targets from least to most significant. It starts each target at `-pi / 2` and
halves the angle while visiting lower controls from nearest to farthest. This
order gives the exact inverse because the controlled-phase gates commute. It
also prevents distant rotations from making later nearby rotations underflow.
`CP` cannot be replaced with a controlled RZ because their relative phases
differ.

The width is limited to 1024 qubits per register. This keeps the smallest
required binary phase and the ideal probability representable as `double`. The
implementation does not add swaps, carry qubits, approximate rotations, or an
alternative QFT convention. Private MLIR helpers own the shared phase loop and
the forward and inverse no-swap transforms. The standard QFT and QPE generators
use the same transforms.

## Validation

The focused MLIR test checks the controlled-addition register and phase
relations. The shared benchmark test checks QC and jeff generation. The Python
test samples the width-three circuit and compares the result with the analytic
correlation. The largest supported instance stays structured and uses finite
angles.
