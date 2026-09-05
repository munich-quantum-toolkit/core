# Add a classical-input QFT adder benchmark

Status: implementation complete. The stacked draft pull request and changelog
reference remain to be completed.

## Goal and scope

Add the `qft-adder-classical` structured benchmark from Beauregard's
[Circuit for Shor's algorithm using 2n+3 qubits](https://arxiv.org/abs/quant-ph/0205095),
Figure 3. Expose the benchmark through the typed C++, JSON, command-line,
Python, and MLIR generation interfaces.

The instance parameter is a nonempty big-endian classical addend. Its length
defines an `n`-bit value. The benchmark prepares an `n+1`-qubit accumulator as
`|1>`, applies Beauregard's exact classical Fourier addition, and measures one
big-endian `result` output. The exact reference is the zero-extended addend plus
one; the extra accumulator qubit preserves overflow.

## Decisions

The generator uses the shared exact no-swap QFT and inverse-QFT helpers in
`mlir/bench/programs/QFTAdderUtils.*`. Between them, it emits exactly one
unconditional phase gate for each accumulator wire, including a zero-angle gate.
For little-endian accumulator wire `j`, the phase is the binary fraction formed
by addend bits `j` through zero. The extra wire receives the continued fraction
and therefore records carry.

Compute the phase table by scanning the addend from least to most significant:
divide the previous angle by two and add pi for a set bit. Append one more
halved angle for the overflow wire. This produces canonical angles in
`[0, 2*pi)` without converting an arbitrary-width addend to a fixed-width
integer. The input length is limited to 1023 so the accumulator and QFT remain
within 1024 qubits.

The fixed `|1>` accumulator is the benchmark harness, not part of the source's
general adder definition. Do not add swaps, approximate rotations, a carry
ancilla, or controlled phases: Beauregard's classical-input optimization
combines each wire's classically known rotations into one single-qubit phase.

## Work completed

- [x] Add the typed family, strict JSON contract, binding, stubs, and reference
      tests.
- [x] Generate and structurally test the complete Figure 3 circuit.
- [x] Document the source, harness, bit order, phase convention, and output.
- [x] Validate the focused native, MLIR, and Python behavior.
- [ ] Create the draft pull request on the quantum-input QFT-adder branch and
      add its number to the rolling structured-benchmark changelog entry.
