# Add a controlled modular multiplication benchmark

Status: complete.

## Goal and scope

Add the `controlled-multiplication-modulo-n` structured benchmark from
Beauregard's
[Circuit for Shor's algorithm using 2n+3 qubits](https://arxiv.org/abs/quant-ph/0205095),
Figures 5 and 6. Expose the benchmark through the typed C++, JSON, command-line,
Python, and MLIR generation interfaces.

The options are equal-width, big-endian `multiplier` and `modulus` bitstrings.
The modulus must be a canonical nonzero `n`-bit integer greater than one, and
the multiplier must satisfy `0 < multiplier < modulus`. The benchmark prepares
the control and multiplicand registers in the uniform superposition. The
accumulator and one work qubit start in zero. The logical output is
`control || multiplicand || accumulator`; the accumulator includes its leading
overflow qubit. Accept `2 <= n <= 63`; the current QCO-to-jeff pipeline supports
general integer expressions of at most 64 bits, and the modular recurrence needs
`n+1` bits.

## Circuit contract

Apply the exact no-swap QFT to the accumulator. For each little-endian
multiplicand bit `x_i`, compute `d_i = 2^i * multiplier mod modulus` and apply
the Figure 5 modular Fourier-adder block controlled by the main control and
`x_i`:

1. double-controlled `phiADD(d_i)`;
2. inverse `phiADD(modulus)`;
3. inverse QFT;
4. CNOT from the accumulator overflow qubit to the work qubit;
5. QFT;
6. work-qubit-controlled `phiADD(modulus)`;
7. inverse double-controlled `phiADD(d_i)`;
8. inverse QFT;
9. X on the accumulator overflow qubit;
10. CNOT from the overflow qubit to the work qubit;
11. X on the overflow qubit;
12. QFT;
13. double-controlled `phiADD(d_i)`.

After all multiplicand bits, apply the inverse QFT. Do not control the complete
modular block and do not decompose its multi-controlled phase gates. Use P gates
for every Fourier addition.

Represent the current `d_i` as a signless `i(n+1)` MLIR value carried by the
outer loop and use unsigned arithmetic operations. Update it as
`(d_i << 1) urem modulus`; the extra bit prevents an overflow during doubling.
Build each phase angle in a target loop that carries the angle and a
right-shifted copy of `d_i`. Test the low bit with integer operations and use an
`scf.if` to choose whether to add pi. Multiply by -1 for inverse phases. This
form avoids `arith.uitofp`, `arith.negf`, and index-to-wide-integer casts, which
the current QCO-to-jeff pipeline cannot lower. It also keeps the circuit
structured, avoids fixed-width host integers, and stays within the pipeline's
64-bit integer limit.

## Reference and tests

Every valid outcome has probability `2^-(n+1)`. For control zero, the
accumulator is zero. For control one, it is the zero-extended value
`multiplier * multiplicand mod modulus`. Compute this relation with bitstring
double-and-add arithmetic, not native fixed-width integers.

Use `multiplier = 011` and `modulus = 101` as the main three-bit case. It covers
both control values, all multiplicand bits, modular wraparound, and values of
the multiplicand greater than or equal to the modulus. Structural tests must
assert every Figure 5 stage in order, all controls and targets, the absence of
swaps and resets, and cleanup of the work qubit by construction. Execution tests
must sample the full control/multiplicand/accumulator correlation. Add boundary
and invalid-input tests and a jeff round trip.

## Work remaining

- [x] Add the typed family, strict JSON contract, binding, stubs, and analytic
      reference.
- [x] Generate and structurally test the exact Figures 5 and 6 circuit.
- [x] Document the source, harness, bit order, validation, and output.
- [x] Validate focused native, MLIR, CLI, and Python behavior.
- [x] Create a draft pull request on the classical-input QFT-adder branch and
      add its number to the rolling structured-benchmark changelog entry.
