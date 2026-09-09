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
overflow qubit. Accept `2 <= n <= 63`. The limit lets the analytic reference use
`uint64_t` modular addition without overflow and bounds the phase-angle table.

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

Precompute the phase angles for every `d_i` and for the modulus. Store the rows
in one rank-one tensor and extract each angle in the target loop. Compute the
row offset from the outer multiplicand-bit loop index. This form keeps the
circuit structured and removes runtime integer arithmetic and conditionals from
the phase additions. The rank-one layout matches the tensor constants supported
by the QCO-to-jeff conversion.

## Reference and tests

Every valid outcome has probability `2^-(n+1)`. For control zero, the
accumulator is zero. For control one, it is the zero-extended value
`multiplier * multiplicand mod modulus`. Parse the bitstrings with
`std::from_chars` and compute this relation with overflow-safe `uint64_t`
double-and-add arithmetic. The 63-bit input limit ensures that adding two
reduced residues fits in `uint64_t`.

Use `multiplier = 011` and `modulus = 101` as the main three-bit case. It covers
both control values, all multiplicand bits, modular wraparound, and values of
the multiplicand greater than or equal to the modulus. Statevector tests must
check the coherent mapping, including relative phases and cleanup of the work
qubit. Sampling tests must check the complete control/multiplicand/accumulator
correlation. A maximum-width test must keep the program structured and all
precomputed angles finite. Add boundary and invalid-input tests. The shared
registry test covers the jeff round trip.

## Work remaining

- [x] Add the typed family, strict JSON contract, binding, stubs, and analytic
      reference.
- [x] Generate and structurally test the exact Figures 5 and 6 circuit.
- [x] Document the source, harness, bit order, validation, and output.
- [x] Validate focused native, MLIR, CLI, and Python behavior.
- [x] Create a draft pull request on the classical-input QFT-adder branch and
      add its number to the rolling structured-benchmark changelog entry.

## Scalable evaluation

The existing `successProbability` field reports the shot-weighted fraction
satisfying the arithmetic relation. Reuse `probability(outcome) > 0`; the
smallest reference weight remains representable at the supported maximum width.
The shared evaluator validates counts and total-shot overflow before relation
counts are accumulated. TVD and fidelity retain their full-distribution meaning.
Sparse samples can have relation success one and TVD near one.

An all-zero output satisfies the relation, so this metric does not establish
uniformity or coherence. Separate balance diagnostics and the existing coherent
state tests remain necessary. Approximate QFT and in-place/order-finding
extensions remain outside this change.
