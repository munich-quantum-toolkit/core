# Add a repeat-until-success benchmark

Status: complete.

## Goal and scope

Add a fixed `repeat-until-success` structured benchmark from Paetznick and
Svore's
[Repeat-Until-Success: Non-deterministic decomposition of single-qubit unitaries](https://arxiv.org/abs/1311.1074v2),
Figure 8. Expose the benchmark through the typed C++, JSON, command-line,
Python, and MLIR generation interfaces. The family has no options.

The benchmark uses one ancilla and one data qubit, both initially in `|0>`. Each
attempt applies this exact sequence:

1. H on the ancilla;
2. T on the ancilla;
3. CNOT from the ancilla to the data qubit;
4. H on the ancilla;
5. CNOT from the ancilla to the data qubit;
6. T on the ancilla;
7. H on the ancilla;
8. measure the ancilla in the computational basis.

Place the attempt in the before region of a post-test `scf.while`. Use the
measurement result directly as the condition, so one repeats and zero exits.
Place one X on the ancilla in the after region. The after region only runs after
a failed attempt and restores the measured `|1>` ancilla to `|0>`. Do not apply
data recovery, use `qc.reset`, add an `scf.if`, or cap the number of attempts.

## Reference and tests

For the paper's T-gate convention, outcome zero applies
`U = (I + i*sqrt(2)*X)/sqrt(3)` with probability `3/4`. Outcome one applies the
identity up to global phase with probability `1/4`. After eventual success,
apply S-dagger and H to the data qubit and measure `result`. This
phase-sensitive readout has probabilities `P(0) = 1/2 + sqrt(2)/3` and
`P(1) = 1/2 - sqrt(2)/3`; it distinguishes U from its adjoint.

Structural tests must assert the one `scf.while`, exact operation order and wire
roles in its before region, direct measurement condition, sole failure-path X,
and S-dagger/H/data measurement after the loop. They must reject hidden resets,
conditionals, recovery gates, and retry bounds. Add native reference and JSON
tests, a seeded QCO sampling test, shared registry and CLI checks, and
QC-to-jeff serialization with byte round-trip.

Version 1 of the paper labels the success unitary incorrectly. Version 2 fixes
the label, matches the stated T convention, and is the version selected by the
tracker's unversioned arXiv link.

## Work remaining

- [x] Add the fixed typed family, JSON contract, binding, stubs, and analytic
      reference.
- [x] Generate and structurally test the exact Figure 8 loop and readout.
- [x] Document the source, loop semantics, and output distribution.
- [x] Validate focused native, MLIR, CLI, and Python behavior.
- [x] Create a draft pull request on the controlled modular multiplication
      branch and add its number to the rolling structured-benchmark changelog
      entry.
