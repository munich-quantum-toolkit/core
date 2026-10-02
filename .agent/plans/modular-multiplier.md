# Modular multiplier benchmark

Status: complete.

## Contract and decisions

Use `ModularMultiplier`, `modular_multiplier`, and `modular-multiplier`
consistently across the unreleased C++, Python, JSON, CLI, tests, and docs. No
compatibility alias is needed for the unreleased name.

Retain the out-of-place Beauregard Figures 5/6 circuit and exact QFT schedule:
`|c>|x>|0>|0> -> |c>|x>|c*a*x mod N>|0>`. Width remains 2..63 bits. The constant
multiplier is nonzero and less than the canonical modulus. Noncoprime values and
x >= N remain valid because the accumulator starts at zero.

Require a same-width multiplicand string of `0`, `1`, and `+`; control accepts
those same states and defaults to `1`. Reuse the QFT adder's register
preparation in the existing QFT utility. Basis inputs expose one
`expected_result`, computed with independent classical modular arithmetic.
Superposed inputs retain an analytic reference, weighted by the number of plus
bits. Evaluation success requires matching both configured input bits and the
product relation.

Basis-state validation avoids exponential sampling requirements. A small number
of plus bits also keeps the reference support manageable at large width. Neither
approach alone detects arbitrary relative phases. Retain the full coherent-state
oracle, including clean work qubits, and avoid an inverse echo that can cancel
matching circuit errors.

## Validation

Cover all 312 basis cases for two- and three-bit canonical moduli, nonzero
multipliers below the modulus, both control values, and every multiplicand.
Compare generated samples with independent integer multiplication. Retain the
coherent oracle at widths 2..5 and add partial-superposition/control cases.
Check malformed inputs, maximum-width references, JSON/schema/identity, Python
options and expected results, compact generation, and jeff/CLI behavior. Run
regenerated stubs, general lint, and full-file C++ lint before publication.

## Follow-ups

Approximate QFT with an explicit error budget, in-place multiplication with
coprimality/domain constraints, and iterative order finding remain separate
features. Gearbox/PAR, distillation, and cultivation remain separate benchmark
families.

Local validation passes: 57 native tests, 27 MLIR tests (including 312 basis
cases), 52 Python benchmark/CLI checks, general lint, and full-file C++ lint.
Python bindings and stubs were rebuilt. The CLI generation regression passes.
