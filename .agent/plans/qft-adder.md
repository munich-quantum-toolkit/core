# QFT adder benchmark

Status: complete.

## Goal and scope

One `qft-adder` family adds equal-width, big-endian operands. Register mode
returns `addend || sum`; constant mode uses precomputed phases and returns the
sum alone. Wrap mode keeps the operand width; carry mode adds one sum bit.
Leading zeros determine width. Only register addends support `+` qubits;
accumulators and constant addends are binary. Arbitrary amplitudes are excluded.

`src/bench/QFTAdder.cpp` owns input validation and analytic results.
`mlir/bench/programs/QFTAdder.cpp` owns preparation and both circuit methods.
JSON, bindings, and the CLI share the family catalog and typed options.

## Decisions

Keep the register and constant implementations as methods because they perform
related arithmetic with different qubit and gate costs. Use one overflow policy
for both. Retain the shared no-swap QFT helpers and bounded f64 phase tables.
Sum width is limited to 1024 bits to retain representable phases and reference
weights. Group repeated preparation bits into structured loops.

Measure the register addend as well as the sum to expose their correlation. A
unique logical result exists only for basis inputs. Sampling alone cannot check
phase coherence; compare coherent DD statevectors with exact amplitudes.

## Validation

Run the native benchmark and generation binaries and
`uv run --no-sync pytest test/python/bench test/python/test_cli.py -k bench`.
The native generation tests check all 336 operand pairs across widths one
through three, both methods, and both overflow policies, plus 28 coherent
register-addition statevectors. Python checks typed options, JSON round trips,
and generated program bindings. The implementation and its arithmetic and
phase-sensitive execution tests are in #2404. Local validation passed all 24
native generation tests and 17 Python benchmark/CLI tests, plus lint and full
C++ lint. Coverage is bounded to these widths and supported inputs.
