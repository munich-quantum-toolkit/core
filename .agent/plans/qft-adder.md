# QFT adder benchmark

Status: in progress; validate the unified API and exhaustive small-width tests.

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

## Work remaining

- [ ] Validate options, references, JSON round trips, bindings, and QC/jeff
      generation.
- [ ] Check every operand pair at widths one through three for both methods and
      overflow policies.
- [ ] Check coherent addition amplitudes, then run required lint and stub
      generation.

## Validation

Run the native benchmark and generation binaries and
`uv run --no-sync pytest test/python/test_bench.py test/python/test_cli.py -k bench`.
Local implementation checks passed: 47 native reference/JSON tests, 17
generation tests, and 26 Python benchmark/CLI tests. Stubs were regenerated. The
implementation belongs to #2404; #2408 adds exhaustive arithmetic and
phase-sensitive execution checks. Both retain the existing PR chain.
