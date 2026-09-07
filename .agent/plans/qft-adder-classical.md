# Constant-input QFT adder benchmark

Status: complete; configurable inputs and family consolidation remain proposals.

## Goal and scope

Expose `qft-adder-classical` through C++, Python, JSON, the CLI, and MLIR.
`src/bench/QFTAdderClassical.cpp` owns the reference;
`mlir/bench/programs/QFTAdderClassical.cpp` emits the Fourier constant adder. A
nonempty big-endian addend defines the input width `n`, including leading zeros.
The `n+1`-qubit accumulator starts in `|1>`; its measured result is the addend
plus one, including carry.

## Decisions

Reuse the no-swap QFT helpers. A known classical addend needs one precomputed
phase per accumulator wire, avoiding an extra quantum register and its
controlled gates. Scan bits from least to most significant, halving the previous
angle and adding pi for a set bit; append the halved angle for the carry wire.
This avoids fixed-width integer conversion. At most 1023 input bits keep the
accumulator within the shared 1024-qubit QFT limit.

Dense f64 phase tables run directly in the DD interpreter. No unrolling is
needed. The fixed accumulator is a benchmark input, not a restriction of the
addition algorithm. Consolidation with the register-input family requires an
agreed overflow contract and configurable input semantics.

## Validation

Run `mqt-core-bench-test` and `mqt-core-mlir-unittests-benchmark` from their
build directories, and `uv run --no-sync pytest test/python/test_bench.py`.
Prior local checks passed for reference/JSON behavior, QC/jeff generation, phase
tables, and direct DD sampling of zero, leading-zero, and carry cases. These
tests cover the fixed accumulator; arbitrary input states remain outside the
benchmark contract.
