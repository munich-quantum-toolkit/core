# Synthesis validation and IBM models

Status: complete locally; hosted CI is checked separately.

## Outcome and contracts

The failing CRX/sqrt-iSWAP test retains its original inputs and now uses the
existing Weyl reconstruction tolerance. Primitive matrix tolerance does not
account for the numerical decomposition's accumulated matrix products.

The new IBM names exposed QDMI test discovery's space-only sanitization. A
shared sanitizer replaces all non-alphanumeric characters and appends the
parameter index, keeping names valid and unique for all three device suites.

Fixed R coverage shares the RX/RY quarter-turn suite. Removed a duplicate
fixed-R integration test and a fixed-RX cost test whose two-qubit cost behavior
is already checked independently of the single-qubit basis. Runtime tests cover
each source operation and native entangler without their redundant Cartesian
product. Phase, reversed placement, native counts, fixed half turns, symbolic
binding, bounds, and rejection checks remain.

Bench provides Nighthawk's 120-qubit 12-by-10 grid and CZ basis. Core's SC
catalogue adds Heron 156 and Nighthawk 120. Both topologies match IBM Runtime's
Fez and Nighthawk snapshots. Calibration is synthetic in Bench and absent in
Core. Fractional Heron remains available through Qiskit targets: QDMI v1 cannot
advertise the bounded RZZ parameter. No private metadata protocol was added.

## Validation and audit

Local checks passed 880 C++ tests across decomposition, native synthesis,
optimization, and compiler suites; 321 QDMI C++ tests; 1,304 Core Python tests
across MLIR and QDMI; and 553 Bench tests. Minimum-dependency Bench validation
passed 360 tests with 14 optional-feature skips. Both documentation builds and
repository lint pass. Full-file C++ lint passed without diagnostics.

The ponytail audit covered Euler/Pauli/Weyl synthesis, target capability
selection, native cost and emission, single-qubit fusion, symbolic composition,
and global-phase normalization. No further material complexity cuts were
identified. Keep balanced symbolic accumulation, phase-aware emission, bounded
angle fallback, and control-flow frame boundaries: each protects a supported
contract. Broader pipeline optimizations remain a separate follow-up.
