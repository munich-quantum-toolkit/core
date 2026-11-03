# Native Qiskit circuit construction

Status: complete.

## Scope and decisions

Use the shared Qiskit 2.5/2.6 C API for loose-bit circuit construction, register
creation, and empty control-flow blocks in
`bindings/mlir/Target/Qiskit/QiskitAdapter.h`. Transfer native ownership to
Python before constructing the wrapper, retaining generated circuit names. Each
writer owns its unpublished circuit; block copies retain bit and register
identities, drop variables, and receive their own phase during emission.
Validate register names before entering native constructors that abort on
invalid UTF-8. General package and QDMI support remains at Qiskit 2.1 or newer.

Qiskit translation lives under `bindings/mlir/Target/Qiskit`, matching the
external-format organization introduced in #2769. Both minor adapters and their
shared implementation move together. CMake, candidate generation, adoption
tooling, and wheel-test staging use the same location. The move does not change
the Python API or introduce IQM output functionality.

Read public conditions through Python: both supported native versions retain
insertion-time conditions after a public `operation.condition` edit. Build the
existing C++ equality tree directly for tuples instead of constructing and
normalizing Python expressions. Preserve wide values, nested capture maps, and
impossible-comparison folding. Native expression access still lacks variable
identities and arbitrary-width values.

## Validation

The full Python suite passes with Qiskit 2.6.0 (1,985 tests). The five
`qiskit-compat` compiler suites pass with Qiskit 2.5.2 (702 tests). The Qiskit
plugin and QDMI suites pass with Qiskit 2.1.0 (627 tests, one expected
MLIR-translation skip). The existing register-rejection test also checks invalid
UTF-8; no new test function was needed. Independent Ponytail review found no
remaining issues. Stub generation leaves no tracked changes.

For 30 register conditions on either minor, Python profiling shows import
removes 60 `equal` and 120 `lift` calls and reduces `find_bit` calls from 240 to
120. Export no longer calls Python `add_bits` or `copy_empty_like`.

Qiskit 2.6.0 timing diagnostics compare Release builds against `5689c4c75` on
Python 3.15, using CPU affinity without reserving the CPU. ABBA runs use two
warmups and seven measured samples per process. Exporting 10,000 leading loose
qubits changes from about 8.9 to 5.2 ms. Importing 500 register-tuple conditions
changes from 16.1 to 6.1 ms. Exporting those conditions changes from 12.6 to
13.2 ms because empty blocks get fresh Python wrappers. The small export cost
does not justify separate wrapper-construction paths.
