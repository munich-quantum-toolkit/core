# Compiler layout metadata

Status: complete.

## Outcome and scope

Target preparation labels input roots in source order. Placement consumes those
labels and writes a complete numeric initial map; routing appends the final
permutation. Metadata lives in the IR, with no externally shared mapper state.
The normal target pipeline serves both the typed API and the CLI checkpoint.

Compilation replaces prior layout metadata. General transformations clear it;
representation conversions preserve it where supported. No public discard API or
export prerequisite is needed. Idle inputs, sparse device site IDs, preplaced
roots, and tensor shrinking retain their input positions. Recording does not
change routing search or its cost model.

The owning files are `mlir/lib/Compiler/TargetCompilation.cpp`,
`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp`, and
`mlir/lib/Dialect/MQT/IR/QubitLayout.cpp`. The metadata schema is documented in
`mlir/include/mqt/Dialect/MQT/IR/MQTDialect.td`; the user contract is in
`docs/mlir/target_compilation.md`.

## Validation

Native compiler, mapping, and MQT IR suites passed with 241, 125, and 38 tests.
They cover placement, routed unitary semantics, recompilation, idle inputs, and
malformed metadata. All three `mqt-core-mqt-cc-` CTest checks passed, including
mapped QCO output. Python MLIR and translation suites passed with 560 tests:

```console
pytest test/python/test_mlir.py test/python/test_mlir_qiskit_translation.py
```

Stub generation, repository lint, whole-file C++ lint, and the complete
executable documentation build passed. Use the nox sessions documented in
`AGENTS.md` to repeat these checks.

## Limits

Tracking requires fixed-size input roots in a single entry block. Programs whose
declared inputs exceed device capacity can still compile after shrinking, but
receive no layout. Mixed static and allocated inputs use adaptive all-to-all
placement. Recompilation treats the current circuit as a new input; it does not
compose prior provenance. Numeric layouts omit source register names and input
ancilla labels; partial layouts remain unsupported.
