# Target-aware Pauli twirling

Status: local implementation and semantic validation complete. Native tests,
Python tests, generated stubs and full changed-file C++ lint pass. External
linkcheck reports two unchanged QIR specification anchors; hosted checks and
human review remain pending.

## Contract and ownership

`CompilationOptions.enablePauliTwirling` defaults to false. The shared target
pipeline applies the existing Pauli-twirling pass after mapping and native
synthesis, then lowers one-qubit corrections and verifies target conformance.
The Python property is `enable_pauli_twirling`; the compiler flag is
`--pauli-twirling`. Both use the shared compilation seed. Synthesis-only and
gate-set targets honor the option. Missing targets and bypassed target pipelines
reject it. Each compilation selects one realization.

`mlir/lib/Dialect/QCO/Transforms/NativeSynthesis/TargetSynthesis.cpp` owns the
restricted single-qubit stage and shares the full synthesis pass's emitter. It
tracks static sites, preserves native entangler shells, respects modifier
boundaries and uses the actual target basis. Re-running full native synthesis is
unsafe because it can fuse gates across the correction boundary. No general
pipeline-hook framework, new twirling table, new gate family, ensemble API or
execution-shot randomization is required.

Target and payload phase policy remains in force. Exact full-unitary checks use
phase-capable targets. In-place failures can leave partial IR changes; consumed
program APIs retain their documented ownership rules.

## Evidence and limits

Tests in the native-synthesis and compiler unit-test trees compare the final
entangler sequence, ordered sites, full unitary and local correction segments.
`mlir/unittests/TwirlingTest.h` shares the checkpoint oracle; negative probes
prove it detects corrections moved through an entangler. Forced-routing tests
prove an eligible gate was introduced before twirling. Python and CLI tests
exercise options, defaults, seeds and rejected contexts.

Normal QC, OpenQASM 3 and serialized jeff round trips retain the selected
realization. jeff's allocated wires are related to compact static fixture sites
by allocation order. Native OpenQASM 2 declarations are opaque to the importer;
its complete output is compared with direct native-checkpoint export instead.
QIR tests compare ordered QIS calls, constant parameters and physical operands
with direct conversion; IQM tests compare the complete serialized instruction
stream. These checks found no need to change cleanup.

Run the native release build and CTest preset, Python compiler tests,
`nox -s stubs`, `nox -s cpp-lint -- upstream/main`, `nox -s lint`, and the
executable docs and linkcheck sessions described in `AGENTS.md`. Inspect
collected tests and report blocked or skipped gates separately from passes.
