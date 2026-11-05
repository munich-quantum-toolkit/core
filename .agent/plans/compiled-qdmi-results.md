# QDMI compilation and serialization

- Static native programs use terminal measurements in ascending site order.
- A nonidentity ResultLayout reconstructs the original classical destinations.
- Dynamic programs retain their classical outputs.
- Bodyless native calls act on explicit operands and precede terminal readout.
- Calls to defined functions decline normalization; measured-site reuse fails.
- Source-only submission refuses to discard a nonidentity layout.
- Compile explicitly and retain the layout when decoding such jobs.
- OpenQASM uses implicit outputs unless non-output globals require qualifiers.
- Programs are never wrapped in a synthetic conditional scope.
- Qiskit register indices preserve physical site IDs, including gaps.
- Each backend retains one immutable target environment.
- Core serializes native IQM MOVE gates with ordered qubit/resonator loci.
- Native serialization does not route or synthesize MOVE.
- Provider adapters own transport conventions and result retrieval.
- Qiskit tests use a device-snapshot fixture for Python boundary doubles.
- Format order: QPY, OpenQASM 3, Adaptive QIR (binary, text), IQM JSON, Base QIR
  (binary, text), OpenQASM 2.
