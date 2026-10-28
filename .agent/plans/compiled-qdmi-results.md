# Native QDMI payloads and logical results

Status: shared Core, provider, and PennyLane integration implemented and
published. Provider draft pins and final hosted CI validation remain.

## Goal and scope

Keep compilation explicit. A Qiskit circuit is compiled for a backend before
submission. Native serialization produces a receiver-compatible payload and,
when needed, a small mapping from raw samples to logical classical outputs. The
shared job adapter restores output widths, order, repeated values, and
constants. Quantum operations retain their compiled physical placement.

The compiler owns terminal readout normalization in `mlir/lib/Compiler/`.
Provider libraries retain service column/key decoding. The Qiskit adapter in
`python/mqt/core/plugins/qiskit/` retains each map through batching and
recovery. PennyLane consumes this interface in its separate compiler
integration.

## Decisions

- Receiver profiles express verified transport constraints. Braket simulators
  use virtual wires; native hardware preserves physical placement. IQM uses its
  measurement instruction/locus order. IBM uses accepted classical declarations
  and self-contained gates.
- A result layout contains a raw width and one raw slot or constant per logical
  bit. Identity results need no map. Both raw validation and logical output
  validation remain mandatory.
- Terminal normalization supports permutations, gaps, repeated measurements, and
  final overwrites. It rejects unsupported dynamic readout before submission.
- Ordinary circuit payloads omit explicit `output` declarations. General exports
  retain explicit selection when removing it would change results.
- Direct native submissions retain provider format semantics. Recovering
  frontend results requires the original result map and framework header.
- No new QDMI ABI, typed-output container, automatic compilation layer, or
  provider-side source interpreter is needed.

## Work remaining

- [x] Native serialization and shared checked result mapping.
- [x] Qiskit serialization, batching, result headers, and projection
      aggregation.
- [x] Provider integration and PennyLane consumption in their owning PRs:
      Braket #288, IQM #321, IBM #94 (on #92 → #71 → #72), and Core #2767.
- [ ] Refresh provider draft pins after final Core changes.
- [ ] Validate the final published heads in hosted CI.
- [x] Focused native/Python tests, repository lint, and rendered documentation.

## Validation

Use asymmetric deterministic circuits with permuted destinations, unused bits,
repeated measurement, overwritten destinations, and nonalphabetical registers.
Compare complete samples as well as counts. Exercise routing and reject
unsupported readout before any submission. Validate actual provider payload
consumers; local simulator support alone does not establish cloud compatibility.

Native compiler, OpenQASM emission/frontend, and Qiskit translation suites pass.
Repository lint, generated stubs, whole-file native lint, and executable
documentation pass. Live integration passed SV1, Rigetti, and AQT (16 shots
each), plus an IQM mock (8 shots independently reconstructed from raw service
measurement columns).

The live Rigetti target exposed a disabled physical site. The Qiskit adapter now
preserves sparse physical IDs rather than inventing an unavailable site. Local
circuit names and deep-copied metadata survive explicit compilation.

Operation CUSTOM2 supplies the temporary versioned `mqt.operation-parameters.v1`
contract. Its alternatives reuse fixed parameters and inclusive bounds already
in the compiler target. Qiskit and direct QDMI compilation share the parser;
providers publish constraints without introducing provider-name heuristics.
