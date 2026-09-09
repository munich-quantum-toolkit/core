# Export independently scheduled measurements

Status: in progress. Validate measurement-store normalization on the export
clone.

## Goal and scope

Export valid mapped QC programs whose measurement stores are separated by
independent measurements or control flow. This change is stacked on the routing
fix in PR [#2351](https://github.com/munich-quantum-toolkit/core/pull/2351). It
adds no scheduling constraint to the mapper. The implementation belongs in
`bindings/mlir/qiskit/QiskitExport.cpp`, with semantic regressions in
`test/python/test_mlir_qiskit_translation.py`.

## Decisions

- Keep the measurement at its original quantum position and fuse its unique,
  static destination only when intervening operations cannot access that bit.
  CBit registers are non-aliasing; distinct static indices are disjoint.
- Inspect nested effects, while retaining the verified QC unitary contract for
  operations with intentionally conservative quantum memory effects.
- Move each supported store immediately after its measurement on the exporter's
  existing clone before indexing writes. Materialize a late constant destination
  index before the measurement. Quantum operations retain their order, and the
  caller's program stays unchanged.
- Snapshot analysis uses the actual normalized order. Do not maintain synthetic
  writes at measurement positions or model other measurements' future stores.
  Retain the parent's indexed lookup and scalar snapshot support.
- Do not add scratch classical bits: Qiskit exposes them in the public result.
  Conflicting destination accesses, unknown effects, and unsupported stale
  snapshots remain diagnosed rather than silently changing results.
- Snapshot checks remain conservative at register granularity. Reuse the
  parent's scalar snapshot support, including reads consumed across a fused
  write. Stale register snapshots wider than 64 bits remain unsupported.
- Benchpress's temporary textual event-order guard cannot prove equivalence and
  rejects legal independent scheduling. Retire it only with the tested Core
  snapshot and deterministic semantic regressions; retain input-profile
  restrictions and validate native export and target compliance separately.

## Validation

Current validation on parent `aa1b13cf8`, with rebuilt Python 3.13 bindings and
Qiskit 2.5.2:

- `pytest test/python/test_mlir_qiskit_translation.py test/python/test_mlir_loops.py`:
  all 398 tests pass. Added cases cover direct measured-bit control before a
  delayed store, disjoint-bit snapshots without redundant scalar variables, late
  destination indices, and ordered writes to a shared destination. Reverse
  writes and conflicting effects remain rejected. Export leaves the input
  program unchanged.
- The nested Qiskit control program that aborted before the parent's fixes now
  compiles through the target pipeline.
- `uvx nox -s stubs`: passes without generated API changes.
- `uvx nox -s cpp-lint -- aa1b13cf8`: no whole-file C++ lint findings.
- `uvx nox -s lint`: passes.

Historical validation at source revision `7dad9e19e`, with Qiskit 2.5.0:

- `pytest test/python/test_mlir_qiskit_translation.py`: 330 tests, including
  deterministic QC/QCO native-export round trips and unsafe-fusion rejections.
  The preceding build fails 12 of the new positive regressions.
- All 31 guarded Benchpress feed-forward profiles, ten previously enabled
  profiles, and BV100: 42 native-export checks, with 4,621 conditionals and
  recursive Qiskit basis/connectivity validation. No OpenQASM fallback is used;
  BV100 retains exactly 99 measurements and classical bits.
- All 80 Benchpress integration tests, including deterministic output checks and
  an explicit rejection regression for unsupported snapshot capture.
- `uvx nox -s stubs`, `uvx nox -s lint`, and `uvx nox -s cpp-lint -- 8936bc2ab`:
  no whole-file C++ lint findings. Stub generation changes no public API files.

The historical Benchpress update pins that snapshot and retains input
restrictions. Its old snapshot-rejection result predates the parent's scalar
snapshot support and has not been revalidated. Those integration checks and the
full benchmark suite have not been rerun for this update; structural condition
counts are not a general semantic equivalence proof.
