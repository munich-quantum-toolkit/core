# Export independently scheduled measurements

Status: complete.

## Goal and scope

Export valid mapped QC programs whose measurement stores are separated by
independent measurements or control flow. This is an exporter change stacked on

## 2351's routing fix, not an additional scheduling constraint on the mapper. The

implementation belongs in `bindings/mlir/qiskit/QiskitExport.cpp`, with semantic
regressions in `test/python/test_mlir_qiskit_translation.py`.

### Decisions

- Keep the measurement at its original quantum position and fuse its unique,
  static destination only when intervening operations cannot access that bit.
  CBit registers are non-aliasing; distinct static indices are disjoint.
- Inspect nested effects, while retaining the verified QC unitary contract for
  operations with intentionally conservative quantum memory effects.
- Account for the exporter's earlier measurement writes when validating lazy
  classical expressions. Memory effects alone do not preserve an SSA read
  captured before a measurement and evaluated by a later conditional.
- Do not add scratch classical bits: Qiskit exposes them in the public result.
  Overlapping destinations, unknown effects, and unsupported stale snapshots
  remain diagnosed rather than silently changing results.
- Snapshot checks remain conservative at register granularity. A whole-register
  read consumed across an early fused write still needs scalar materialization,
  even if only another bit changes. This is outside the benchmark profiles fixed
  here; retain a specific diagnostic rather than weakening the check.
- Benchpress's temporary textual event-order guard cannot prove equivalence and
  rejects legal independent scheduling. Retire it only with the tested Core
  snapshot and deterministic semantic regressions; retain input-profile
  restrictions and validate native export and target compliance separately.

### Validation

At source revision `7dad9e19e`, a fresh wheel with Qiskit 2.5.0 passes:

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

The Benchpress update pins this PR snapshot and retains input restrictions. Its
additional whole-register snapshot probe is valid mapped IR but remains
unsupported by native export, as it was before this change. The full benchmark
suite has not been restarted; structural condition counts are not a general
semantic equivalence proof.
