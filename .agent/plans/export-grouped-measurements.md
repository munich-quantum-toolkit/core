# Export independently scheduled measurements

Status: in progress; implementation and focused validation remain.

## Goal and scope

Export valid mapped QC programs whose measurement stores are separated by
independent measurements or control flow. This is an exporter change stacked
on #2351's routing fix, not an additional scheduling constraint on the mapper.
The implementation belongs in `bindings/mlir/qiskit/QiskitExport.cpp`, with
semantic regressions in `test/python/test_mlir_qiskit_translation.py`.

## Decisions

- Keep the measurement at its original quantum position and fuse its unique,
  static destination only when intervening operations cannot access that bit.
  CBit registers are non-aliasing; distinct static indices are disjoint.
- Inspect nested effects, while retaining the verified QC unitary contract
  for operations with intentionally conservative quantum memory effects.
- Account for the exporter's earlier measurement writes when validating lazy
  classical expressions. Memory effects alone do not preserve an SSA read
  captured before a measurement and evaluated by a later conditional.
- Do not add scratch classical bits: Qiskit exposes them in the public result.
  Overlapping destinations, unknown effects, and unsupported stale snapshots
  remain diagnosed rather than silently changing results.
- Benchpress's temporary textual event-order guard cannot prove equivalence
  and rejects legal independent scheduling. Retire it only with the tested
  Core snapshot and deterministic semantic regressions; retain input-profile
  restrictions and validate native export and target compliance separately.

## Validation

Run the Qiskit translation tests, required stub generation, repository lint,
and whole-changed-file C++ lint. Rebuild the Python wheel and test all 31 guarded
Benchpress feed-forward profiles plus BV100 through native export, without
restarting the full benchmark suite. These checks are not yet complete.
