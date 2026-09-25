# Recoverable QDMI adapter batches

Status: complete.

## Behavior and ownership

Both adapters preserve accepted work, cached successes, submission attempts, and
original errors. Automatic replacement is opt-in (`max_retries=0` by default);
only confirmed `FAILED` jobs qualify. Retry allowances belong to logical entries
and do not reset across collection or manual replacement.

`submit(indices=None)` admits untouched entries, defaulting to all remaining
untouched entries. `resubmit(indices, allow_unknown=False)` replaces previous
attempts. Both use one internal submission path. Unknown acceptance requires
explicit permission to duplicate work; known running or completed jobs cannot be
replaced. Submission failures stop further admission. Cancellation is explicit
and disables automatic replacement.

Exceptions expose `job`; adapters retain `last_job` before first submission.
Qiskit's documented `programs` input holds prepared payload/format pairs, while
its existing positional constructor continues to accept submitted handles.

The shared Python implementation and inspection records live in
`python/mqt/core/qdmi/batch.py`. Native QDMI is installed as the package
initializer, retaining public type names and native submodules. A narrow binding
fallback restores package paths omitted by scikit-build-core 1.0.3's editable
loader. It can be removed when native initializers are recognized.

## Recovery and native multi-program boundary

The adapter guides contain opt-in and recovery examples. PennyLane's public
`construct_batch(..., level="device")` retains the postprocessor needed to
recover forward measurement values, including broadcasts and shot vectors.
Recovery does not resume differentiation, optimizers, or arbitrary QNode return
containers, and remains within the same process.

Logical result ordering is separate from provider job identity. Native
multi-program selection remains in the follow-ups to QDMI #509 and Core #2373,
coordinated in Core #2359. Those consumers must map logical outputs to native
jobs and result indices, group compatible format/shot/options settings, and
retain independent submissions when native batching is unsupported. An ambiguous
submission must never trigger fallback. Under the proposed aggregate contract,
recovery follows the whole native job's lifecycle and failure state; per-program
failure cannot be inferred. No grouping machinery, C++ recovery API, scheduler,
or transport retries are introduced here.

## Validation

```bash
uv run --no-sync pytest -q test/python/qdmi \
  test/python/plugins/qdmi_pennylane test/python/plugins/qiskit
```

- Editable install: 621 tests passed on Python 3.14.
- Fresh wheel in an isolated environment: the same 621 tests passed, plus an
  explicit package/submodule/type-name and native DDSIM execution probe.
- Stub regeneration completed without generated API changes.
- Repository lint and type checks passed.
- Full executable documentation and local HTML link checks passed.
- The changed binding passed full-file C++ analysis and formatting checks.
- Synthetic failures and local DDSIM only; no paid cloud jobs were submitted.
