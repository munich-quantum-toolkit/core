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
Qiskit's `from_circuits()` constructor snapshots headers and serializes the
backend's prepared circuits before admission. Its existing positional
constructor continues to accept submitted handles, which support collection and
cancellation.

The shared `Batch` engine and inspection records live in
`python/mqt/core/plugins/qdmi_batch.py`; native QDMI packaging is unchanged.
Adapters own preparation, result conversion, and tracking. Decoded results
cannot be None, and operations on one batch are synchronous. No locking or
concurrent access support is required. Immutable snapshots retain submission
uncertainty and cached results without copying native handles or result data.

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

Shared lifecycle tests cover retry limits, replacement eligibility, submission
uncertainty, interruption, and cancellation. Adapter tests cover configuration,
result conversion, tracking, recovery handles, and PennyLane postprocessing.
Repeated tests of shared mechanics are removed from the adapter suites.

Final editable, wheel, lint, type, stub, and documentation checks are in
progress. Synthetic failures and local DDSIM only; no paid cloud jobs are
needed.
