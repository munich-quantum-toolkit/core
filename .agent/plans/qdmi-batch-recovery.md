# Recoverable QDMI adapter batches

Status: complete.

## Outcome and scope

Both Python QDMI adapters preserve accepted work, partial results, submission
attempts, and original errors. They collect submitted entries before reporting
execution failures and automatically replace confirmed failed jobs up to
`max_retries` times (default three). The shared implementation lives in
`python/mqt/core/plugins/qdmi_batch.py`; result conversion stays in each
adapter. The user guide is `docs/qdmi/batch_recovery.md`.

Exceptions expose the batch through `job`; adapters retain `last_job` for
interruption recovery. Entry snapshots include input and shot-copy indices,
handles, last observed statuses, cached results, and failure stages and causes.
A submission failure stops new admissions without cancelling accepted work.

## Decisions and limits

Automatic retry counts belong to logical entries and never reset across result
collection or explicit replacement. Only a successful status query reporting
`FAILED` authorizes automatic replacement. Cancellation, uncertain submission,
timeouts, and result-read errors require explicit recovery. Provider SDKs retain
transport retry ownership.

Manual replacement of unknown outcomes requires `allow_unknown=True` because it
can duplicate an execution. Completed or known running jobs cannot be replaced.
Explicit cancellation disables further automatic replacements, attempts all
outstanding known handles, and skips confirmed terminal attempts.

Recovery is in-process and applies to adapter batches, including expanded shot
copies. It does not reconstruct QNodes, gradients, optimizers, or notebook
state. Qiskit batches created by `backend.run()` retain programs for
replacement; direct wrappers of existing handles support collection and
cancellation only. No QDMI ABI, provider, scheduler, or dependency changes are
required.

## Validation

```bash
uv run --no-sync pytest -q \
  test/python/plugins/test_qdmi_batch.py \
  test/python/plugins/qdmi_pennylane test/python/plugins/qiskit
```

306 passed on Python 3.14. Coverage includes retry budgets, partial admission,
interruption, cancellation, result reuse, failure causes, ordering, shot copies,
native primitives, tracking, and DDSIM handle lifetimes.

The full Ponytail review removed redundant wrappers, guards, and unused test
setup, and replaced custom retrieval-error storage with `ExceptionGroup`.

- `uvx nox -s lint`: passed, including repository-wide Ruff and ty checks.
- `uvx nox --non-interactive -s docs`: passed, including executable notebooks,
  generated references, and local HTML links.
- Native DDSIM Qiskit probe: submission, cached results, and cancellation of a
  completed batch passed. No paid cloud jobs were submitted.
