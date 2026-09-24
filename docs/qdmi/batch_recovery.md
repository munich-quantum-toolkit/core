# Batch retries and recovery

The PennyLane and Qiskit QDMI adapters collect all submitted jobs before
reporting execution failures. Successful results remain cached. A failed
submission stops further submissions without cancelling accepted work.

`max_retries` defaults to **3** automatic replacement executions per batch
entry; set it to `0` to disable them. Only a job confirmed `FAILED` qualifies.
Cancelled jobs, timeouts, uncertain submissions, and result-read errors are not
automatically replaced. Replacements can incur additional charges. Repeated
`result()` calls and manual replacements do not reset the automatic allowance.
Provider SDKs control retries of individual service requests.

## Inspect and recover a batch

An execution or submission exception exposes its batch as `error.job`.
`device.last_job` (PennyLane) or `backend.last_job` (Qiskit) also retains the
latest batch, including after interruption. Keep a reference to retain an older
batch. See the [PennyLane example](pennylane_device.md#recovery) and
[Qiskit example](qdmi_backend.md#recovery).

| Interface               | Behavior                                                                                                                                                                                             |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `job.entries`           | Inspect ordered, immutable entry and attempt records without remote queries. Each attempt retains its handle, last observed status, cached result, and failures with their stage and original cause. |
| `job.collect()`         | Read existing jobs and return entry records, including failures. It never submits new executions.                                                                                                    |
| `job.result()`          | Collect results, perform eligible automatic replacements, and return the complete adapter result or raise an aggregate error.                                                                        |
| `job.resubmit(indices)` | Explicitly submit selected entries, retaining previous attempts. Indices identify entries in `job.entries`.                                                                                          |
| `job.cancel()`          | Disable further automatic replacements and explicitly attempt cancellation. Return whether all requests succeeded without uncertain admissions.                                                      |

Manual replacement accepts failed, cancelled, or untouched entries. Unknown
outcomes require `allow_unknown=True`, which can duplicate a still-running
execution. Known running or completed jobs cannot be replaced. For a result-read
failure, call `collect()` or `result()` again to reuse the existing job.

Recovery works within the same Python process. PennyLane entry indices identify
preprocessed tapes and shot copies through `input_index` and `shot_index`;
recovered results do not reconstruct an interrupted QNode, gradient, or
optimizer. Batch operations are synchronous; do not call recovery methods
concurrently on the same handle.
