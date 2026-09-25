# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Recovery contracts shared by the two QDMI adapters."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast
from unittest.mock import Mock

import pytest

from mqt.core.plugins.qdmi_batch import Batch, BatchEntry
from mqt.core.qdmi import Job


class BatchFixture:
    """Three entries with independently controllable provider outcomes."""

    def __init__(self, max_retries: int = 3) -> None:
        """Build a batch with a bounded replacement count and recorded submissions."""
        self.submitted: list[int] = []
        self.jobs: list[Mock] = []
        self.error_at: int | None = None
        self.submission_error: BaseException = RuntimeError("admission failed")
        self.status = Job.Status.DONE
        self.batch: Batch[str] = Batch(
            [BatchEntry(i) for i in range(3)],
            submit=self.submit,
            decode=lambda _i, job: job.get_shots()[0],
            submission_error=RuntimeError,
            execution_error=RuntimeError,
            max_retries=max_retries,
        )

    def submit(self, index: int) -> Job:
        """Return a handle with configurable status and result behavior."""
        self.submitted.append(index)
        if len(self.submitted) == self.error_at:
            raise self.submission_error
        job = Mock()
        job.check.return_value = self.status
        job.wait.return_value = True
        job.get_shots.return_value = [str(index)]
        self.jobs.append(job)
        return cast("Job", job)


@pytest.mark.parametrize("max_retries", [0, 1, 3])
def test_retry_allowance_survives_collection_and_manual_replacements(max_retries: int) -> None:
    """A batch lifetime bounds automatic replacements even after manual recovery."""
    fixture = BatchFixture(max_retries)
    fixture.status = Job.Status.FAILED
    batch = fixture.batch
    batch.submit()
    snapshot = batch.entries
    for _ in range(2):
        with pytest.raises(RuntimeError, match="3 batch entries"):
            batch.complete()
    assert fixture.submitted == [0, 1, 2] * (1 + max_retries)
    assert all(entry.automatic_retries == max_retries for entry in batch.entries)
    assert len(snapshot[0].attempts) == 1
    with pytest.raises(FrozenInstanceError):
        snapshot[0].input_index = 99  # ty: ignore[invalid-assignment] Verify the runtime immutability contract.
    batch.resubmit([1])
    with pytest.raises(RuntimeError):
        batch.complete()
    assert fixture.submitted == [0, 1, 2] * (1 + max_retries) + [1]
    assert batch.entries[1].automatic_retries == max_retries
    for job in fixture.jobs:
        job.cancel.assert_not_called()


def test_collect_first_and_retry_only_failed_entries() -> None:
    """Collect siblings before replacement and reuse successful results."""
    fixture = BatchFixture()
    batch = fixture.batch
    batch.submit()
    fixture.jobs[0].check.return_value = Job.Status.FAILED
    batch.collect()
    assert fixture.submitted == [0, 1, 2]
    assert [entry.result for entry in batch.entries] == [None, "1", "2"]
    batch.complete()
    batch.complete()
    assert fixture.submitted == [0, 1, 2, 0]
    assert [entry.result for entry in batch.entries] == ["0", "1", "2"]
    assert len(batch.entries[0].attempts) == 2
    assert batch.entries[0].attempts[0].status == Job.Status.FAILED
    for job in fixture.jobs[1:]:
        job.get_shots.assert_called_once()


@pytest.mark.parametrize("failure", ["cancelled", "running", "timeout", "status", "read"])
def test_uncertain_or_completed_jobs_are_never_automatically_replaced(failure: str) -> None:
    """Cancellation, uncertain status, timeouts, and read failures require explicit recovery."""
    fixture = BatchFixture()
    batch = fixture.batch
    batch.submit()
    handle = fixture.jobs[0]
    cause = RuntimeError("provider unavailable")
    if failure == "cancelled":
        handle.check.return_value = Job.Status.CANCELED
    elif failure == "running":
        handle.check.return_value = Job.Status.RUNNING
    elif failure == "timeout":
        handle.check.return_value = Job.Status.RUNNING
        handle.wait.return_value = False
    elif failure == "status":
        handle.check.side_effect = cause
    else:
        handle.get_shots.side_effect = cause
    with pytest.raises(RuntimeError) as caught:
        batch.complete()
    assert fixture.submitted == [0, 1, 2]
    assert [entry.result for entry in batch.entries] == [None, "1", "2"]
    if failure in {"read", "status"}:
        assert caught.value.__cause__ is cause
    if failure == "read":
        handle.check.side_effect = cause
        with pytest.raises(ValueError, match="Cannot replace"):
            batch.resubmit([0], allow_unknown=True)
    if failure == "timeout":
        assert isinstance(batch.entries[0].attempts[-1].failures[-1].cause, TimeoutError)
    for job in fixture.jobs:
        job.cancel.assert_not_called()


def test_read_recovery_retains_all_causes_and_successes() -> None:
    """Read failures aggregate without hiding later successes, then recover on the same handles."""
    fixture = BatchFixture()
    batch = fixture.batch
    batch.submit()
    causes = [RuntimeError("first"), RuntimeError("second")]
    for handle, cause in zip(fixture.jobs, causes, strict=False):
        handle.get_shots.side_effect = [cause, ["recovered"]]
    with pytest.raises(RuntimeError, match="2 batch entries") as caught:
        batch.complete()
    assert caught.value.__cause__ is causes[0]
    assert batch.entries[2].result == "2"
    batch.complete()
    assert fixture.submitted == [0, 1, 2]
    assert [entry.result for entry in batch.entries] == ["recovered", "recovered", "2"]
    assert [entry.attempts[0].failures[0].cause for entry in batch.entries[:2]] == causes
    fixture.jobs[2].get_shots.assert_called_once()


@pytest.mark.parametrize("interrupted", [False, True])
def test_partial_submission_preserves_unknown_and_untouched_entries(*, interrupted: bool) -> None:
    """Keep accepted handles separate from uncertain and untouched inputs."""
    fixture = BatchFixture()
    fixture.error_at = 2
    if interrupted:
        fixture.submission_error = KeyboardInterrupt()
    batch = fixture.batch
    with pytest.raises(KeyboardInterrupt if interrupted else RuntimeError):
        batch.submit()
    assert len(fixture.jobs) == 1
    batch.collect()
    assert batch.entries[0].result == "0"
    assert batch.entries[1].attempts[0].handle is None
    assert batch.entries[1].attempts[0].failures[0].cause is fixture.submission_error
    assert not batch.entries[2].attempts
    with pytest.raises(ValueError, match="allow_unknown"):
        batch.resubmit([1, 2])
    assert fixture.submitted == [0, 1]
    batch.submit()
    assert fixture.submitted == [0, 1, 2]
    with pytest.raises(ValueError, match="use resubmit"):
        batch.submit([1])
    batch.resubmit([1], allow_unknown=True)
    batch.complete()
    assert fixture.submitted == [0, 1, 2, 1]
    assert len(batch.entries[1].attempts) == 2
    fixture.jobs[0].cancel.assert_not_called()


def test_failed_replacement_preserves_attempts_and_stops_admission() -> None:
    """A replacement submission failure preserves earlier replacements and stops new ones."""
    fixture = BatchFixture()
    fixture.status = Job.Status.FAILED
    fixture.batch.submit()
    fixture.error_at = 5
    fixture.status = Job.Status.DONE
    with pytest.raises(RuntimeError, match="admission failed"):
        fixture.batch.complete()
    assert fixture.submitted == [0, 1, 2, 0, 1]
    assert [entry.automatic_retries for entry in fixture.batch.entries] == [1, 1, 0]
    fixture.batch.collect()
    assert fixture.batch.entries[0].result == "0"
    assert fixture.batch.entries[1].attempts[-1].handle is None
    for job in fixture.jobs:
        job.cancel.assert_not_called()


def test_interrupt_stops_collection_but_keeps_batch_recoverable() -> None:
    """Interruption leaves collected results and unvisited handles available."""
    fixture = BatchFixture()
    fixture.batch.submit()
    fixture.jobs[1].check.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        fixture.batch.complete()
    assert fixture.batch.entries[0].result == "0"
    fixture.jobs[2].check.assert_not_called()
    fixture.jobs[1].check.side_effect = None
    fixture.batch.complete()
    assert fixture.submitted == [0, 1, 2]
    assert [entry.result for entry in fixture.batch.entries] == ["0", "1", "2"]


def test_explicit_cancel_disables_automatic_replacements() -> None:
    """Cancellation attempts every handle and prevents subsequent automatic replacement."""
    fixture = BatchFixture()
    fixture.status = Job.Status.FAILED
    fixture.batch.submit()
    failure = RuntimeError("cancel failed")
    fixture.jobs[0].cancel.side_effect = failure
    assert fixture.batch.cancel() is False
    for handle in fixture.jobs:
        handle.cancel.assert_called_once()
    with pytest.raises(RuntimeError):
        fixture.batch.complete()
    assert fixture.submitted == [0, 1, 2]
    assert fixture.batch.entries[0].attempts[0].failures[0].cause is failure


@pytest.mark.parametrize(
    ("method", "indices"),
    [("submit", [0, 0]), ("resubmit", [0, -1]), ("submit", [0, 3])],
)
def test_validate_entire_selection_before_submitting(indices: list[int], method: str) -> None:
    """Invalid or duplicate indices must not admit any replacement."""
    fixture = BatchFixture()
    with pytest.raises(ValueError, match="distinct valid"):
        getattr(fixture.batch, method)(indices)
    assert not fixture.submitted


def test_submit_only_admits_untouched_entries() -> None:
    """Initial submission and resumption never replace an existing attempt."""
    fixture = BatchFixture()
    batch = fixture.batch
    batch.submit([1])
    with pytest.raises(ValueError, match="use resubmit"):
        batch.submit([0, 1])
    with pytest.raises(ValueError, match="use submit"):
        batch.resubmit([0], allow_unknown=True)
    assert fixture.submitted == [1]
    batch.submit()
    batch.submit()
    batch.submit([])
    batch.complete()
    assert fixture.submitted == [1, 0, 2]
    assert [entry.result for entry in batch.entries] == ["0", "1", "2"]


def test_other_retries_do_not_repeat_unrelated_result_reads() -> None:
    """A failed task must not turn another task's read error into an automatic read loop."""
    fixture = BatchFixture()
    fixture.batch.submit()
    fixture.jobs[0].check.return_value = Job.Status.FAILED
    fixture.jobs[1].get_shots.side_effect = RuntimeError("read unavailable")
    with pytest.raises(RuntimeError, match="read unavailable"):
        fixture.batch.complete()
    assert fixture.submitted == [0, 1, 2, 0]
    fixture.jobs[1].get_shots.assert_called_once()
    fixture.jobs[2].get_shots.assert_called_once()


def test_manual_replacement_does_not_override_known_running_work() -> None:
    """Even explicit uncertainty permission cannot replace a known active job."""
    fixture = BatchFixture()
    fixture.status = Job.Status.RUNNING
    fixture.batch.submit()
    with pytest.raises(ValueError, match="Cannot replace"):
        fixture.batch.resubmit([0], allow_unknown=True)
    assert fixture.submitted == [0, 1, 2]


def test_cancel_skips_confirmed_terminal_attempts() -> None:
    """Cancel an outstanding replacement without sending requests for finished attempts."""
    fixture = BatchFixture()
    fixture.batch.submit()
    fixture.jobs[0].check.return_value = Job.Status.FAILED
    fixture.batch.collect()
    fixture.batch.resubmit([0])
    assert fixture.batch.cancel() is True
    for handle in fixture.jobs[:3]:
        handle.cancel.assert_not_called()
    fixture.jobs[3].cancel.assert_called_once()
