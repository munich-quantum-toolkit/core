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
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock, PropertyMock

import pytest

from mqt.core.plugins.qdmi_batch import Batch, BatchEntry
from mqt.core.qdmi import Job

if TYPE_CHECKING:
    from collections.abc import Sequence


class BatchFixture:
    """Three entries with independently controllable provider outcomes."""

    def __init__(self, max_retries: int = 3, *, native: bool = False) -> None:
        """Build a batch with a bounded replacement count and recorded submissions."""
        self.submitted: list[int] = []
        self.native_calls: list[list[int]] = []
        self.native = native
        self.jobs: list[Mock] = []
        self.error_at: int | None = None
        self.submission_error: BaseException = RuntimeError("admission failed")
        self.status = Job.Status.DONE
        self.batch: Batch[str] = Batch(
            [BatchEntry(i) for i in range(3)],
            submit=self.submit,
            submit_programs=self.submit_programs,
            group_by=[0, 0, 0] if native else None,
            decode=lambda _i, job, program_index: job.get_shots(program_index)[0],
            submission_error=RuntimeError,
            execution_error=RuntimeError,
            max_retries=max_retries,
        )

    def submit(self, index: int) -> Job:
        """Return a handle with configurable status and result behavior."""
        self.submitted.append(index)
        if len(self.submitted) == self.error_at:
            raise self.submission_error
        return self._job([index])

    def submit_programs(self, indices: Sequence[int]) -> Job | None:
        """Return a recorded native attempt, or reject it before submission."""
        self.native_calls.append(list(indices))
        if not self.native:
            return None
        self.submitted.extend(indices)
        if self.error_at is not None:
            raise self.submission_error
        return self._job(indices)

    def _job(self, indices: Sequence[int]) -> Job:
        job = Mock()
        job.program_statuses = None
        job.check.return_value = self.status
        job.wait.return_value = True
        job.get_shots.side_effect = lambda program_index=0: [str(indices[program_index])]
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
    with pytest.raises(ValueError, match="use submit"):
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


@pytest.mark.parametrize("unsupported", [False, True])
def test_native_group_or_single_fallback_preserves_attempts(*, unsupported: bool) -> None:
    """A rejected setter has no attempt or budget; accepted lists share one handle."""
    fixture = BatchFixture(native=True)
    fixture.native = not unsupported
    fixture.batch.submit()
    fixture.batch.complete()
    assert fixture.native_calls == [[0, 1, 2]]
    assert fixture.submitted == [0, 1, 2]
    assert [entry.result for entry in fixture.batch.entries] == ["0", "1", "2"]
    assert all(len(entry.attempts) == 1 and entry.automatic_retries == 0 for entry in fixture.batch.entries)
    assert [entry.attempts[0].program_index for entry in fixture.batch.entries] == (
        [0, 0, 0] if unsupported else [0, 1, 2]
    )
    assert len(fixture.jobs) == (3 if unsupported else 1)
    for job in fixture.jobs:
        job.wait.assert_called_once()
        job.check.assert_called_once()


def test_native_partial_failure_retries_only_failed_programs() -> None:
    """A failed aggregate retains successful siblings and remaps replacement indices."""
    fixture = BatchFixture(native=True)
    fixture.batch.submit()
    shared = fixture.jobs[0]
    shared.check.return_value = Job.Status.FAILED
    outcomes = PropertyMock(return_value=[Job.Status.FAILED, Job.Status.DONE, Job.Status.FAILED])
    type(shared).program_statuses = outcomes
    fixture.batch.complete()
    assert fixture.native_calls == [[0, 1, 2], [0, 2]]
    assert fixture.submitted == [0, 1, 2, 0, 2]
    assert [entry.result for entry in fixture.batch.entries] == ["0", "1", "2"]
    assert [entry.attempts[-1].program_index for entry in fixture.batch.entries] == [0, 1, 1]
    assert [entry.automatic_retries for entry in fixture.batch.entries] == [1, 0, 1]
    shared.get_shots.assert_called_once_with(1)
    shared.check.assert_called_once()
    shared.wait.assert_called_once()
    outcomes.assert_called_once()
    assert fixture.batch.statuses() == (Job.Status.DONE,) * 3
    shared.check.assert_called_once()


@pytest.mark.parametrize("stage", ["outcomes", "result"])
def test_native_read_failure_never_replaces_execution(stage: str) -> None:
    """A failed status artifact or result download is retried on its original handle."""
    fixture = BatchFixture(native=True)
    fixture.batch.submit()
    shared = fixture.jobs[0]
    cause = RuntimeError("download unavailable")
    if stage == "outcomes":
        shared.check.return_value = Job.Status.FAILED
        type(shared).program_statuses = PropertyMock(side_effect=[cause, [Job.Status.DONE] * 3])
    else:
        shared.get_shots.side_effect = [cause, ["1"], ["2"], ["0"]]
    with pytest.raises(RuntimeError, match="download unavailable"):
        fixture.batch.complete()
    assert fixture.native_calls == [[0, 1, 2]]
    assert fixture.submitted == [0, 1, 2]
    if stage == "result":
        assert [entry.result for entry in fixture.batch.entries] == [None, "1", "2"]
    fixture.batch.complete()
    assert [entry.result for entry in fixture.batch.entries] == ["0", "1", "2"]
    assert fixture.native_calls == [[0, 1, 2]]


def test_uncertain_native_admission_has_no_single_fallback() -> None:
    """A native submission error marks all selected programs uncertain without replay."""
    fixture = BatchFixture(native=True)
    fixture.error_at = 1
    with pytest.raises(RuntimeError, match="admission failed"):
        fixture.batch.submit()
    assert not fixture.jobs
    assert fixture.submitted == [0, 1, 2]
    assert all(entry.attempts[0].handle is None for entry in fixture.batch.entries)
    with pytest.raises(ValueError, match="allow_unknown"):
        fixture.batch.resubmit([0, 1])
    assert not fixture.batch.cancel()
    fixture.error_at = None
    fixture.batch.resubmit([0, 1], allow_unknown=True)
    assert fixture.native_calls == [[0, 1, 2], [0, 1]]


def test_shared_cancel_and_individual_status_before_manual_replacement() -> None:
    """Cancellation visits a shared handle once; a failed sibling cannot replace a success."""
    fixture = BatchFixture(native=True)
    fixture.batch.submit()
    shared = fixture.jobs[0]
    cause = RuntimeError("cancellation unavailable")
    shared.cancel.side_effect = cause
    assert not fixture.batch.cancel()
    shared.cancel.assert_called_once()
    assert all(entry.attempts[0].failures[-1].cause is cause for entry in fixture.batch.entries)
    shared.check.return_value = Job.Status.FAILED
    shared.program_statuses = [Job.Status.DONE, Job.Status.CANCELED, Job.Status.FAILED]
    with pytest.raises(ValueError, match="Cannot replace"):
        fixture.batch.resubmit([0, 2])
    fixture.batch.resubmit([1, 2])
    assert fixture.native_calls == [[0, 1, 2], [1, 2]]
    assert shared.check.call_count == 2


def test_running_native_job_does_not_require_final_outcomes() -> None:
    """Unavailable final artifacts cannot turn confirmed active work into uncertainty."""
    fixture = BatchFixture(native=True)
    fixture.batch.submit()
    shared = fixture.jobs[0]
    shared.check.return_value = Job.Status.RUNNING
    outcomes = PropertyMock(side_effect=RuntimeError("outcomes not ready"))
    type(shared).program_statuses = outcomes
    assert fixture.batch.statuses() == (Job.Status.RUNNING,) * 3
    with pytest.raises(ValueError, match="Cannot replace"):
        fixture.batch.resubmit([0, 1], allow_unknown=True)
    assert fixture.submitted == [0, 1, 2]
    outcomes.assert_not_called()
