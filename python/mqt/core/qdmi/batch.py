# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Retained attempts and recovery of QDMI submissions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import TYPE_CHECKING, Generic, TypeVar

from mqt.core.qdmi import Job

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["BatchEntry", "JobAttempt", "JobFailure"]

_Result = TypeVar("_Result")
_TERMINAL = {Job.Status.DONE, Job.Status.FAILED, Job.Status.CANCELED}


@dataclass(frozen=True)
class JobFailure:
    """An original exception and the operation that raised it."""

    stage: str
    cause: BaseException


@dataclass(frozen=True)
class JobAttempt(Generic[_Result]):
    """A submission attempt, including an uncertain submission without a handle."""

    handle: Job | None = None
    status: Job.Status | None = None
    result: _Result | None = None
    failures: tuple[JobFailure, ...] = ()


@dataclass(frozen=True)
class BatchEntry(Generic[_Result]):
    """An input's ordered attempts; indices refer to the adapter's input batch."""

    input_index: int
    shot_index: int = 0
    attempts: tuple[JobAttempt[_Result], ...] = ()
    automatic_retries: int = 0

    @property
    def result(self) -> _Result | None:
        """Cached result of the latest attempt, or None when unavailable."""
        return self.attempts[-1].result if self.attempts else None


def _validate_max_retries(value: object) -> int:
    """Validate the number of automatic replacement executions.

    Returns:
        The validated retry count.

    Raises:
        ValueError: If the value is not a nonnegative integer.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        msg = f"max_retries must be a nonnegative integer, got {value!r}."
        raise ValueError(msg)
    return int(value)


class _Batch(Generic[_Result]):
    """Own attempts and reuse the same submission and collection paths for both adapters."""

    def __init__(
        self,
        entries: Sequence[BatchEntry[_Result]],
        *,
        submit: Callable[[int], Job] | None,
        decode: Callable[[int, Job], _Result],
        submission_error: Callable[[str], Exception],
        execution_error: Callable[[str], Exception],
        max_retries: int,
        on_submit: Callable[[int], None] | None = None,
    ) -> None:
        self._entries = list(entries)
        self._submit = submit
        self._decode = decode
        self._submission_error = submission_error
        self._execution_error = execution_error
        self._max_retries = _validate_max_retries(max_retries)
        self._cancelled = False
        self._on_submit = on_submit

    @property
    def entries(self) -> tuple[BatchEntry[_Result], ...]:
        """Immutable snapshots without querying the provider."""
        return tuple(self._entries)

    def _set_attempt(self, index: int, attempt: JobAttempt[_Result], position: int = -1) -> None:
        entry = self._entries[index]
        attempts = list(entry.attempts)
        attempts[position] = attempt
        self._entries[index] = replace(entry, attempts=tuple(attempts))

    def record_failure(self, index: int, stage: str, cause: BaseException, position: int = -1) -> None:
        """Retain a failure without discarding the handle or a decoded result."""
        attempt = self._entries[index].attempts[position]
        self._set_attempt(index, replace(attempt, failures=(*attempt.failures, JobFailure(stage, cause))), position)

    def _validate_indices(self, indices: Sequence[int]) -> tuple[int, ...]:
        selected = tuple(indices)
        if any(
            isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(self._entries)
            for index in selected
        ) or len(set(selected)) != len(selected):
            msg = "Indices must be distinct valid batch entry indices."
            raise ValueError(msg)
        return selected

    def submit(self, indices: Sequence[int] | None = None) -> None:
        """Submit untouched entries, or all remaining untouched entries by default.

        Raises:
            ValueError: If a selected entry has already been attempted.
        """
        selected = (
            tuple(i for i, entry in enumerate(self._entries) if not entry.attempts)
            if indices is None
            else self._validate_indices(indices)
        )
        if any(self._entries[index].attempts for index in selected):
            msg = "Selected entries have already been attempted; use resubmit() to replace them."
            raise ValueError(msg)
        self._submit_entries(selected)

    def _submit_entries(self, indices: Sequence[int], *, automatic: bool = False) -> None:
        """Submit selected entries, retaining earlier work if admission fails.

        The adapter's submission error preserves the original cause.

        Raises:
            RuntimeError: If the batch was constructed without submission data.
        """
        if not indices:
            return
        if self._submit is None:
            msg = "Submission requires prepared programs."
            raise RuntimeError(msg)
        for index in indices:
            entry = self._entries[index]
            self._entries[index] = replace(
                entry,
                attempts=(*entry.attempts, JobAttempt()),
                automatic_retries=entry.automatic_retries + int(automatic),
            )
            stage = "submission"
            try:
                handle = self._submit(index)
                self._set_attempt(index, JobAttempt(handle=handle))
                stage = "tracking"
                if self._on_submit is not None:
                    self._on_submit(index)
            except BaseException as exc:
                self.record_failure(index, stage, exc)
                if not isinstance(exc, Exception):
                    raise
                msg = f"Failed to submit batch entry {index}: {exc}"
                raise self._submission_error(msg) from exc

    def collect(self, indices: Sequence[int] | None = None) -> tuple[BatchEntry[_Result], ...]:
        """Collect existing attempts without replacements or aggregate failure.

        Returns:
            Ordered entry snapshots, including every recorded failure.
        """  # ruff:ignore[docstring-missing-exception] Per-entry errors are recorded, not propagated.
        for index in range(len(self._entries)) if indices is None else indices:
            entry = self._entries[index]
            if not entry.attempts or entry.result is not None:
                continue
            attempt = entry.attempts[-1]
            if attempt.handle is None or self._terminal_failure(attempt):
                continue
            handle = attempt.handle
            stage = "status"
            try:  # ruff:ignore[too-many-statements-in-try-clause] Each stage records its own failure context.
                status = handle.check()
                self._set_attempt(index, replace(attempt, status=status))
                if status not in _TERMINAL:
                    stage = "wait"
                    if not handle.wait():
                        msg = "Timed out waiting for the QDMI job."
                        raise TimeoutError(msg)  # ruff:ignore[raise-within-try] Record this entry and continue collecting.
                    stage = "status"
                    status = handle.check()
                    self._set_attempt(index, replace(attempt, status=status))
                stage = "execution"
                if status != Job.Status.DONE:
                    msg = f"QDMI job did not complete successfully: {status.name}."
                    raise RuntimeError(msg)  # ruff:ignore[raise-within-try] Record this entry and continue collecting.
                stage = "result"
                result = self._decode(index, handle)
                current = self._entries[index].attempts[-1]
                self._set_attempt(index, replace(current, result=result))
            except BaseException as exc:
                self.record_failure(index, stage, exc)
                if not isinstance(exc, Exception):
                    raise
        return self.entries

    @staticmethod
    def _terminal_failure(attempt: JobAttempt[_Result]) -> bool:
        return (
            attempt.status in {Job.Status.FAILED, Job.Status.CANCELED}
            and bool(attempt.failures)
            and attempt.failures[-1].stage == "execution"
        )

    def complete(self) -> None:
        """Collect and replace confirmed failures within each entry's lifetime budget.

        The adapter's error factory supplies aggregate and submission errors.
        """
        self.collect()
        while True:
            retry = [
                index
                for index, entry in enumerate(self._entries)
                if not self._cancelled
                and self._submit is not None
                and entry.automatic_retries < self._max_retries
                and entry.attempts
                and self._terminal_failure(entry.attempts[-1])
                and entry.attempts[-1].status == Job.Status.FAILED
            ]
            if not retry:
                break
            self._submit_entries(retry, automatic=True)
            self.collect(retry)
        missing = [index for index, entry in enumerate(self._entries) if entry.result is None]
        if missing:
            details = ", ".join(f"{i} ({len(self._entries[i].attempts)} attempts)" for i in missing[:10])
            if len(missing) > 10:
                details += ", ..."
            causes = [
                entry.attempts[-1].failures[-1].cause
                for i in missing
                if (entry := self._entries[i]).attempts and entry.attempts[-1].failures
            ]
            first = causes[0] if causes else None
            msg = f"{len(missing)} batch entries have no result: {details}."
            if first is not None:
                msg += f" First failure: {first}"
            raise self._execution_error(msg) from first

    def resubmit(self, indices: Sequence[int], *, allow_unknown: bool = False) -> None:
        """Validate all selected entries before explicitly submitting replacements.

        Raises:
            ValueError: If indices are invalid or replacement could duplicate active work.
            TypeError: If allow_unknown is not a boolean.
        """
        if not isinstance(allow_unknown, bool):
            msg = "allow_unknown must be a boolean."
            raise TypeError(msg)
        selected = self._validate_indices(indices)
        for index in selected:
            entry = self._entries[index]
            if not entry.attempts:
                msg = f"Entry {index} has not been attempted; use submit() for its first submission."
                raise ValueError(msg)
            attempt = entry.attempts[-1]
            status = attempt.status
            if attempt.result is not None or attempt.status == Job.Status.DONE:
                status = Job.Status.DONE
            elif attempt.handle is not None:
                try:
                    status = attempt.handle.check()
                    self._set_attempt(index, replace(attempt, status=status))
                except Exception as exc:  # ruff:ignore[blind-except] An unavailable status makes replacement uncertain.
                    self.record_failure(index, "status", exc)
                    status = None
            if status in {Job.Status.FAILED, Job.Status.CANCELED}:
                continue
            if status is not None or not allow_unknown:
                msg = f"Cannot replace entry {index}: status {status}; unknown outcomes require allow_unknown=True."
                raise ValueError(msg)
        self._submit_entries(selected)

    def cancel(self) -> bool:
        """Disable automatic replacements and attempt every explicit cancellation.

        Returns:
            Whether all known handles accepted cancellation and no admission is uncertain.
        """
        self._cancelled = True
        success = True
        for index, entry in enumerate(self._entries):
            for position, attempt in enumerate(entry.attempts):
                if attempt.status in _TERMINAL:
                    continue
                if attempt.handle is None:
                    success = False
                    continue
                try:
                    attempt.handle.cancel()
                except Exception as exc:  # ruff:ignore[blind-except] Attempt the remaining explicit cancellations.
                    self.record_failure(index, "cancellation", exc, position)
                    success = False
        return success
