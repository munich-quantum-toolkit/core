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
from typing import TYPE_CHECKING, Generic, TypeVar

from mqt.core.qdmi import Job

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator, Sequence

__all__ = ["Batch", "BatchEntry", "JobAttempt", "JobFailure"]

_Result = TypeVar("_Result")
_TERMINAL = {Job.Status.DONE, Job.Status.FAILED, Job.Status.CANCELED}


@dataclass(frozen=True, slots=True)
class JobFailure:
    """An original exception and the operation that raised it."""

    stage: str
    cause: BaseException


@dataclass(frozen=True, slots=True)
class JobAttempt(Generic[_Result]):
    """An attempt and its result index, or an uncertain submission without a handle."""

    handle: Job | None = None
    program_index: int = 0
    status: Job.Status | None = None
    result: _Result | None = None
    failures: tuple[JobFailure, ...] = ()


@dataclass(frozen=True, slots=True)
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


class Batch(Generic[_Result]):
    """Shared recovery engine for the QDMI adapters.

    Adapters prepare every input before constructing a batch and own result
    conversion; decoded results must not be None. Use adapter job handles to
    submit or recover work. Operations on one batch are synchronous; concurrent
    calls on the same batch are unsupported.
    """

    def __init__(
        self,
        entries: Sequence[BatchEntry[_Result]],
        *,
        submit: Callable[[int], Job] | None,
        decode: Callable[[int, Job, int], _Result],
        submission_error: Callable[[str], Exception],
        execution_error: Callable[[str], Exception],
        max_retries: int,
        on_submit: Callable[[Sequence[int]], None] | None = None,
        submit_programs: Callable[[Sequence[int]], Job | None] | None = None,
        group_by: Sequence[Hashable] | None = None,
    ) -> None:
        """Capture prepared entries, adapter callbacks, and a lifetime retry limit."""
        self._entries = list(entries)
        self._submit = submit
        self._decode = decode
        self._submission_error = submission_error
        self._execution_error = execution_error
        self._max_retries = max_retries
        self._cancelled = False
        self._on_submit = on_submit
        self._submit_programs = submit_programs
        self._group_by = tuple(group_by) if group_by is not None else None

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
        if any(not 0 <= index < len(self._entries) for index in selected) or len(set(selected)) != len(selected):
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
        groups: dict[Hashable, list[int]] = {}
        for index in indices:
            key = self._group_by[index] if self._group_by is not None else index
            groups.setdefault(key, []).append(index)
        for group in groups.values():
            if len(group) > 1 and self._submit_programs is not None and self._admit(group, automatic=automatic):
                continue
            for index in group:
                self._admit([index], automatic=automatic)

    def _admit(self, indices: Sequence[int], *, automatic: bool) -> bool:
        previous = [self._entries[index] for index in indices]
        for index, entry in zip(indices, previous, strict=True):
            self._entries[index] = replace(
                entry,
                attempts=(*entry.attempts, JobAttempt()),
                automatic_retries=entry.automatic_retries + int(automatic),
            )
        stage = "submission"
        try:  # ruff:ignore[too-many-statements-in-try-clause] Retain every accepted handle before tracking.
            if len(indices) > 1:
                assert self._submit_programs is not None
                handle = self._submit_programs(indices)
            else:
                assert self._submit is not None
                handle = self._submit(indices[0])
            if handle is None:
                # NOTSUPPORTED before submission consumes neither an attempt nor a retry.
                for index, entry in zip(indices, previous, strict=True):
                    self._entries[index] = entry
                return False
            for program_index, index in enumerate(indices):
                self._set_attempt(index, JobAttempt(handle=handle, program_index=program_index))
            stage = "tracking"
            if self._on_submit is not None:
                self._on_submit(indices)
        except BaseException as exc:
            for index in indices:
                self.record_failure(index, stage, exc)
            if not isinstance(exc, Exception):
                raise
            msg = f"Failed to submit batch entries {list(indices)}: {exc}"
            raise self._submission_error(msg) from exc
        return True

    def _refresh(self, indices: Sequence[int], *, wait: bool = False) -> Iterator[list[int]]:
        groups: dict[int, list[int]] = {}
        for index in indices:
            entry = self._entries[index]
            if entry.attempts and (handle := entry.attempts[-1].handle) is not None:
                groups.setdefault(id(handle), []).append(index)
        for group in groups.values():
            pending = [i for i in group if self._entries[i].attempts[-1].status not in _TERMINAL]
            if pending:
                handle = self._entries[pending[0]].attempts[-1].handle
                assert handle is not None
                stage = "wait" if wait else "status"
                try:  # ruff:ignore[too-many-statements-in-try-clause] Keep shared query failures on every affected entry.
                    if wait and not handle.wait():
                        msg = "Timed out waiting for the QDMI job."
                        raise TimeoutError(msg)  # ruff:ignore[raise-within-try] Record timeouts with other wait failures.
                    stage = "status"
                    aggregate = handle.check()
                    outcomes = (
                        handle.program_statuses if aggregate in {Job.Status.FAILED, Job.Status.CANCELED} else None
                    )
                    # Resolve individual outcomes before attributing an aggregate failure.
                    for index in pending:
                        attempt = self._entries[index].attempts[-1]
                        status = aggregate if outcomes is None else outcomes[attempt.program_index]
                        self._set_attempt(index, replace(attempt, status=status))
                except BaseException as exc:
                    for index in pending:
                        attempt = self._entries[index].attempts[-1]
                        self._set_attempt(index, replace(attempt, status=None))
                        self.record_failure(index, stage, exc)
                    if not isinstance(exc, Exception):
                        raise
            yield group

    def statuses(self) -> tuple[Job.Status | None, ...]:
        """Query each shared job once and return ordered per-entry outcomes.

        Provider query failures propagate after being retained on the attempts.

        Returns:
            None for untouched entries or uncertain submissions.

        """
        for group in self._refresh(range(len(self._entries))):
            for index in group:
                attempt = self._entries[index].attempts[-1]
                if attempt.status is None and attempt.failures:
                    raise attempt.failures[-1].cause
        return tuple(entry.attempts[-1].status if entry.attempts else None for entry in self._entries)

    def collect(self, indices: Sequence[int] | None = None) -> tuple[BatchEntry[_Result], ...]:
        """Collect existing attempts without replacements or aggregate failure.

        Returns:
            Ordered entry snapshots, including every recorded failure.
        """
        selected = range(len(self._entries)) if indices is None else indices
        for group in self._refresh(selected, wait=True):
            for index in group:
                entry = self._entries[index]
                attempt = entry.attempts[-1]
                if entry.result is not None or self._terminal_failure(attempt) or attempt.status is None:
                    continue
                if attempt.status != Job.Status.DONE:
                    msg = f"QDMI program did not complete successfully: {attempt.status.name}."
                    self.record_failure(index, "execution", RuntimeError(msg))
                    continue
                assert attempt.handle is not None
                try:
                    result = self._decode(index, attempt.handle, attempt.program_index)
                    self._set_attempt(index, replace(attempt, result=result))
                except BaseException as exc:
                    self.record_failure(index, "result", exc)
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
        while not self._cancelled and self._submit is not None:
            retry = [
                index
                for index, entry in enumerate(self._entries)
                if entry.automatic_retries < self._max_retries
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
            first = next(
                (
                    entry.attempts[-1].failures[-1].cause
                    for i in missing
                    if (entry := self._entries[i]).attempts and entry.attempts[-1].failures
                ),
                None,
            )
            msg = f"{len(missing)} batch entries have no result: {details}."
            if first is not None:
                msg += f" First failure: {first}"
            raise self._execution_error(msg) from first

    def resubmit(self, indices: Sequence[int], *, allow_unknown: bool = False) -> None:
        """Validate all selected entries before explicitly submitting replacements.

        Raises:
            ValueError: If indices are invalid or replacement could duplicate active work.
        """
        selected = self._validate_indices(indices)
        if any(not self._entries[index].attempts for index in selected):
            msg = "Selected entries have not been attempted; use submit() for their first submission."
            raise ValueError(msg)
        for _ in self._refresh(selected):
            pass
        for index in selected:
            attempt = self._entries[index].attempts[-1]
            status = attempt.status
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
        groups: dict[int, list[tuple[int, int]]] = {}
        for index, entry in enumerate(self._entries):
            for position, attempt in enumerate(entry.attempts):
                if attempt.status in _TERMINAL:
                    continue
                if attempt.handle is None:
                    success = False
                    continue
                groups.setdefault(id(attempt.handle), []).append((index, position))
        for group in groups.values():
            index, position = group[0]
            handle = self._entries[index].attempts[position].handle
            assert handle is not None
            try:
                handle.cancel()
            except Exception as exc:  # ruff:ignore[blind-except] Attempt the remaining explicit cancellations.
                for index, position in group:
                    self.record_failure(index, "cancellation", exc, position)
                success = False
        return success
