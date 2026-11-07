# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Native QDMI program groups shared by the SDK adapters."""

from __future__ import annotations

__lazy_modules__ = {"contextlib"}

from contextlib import suppress
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.core.qdmi import Device, Job, ProgramFormat
    from mqt.core.typing import QDMIJobParameters


def submit_programs(
    device: Device,
    programs: Sequence[tuple[str | bytes, ProgramFormat, int]],
    parameters: QDMIJobParameters | None = None,
    *,
    submitted: list[Job] | None = None,
) -> list[tuple[Job, int]]:
    """Submit compatible groups and preserve each input's native result index.

    Args:
        device: Destination QDMI device.
        programs: Payload, format, and shot count for each input.
        parameters: Common custom job parameters.
        submitted: Records accepted jobs, including those cancelled after a later failure.

    Returns:
        A job and program index per input, in input order.

    """
    groups: dict[tuple[ProgramFormat, int], list[int]] = {}
    for index, (_program, program_format, shots) in enumerate(programs):
        groups.setdefault((program_format, shots), []).append(index)
    submissions: dict[int, tuple[Job, int]] = {}
    handles: list[Job] = submitted if submitted is not None else []
    parameters = parameters or {}

    def submit_group(program_format: ProgramFormat, shots: int, indices: list[int]) -> None:
        """Admit one native list or submit its inputs independently."""
        native = None
        if len(indices) > 1:
            # A program format fixes the payload type for the whole group.
            payloads = cast("Sequence[str] | Sequence[bytes]", [programs[index][0] for index in indices])
            native = device.try_submit_job(payloads, program_format, shots, **parameters)
        if native is not None:
            handles.append(native)
            submissions.update((index, (native, program_index)) for program_index, index in enumerate(indices))
        else:
            for index in indices:
                handle = device.submit_job(
                    program=programs[index][0], program_format=program_format, num_shots=shots, **parameters
                )
                handles.append(handle)
                submissions[index] = (handle, 0)

    try:
        for (program_format, shots), indices in groups.items():
            submit_group(program_format, shots, indices)
    except BaseException:
        for handle in handles:
            with suppress(BaseException):
                handle.cancel()
        raise
    return [submissions[index] for index in range(len(programs))]
