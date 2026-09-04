# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Native program-list submission remains separate from SDK concurrency."""

import pytest

from mqt.core.qdmi import Job, ProgramFormat
from mqt.core.qdmi.driver import open_device

PROGRAM = 'OPENQASM 3.0; include "stdgates.inc"; qubit q; bit c; x q; c = measure q;'


def test_single_program_list_and_indexed_results() -> None:
    """The one-program list uses the aggregate API without changing results."""
    device = open_device("mqt.ddsim.default")
    job = device.submit_programs([PROGRAM], ProgramFormat.QASM3, 32)
    assert job.wait()
    assert job.programs_num == 1
    assert job.get_counts(program_index=0) == {"1": 32}
    with pytest.raises(IndexError):
        job.get_counts(program_index=1)


def test_program_list_preserves_default_shots() -> None:
    """Omitting shots retains DDSIM's default instead of requiring a value."""
    device = open_device("mqt.ddsim.default")
    job = device.submit_programs([PROGRAM], ProgramFormat.QASM3)
    assert job.wait()
    assert job.get_counts() == {"1": job.num_shots}


def test_unsupported_aggregate_is_not_emulated() -> None:
    """DDSIM declines larger lists instead of disguising separate jobs."""
    device = open_device("mqt.ddsim.default")
    with pytest.raises(RuntimeError):
        device.submit_programs([PROGRAM, PROGRAM], ProgramFormat.QASM3, 32)


def test_program_list_preserves_job_failure() -> None:
    """An invalid program fails the real job, not a synthetic batch wrapper."""
    device = open_device("mqt.ddsim.default")
    job = device.submit_programs(["not an OpenQASM program"], ProgramFormat.QASM3, 32)
    assert job.wait()
    assert job.check() == Job.Status.FAILED
    with pytest.raises(RuntimeError):
        job.get_counts()


def test_program_list_rejects_invalid_payload_kinds() -> None:
    """Text cannot silently become binary and empty lists are invalid."""
    device = open_device("mqt.ddsim.default")
    with pytest.raises(ValueError, match="Binary program formats require exact-byte submission"):
        device.submit_programs([PROGRAM], ProgramFormat.QIR_BASE_MODULE, 32)
    with pytest.raises(ValueError, match="Setting programs"):
        device.submit_programs([], ProgramFormat.QASM3, 32)
