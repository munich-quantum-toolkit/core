# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Native program lists preserve input order and individual outcomes."""

import pytest

from mqt.core.mlir import OutputFormat, compile_program
from mqt.core.qdmi import Job, ProgramFormat, open_device

PROGRAM = 'OPENQASM 3.0; include "stdgates.inc"; qubit q; bit c; x q; c = measure q;'


def test_single_program_and_indexed_results() -> None:
    """Single-program convenience preserves indexed results."""
    device = open_device("mqt.ddsim.default")
    job = device.submit_job(PROGRAM, ProgramFormat.QASM3, 32)
    assert job.wait()
    assert job.num_programs == 1
    assert job.get_counts(program_index=0) == {"1": 32}
    with pytest.raises(IndexError):
        job.get_counts(program_index=1)


def test_single_program_preserves_default_shots() -> None:
    """Omitting shots retains DDSIM's default instead of requiring a value."""
    device = open_device("mqt.ddsim.default")
    job = device.submit_job(PROGRAM, ProgramFormat.QASM3)
    assert job.wait()
    assert job.get_counts() == {"1": job.num_shots}


def test_try_submit_job_accepts_one_program() -> None:
    """The non-throwing submission probe uses the same scalar interface."""
    device = open_device("mqt.ddsim.default")
    job = device.try_submit_job(PROGRAM, ProgramFormat.QASM3, 4)
    assert job is not None
    assert job.wait()
    assert job.get_counts() == {"1": 4}


def test_native_programs_preserve_successful_siblings() -> None:
    """A failed program does not discard independently completed results."""
    device = open_device("mqt.ddsim.default")
    job = device.submit_job([PROGRAM, "invalid", PROGRAM], ProgramFormat.QASM3, 32)
    assert job.wait()
    assert job.num_programs == 3
    assert job.check() == Job.Status.FAILED
    assert [job.get_program_status(i) for i in range(3)] == [Job.Status.DONE, Job.Status.FAILED, Job.Status.DONE]
    assert job.get_program_status(1) == Job.Status.FAILED
    with pytest.raises(IndexError):
        job.get_program_status(3)
    assert job.get_counts(0) == {"1": 32}
    assert job.get_counts(2) == {"1": 32}
    with pytest.raises(RuntimeError):
        job.get_counts(1)


def test_binary_program_sequence() -> None:
    """A sequence of byte payloads uses the binary submission path."""
    bitcode = compile_program(PROGRAM, output=OutputFormat.QIR_BASE).to_bitcode()
    device = open_device("mqt.ddsim.default")
    job = device.submit_job((bitcode, bitcode), ProgramFormat.QIR_BASE_MODULE, 4)
    assert job.wait()
    assert job.get_program(bytes, 1) == bitcode
    assert [job.get_counts(i) for i in range(2)] == [{"1": 4}, {"1": 4}]


def test_program_list_preserves_job_failure() -> None:
    """An invalid program fails the real job, not a synthetic batch wrapper."""
    device = open_device("mqt.ddsim.default")
    job = device.submit_job("not an OpenQASM program", ProgramFormat.QASM3, 32)
    assert job.wait()
    assert job.check() == Job.Status.FAILED
    with pytest.raises(RuntimeError):
        job.get_counts()


def test_program_list_rejects_invalid_payload_kinds() -> None:
    """Text cannot silently become binary and empty lists are invalid."""
    device = open_device("mqt.ddsim.default")
    with pytest.raises(ValueError, match="Binary program formats require exact-byte submission"):
        device.submit_job(PROGRAM, ProgramFormat.QIR_BASE_MODULE, 32)
    with pytest.raises(ValueError, match="Setting programs"):
        device.submit_job([], ProgramFormat.QASM3, 32)
