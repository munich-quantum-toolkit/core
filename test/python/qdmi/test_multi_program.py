# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Native QDMI program lists and indexed results."""

from __future__ import annotations

import pytest

from mqt.core.qdmi import CustomProperty, Job, ProgramFormat
from mqt.core.qdmi.driver import open_device

ZERO = "OPENQASM 3; qubit q; bit c; c = measure q;"
ONE = 'OPENQASM 3; include "stdgates.inc"; qubit q; bit c; x q; c = measure q;'
TWO = 'OPENQASM 3; include "stdgates.inc"; qubit[2] q; x q;'


@pytest.mark.parametrize("exact_bytes", [False, True])
def test_sampling_batch_preserves_input_order(*, exact_bytes: bool) -> None:
    """Each program retains its own payload, shots, and counts."""
    device = open_device("mqt.ddsim.default")
    texts = [ZERO, ONE]
    programs = [text.encode() + b"\0" for text in texts] if exact_bytes else texts
    job = device.submit_job(programs, ProgramFormat.QASM3, num_shots=32)
    assert job.wait()
    assert job.check() == Job.Status.DONE
    assert job.num_programs == 2
    assert job.program == ZERO
    assert job.program_bytes == ZERO.encode() + b"\0"
    for index, text in enumerate(texts):
        assert job.get_program(index) == text
        assert job.get_program(bytes, index) == text.encode() + b"\0"
        assert job.get_program_status(index) == Job.Status.DONE
        assert job.get_counts(index) == {str(index): 32}
        assert job.get_shots(index) == [str(index)] * 32
        assert job.get_custom_result(CustomProperty.CUSTOM1, bytes, index) is None
    with pytest.raises(IndexError, match="Out of range"):
        job.get_program(2)
    with pytest.raises(IndexError, match="Out of range"):
        job.get_counts(2)
    with pytest.raises(IndexError, match="Out of range"):
        job.get_program_status(2)


def test_failed_program_preserves_successful_siblings() -> None:
    """A failed program does not discard earlier or later results."""
    device = open_device("mqt.ddsim.default")
    job = device.submit_job([ZERO, "invalid OpenQASM", ONE], ProgramFormat.QASM3, num_shots=8)
    assert job.wait()
    assert job.check() == Job.Status.FAILED
    assert [job.get_program_status(i) for i in range(3)] == [Job.Status.DONE, Job.Status.FAILED, Job.Status.DONE]
    assert job.get_counts(0) == {"0": 8}
    assert job.get_counts(2) == {"1": 8}
    with pytest.raises(RuntimeError, match="Bad state"):
        job.get_counts(1)


def test_state_batch_keeps_dense_and_sparse_results_separate() -> None:
    """Programs with different widths retain independent simulation state."""
    device = open_device("mqt.ddsim.default")
    job = device.submit_job([ZERO, TWO], ProgramFormat.QASM3, num_shots=0)
    assert job.wait()
    assert job.get_dense_statevector(0) == pytest.approx([1, 0])
    assert job.get_dense_statevector(1) == pytest.approx([0, 0, 0, 1])
    assert job.get_dense_probabilities(0) == pytest.approx([1, 0])
    assert job.get_dense_probabilities(1) == pytest.approx([0, 0, 0, 1])
    assert job.get_sparse_statevector(0) == {"0": 1}
    assert job.get_sparse_statevector(1) == {"11": 1}
    assert job.get_sparse_probabilities(0) == {"0": 1}
    assert job.get_sparse_probabilities(1) == {"11": 1}


def test_empty_batch_is_rejected() -> None:
    """A native job must contain at least one program."""
    device = open_device("mqt.ddsim.default")
    with pytest.raises(ValueError, match="at least one program"):
        device.submit_job([], ProgramFormat.QASM3)


@pytest.mark.parametrize("invalid", [b"OPENQASM 3;", b"OPENQASM 3;\0garbage\0"])
def test_batch_validates_every_text_payload(invalid: bytes) -> None:
    """A later malformed payload rejects the entire list before execution."""
    device = open_device("mqt.ddsim.default")
    with pytest.raises(ValueError, match="Setting program: Invalid argument"):
        device.submit_job([ZERO.encode() + b"\0", invalid], ProgramFormat.QASM3)


def test_text_batch_rejects_binary_format() -> None:
    """Known binary formats still require exact bytes for every program."""
    device = open_device("mqt.ddsim.default")
    with pytest.raises(ValueError, match="require exact-byte submission"):
        device.submit_job([ZERO, ONE], ProgramFormat.QIR_BASE_MODULE)


def test_binary_batch_reaches_the_device_without_text_conversion() -> None:
    """Arbitrary binary bytes reach the device's format support check."""
    device = open_device("mqt.ddsim.default")
    with pytest.raises(RuntimeError, match="Setting program: Not supported"):
        device.submit_job([b"\x00\xff", b"\x80"], ProgramFormat.QIR_BASE_MODULE)


@pytest.mark.parametrize("as_list", [False, True])
def test_text_submission_preserves_one_terminating_null(*, as_list: bool) -> None:
    """Text with an explicit terminator matches ordinary text submission."""
    device = open_device("mqt.ddsim.default")
    program = ZERO + "\0"
    job = device.submit_job([program, program] if as_list else program, ProgramFormat.QASM3, 4)
    assert job.wait()
    assert job.program == ZERO
    assert job.get_counts(job.num_programs - 1) == {"0": 4}


def test_try_submit_job_only_falls_back_on_unsupported_programs() -> None:
    """Unsupported admission returns None; malformed input still raises."""
    device = open_device("mqt.ddsim.default")
    assert device.try_submit_job([ZERO], ProgramFormat.CUSTOM1) is None
    with pytest.raises(ValueError, match="embedded null"):
        device.try_submit_job([ZERO, "bad\0text"], ProgramFormat.QASM3)
    with pytest.raises(ValueError, match="at least one program"):
        device.try_submit_job([], ProgramFormat.QASM3)
    job = device.try_submit_job([ZERO, ONE], ProgramFormat.QASM3, 4)
    assert job is not None
    assert job.wait()
    assert [job.get_counts(index) for index in range(2)] == [{"0": 4}, {"1": 4}]
