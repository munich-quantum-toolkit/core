# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for capability-driven PennyLane program conversion."""

from __future__ import annotations

import re
import sys
from typing import cast

import numpy as np
import pytest

if sys.version_info < (3, 11):
    pytest.skip("PennyLane requires Python 3.11 or newer.", allow_module_level=True)

try:
    import pennylane as qp
except ImportError:
    pytest.skip("Install the PennyLane extra to run these tests.", allow_module_level=True)

from mqt.core.plugins.pennylane import (
    PennyLaneTranslationError,
    PennyLaneUnsupportedFormatError,
    PennyLaneUnsupportedOperationError,
    PennyLaneValidationError,
    convert_program,
)
from mqt.core.qdmi import Device as QDMIDevice
from mqt.core.qdmi import ProgramFormat

from .helpers import StubDevice, operation


def _device(value: StubDevice) -> QDMIDevice:
    """Present a test double at the public converter boundary.

    Returns:
        The fake device with the public QDMI type.
    """
    return cast("QDMIDevice", value)


def test_qasm3_prefers_and_resolves_braket_spellings() -> None:
    """Prefer QASM3 and emit only spellings advertised by a Braket-style device."""
    device = StubDevice(
        [
            operation("h", 1),
            operation("cnot", 2),
            operation("phaseshift", 1, 1),
            operation("xx", 2, 1),
        ],
        [ProgramFormat.QASM2, ProgramFormat.QASM3],
    )
    tape = qp.tape.QuantumScript(
        [
            qp.Hadamard("left"),
            qp.CNOT(wires=["left", "right"]),
            qp.PhaseShift(0.25, wires="right"),
            qp.IsingXX(0.5, wires=["right", "left"]),
        ],
        [qp.sample(wires=["right", "left"])],
        shots=10,
    )

    converted = convert_program(tape, _device(device), qp.wires.Wires(["left", "right"]))

    assert converted.program_format == ProgramFormat.QASM3
    assert converted.measurement_order == (1, 0)
    assert converted.payload == (
        "OPENQASM 3.0;\n"
        "qubit[2] q;\n"
        "bit[2] c;\n"
        "h q[0];\n"
        "cnot q[0],q[1];\n"
        "phaseshift(0.25) q[1];\n"
        "xx(0.5) q[1],q[0];\n"
        "c = measure q;\n"
    )
    assert "include" not in converted.payload
    assert "gate " not in converted.payload
    assert "pragma" not in converted.payload
    assert "inv @" not in converted.payload


def test_qasm3_resolves_ddsim_aliases_and_inverse_gates() -> None:
    """Resolve MQT Core-style aliases for controls, phases, rotations, and inverses."""
    device = StubDevice(
        [
            operation("cx", 2),
            operation("p", 1, 1),
            operation("sdg", 1),
            operation("tdg", 1),
            operation("sx", 1),
            operation("sxdg", 1),
            operation("rxx", 2, 1),
            operation("ryy", 2, 1),
            operation("rzz", 2, 1),
        ],
        [ProgramFormat.QASM3],
    )
    tape = qp.tape.QuantumScript(
        [
            qp.CNOT(wires=[0, 1]),
            qp.PhaseShift(-0.125, wires=1),
            qp.adjoint(qp.S)(0),
            qp.adjoint(qp.T)(1),
            qp.SX(0),
            qp.adjoint(qp.SX)(1),
            qp.IsingXX(0.1, wires=[0, 1]),
            qp.IsingYY(0.2, wires=[0, 1]),
            qp.IsingZZ(0.3, wires=[0, 1]),
        ],
        [qp.sample(wires=[0, 1])],
        shots=5,
    )

    payload = convert_program(tape, _device(device), qp.wires.Wires([0, 1])).payload

    assert "cx q[0],q[1];" in payload
    assert "p(-0.125) q[1];" in payload
    assert "sdg q[0];" in payload
    assert "tdg q[1];" in payload
    assert "sx q[0];" in payload
    assert "sxdg q[1];" in payload
    emitted = dict(re.findall(r"(rxx|ryy|rzz)\(([^)]+)\) q\[0\],q\[1\];", payload))
    assert {name: float(value) for name, value in emitted.items()} == {"rxx": 0.1, "ryy": 0.2, "rzz": 0.3}


def test_qasm3_failure_does_not_fall_back_to_qasm2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a QASM3 capability error visible when both formats are advertised."""
    device = StubDevice(
        [operation("h", 1)],
        [ProgramFormat.QASM3, ProgramFormat.QASM2],
    )
    tape = qp.tape.QuantumScript([qp.RX(0.2, 0)], [qp.sample(wires=0)], shots=4)
    qasm2_called = False

    def fail_if_called(*_args: object, **_kwargs: object) -> str:
        nonlocal qasm2_called
        qasm2_called = True
        return ""

    monkeypatch.setattr(qp, "to_openqasm", fail_if_called)
    with pytest.raises(PennyLaneUnsupportedOperationError, match="RX"):
        convert_program(tape, _device(device), qp.wires.Wires([0]))
    assert not qasm2_called


def test_qasm2_fallback_uses_pennylane_serializer_without_rotations() -> None:
    """Use PennyLane's QASM2 serializer only when QASM3 is unavailable."""
    device = StubDevice(
        [operation("h", 1), operation("cx", 2), operation("rx", 1, 1)],
        [ProgramFormat.QASM2],
    )
    tape = qp.tape.QuantumScript(
        [qp.Hadamard(0), qp.CNOT(wires=[0, 1]), qp.RX(0.25, 1)],
        [qp.sample(wires=[0, 1])],
        shots=10,
    )

    converted = convert_program(tape, _device(device), qp.wires.Wires([0, 1]))

    assert converted.program_format == ProgramFormat.QASM2
    assert converted.payload == (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[2];\n"
        "creg c[2];\n"
        "h q[0];\n"
        "cx q[0],q[1];\n"
        "rx(0.25) q[1];\n"
        "measure q[0] -> c[0];\n"
        "measure q[1] -> c[1];\n"
    )


def test_qasm2_rejects_non_intersection_operation() -> None:
    """Reject an operation the serializer knows when the QDMI device does not."""
    device = StubDevice([operation("h", 1)], [ProgramFormat.QASM2])
    tape = qp.tape.QuantumScript([qp.CNOT(wires=[0, 1])], [qp.sample(wires=[0, 1])], shots=2)

    with pytest.raises(PennyLaneUnsupportedOperationError, match="CNOT"):
        convert_program(tape, _device(device), qp.wires.Wires([0, 1]))


def test_qasm2_wraps_serializer_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose serializer failures as focused translation errors."""
    device = StubDevice([operation("h", 1)], [ProgramFormat.QASM2])
    tape = qp.tape.QuantumScript([qp.Hadamard(0)], [qp.sample(wires=0)], shots=2)

    def fail(*_args: object, **_kwargs: object) -> str:
        msg = "serializer failed"
        raise ValueError(msg)

    monkeypatch.setattr(qp, "to_openqasm", fail)
    with pytest.raises(PennyLaneTranslationError, match="serializer failed"):
        convert_program(tape, _device(device), qp.wires.Wires([0]))


def test_rejects_device_without_qasm() -> None:
    """Fail before submission when neither supported format is advertised."""
    device = StubDevice([], [ProgramFormat.QIR_BASE_STRING])
    tape = qp.tape.QuantumScript([], [qp.sample(wires=0)], shots=2)

    with pytest.raises(PennyLaneUnsupportedFormatError, match="OpenQASM 3 or OpenQASM 2"):
        convert_program(tape, _device(device), qp.wires.Wires([0]))


@pytest.mark.parametrize("parameter", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_parameters(parameter: float) -> None:
    """Reject non-finite bound parameters before submission."""
    device = StubDevice([operation("rx", 1, 1)], [ProgramFormat.QASM3])
    tape = qp.tape.QuantumScript([qp.RX(parameter, 0)], [qp.sample(wires=0)], shots=2)

    with pytest.raises(PennyLaneValidationError, match="non-finite"):
        convert_program(tape, _device(device), qp.wires.Wires([0]))


def test_validates_operation_topology() -> None:
    """Honor operation-specific QDMI site pairs."""
    device = StubDevice(
        [operation("cx", 2, site_pairs=[(0, 1)])],
        [ProgramFormat.QASM3],
        qubits=3,
    )
    tape = qp.tape.QuantumScript([qp.CNOT(wires=[1, 0])], [qp.sample(wires=[0, 1])], shots=2)

    with pytest.raises(PennyLaneValidationError, match=r"not advertised on device wires \(1, 0\)"):
        convert_program(tape, _device(device), qp.wires.Wires([0, 1, 2]))
