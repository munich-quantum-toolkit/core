# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for capability-driven PennyLane program conversion."""

# ruff: file-ignore[missing-return-type-private-function]

from __future__ import annotations

import re
from unittest.mock import Mock

import numpy as np
import pytest

try:
    import pennylane as qp
except ImportError:
    pytest.skip("Install the PennyLane extra to run these tests.", allow_module_level=True)

from mqt.core.plugins.pennylane import (
    PennyLaneTranslationError,
    PennyLaneUnsupportedOperationError,
    PennyLaneValidationError,
    QDMIDevice,
)
from mqt.core.plugins.pennylane.converter import _ProgramConverter  # ruff:ignore[import-private-name]
from mqt.core.qdmi import ProgramFormat

from .helpers import StubDevice, operation, patch_open_device

_DYNAMIC_FEATURES = frozenset({
    "mid-circuit-measurement",
    "measured-qubit-reuse",
    "measurement-result-use",
    "boolean-computation",
    "forward-branching",
})


def test_qasm3_emits_measurement_feedback_and_reset() -> None:
    """Encode PennyLane's one-shot MCM bundle in the declared output bits."""
    qdmi = StubDevice(
        [operation("measure", 1), operation("reset", 1), operation("x", 1)],
        [ProgramFormat.OPENQASM3],
        program_features=tuple(_DYNAMIC_FEATURES),
    )

    def circuit():
        measurement = qp.measure(0, reset=True)
        qp.cond(measurement, qp.PauliX)(1)
        return qp.sample(wires=[0, 1])

    tape = qp.tape.make_qscript(circuit)()
    converted = _ProgramConverter(
        qdmi,  # ty: ignore[invalid-argument-type]
        qp.wires.Wires([0, 1]),
        ProgramFormat.OPENQASM3,
        _DYNAMIC_FEATURES,
    ).convert(tape)

    assert "bit[3] c;" in converted.payload
    assert "c[2] = measure q[0];\nreset q[0];" in converted.payload
    assert "if (c[2] == 1) { x q[1]; }" in converted.payload
    assert converted.mcm_slot_by_uid[tape.operations[0].meas_uid] == 2


@pytest.mark.parametrize(
    "operations",
    [
        [qp.ops.MidMeasure(qp.wires.Wires([0]), meas_uid="")],
        [
            qp.ops.MidMeasure(qp.wires.Wires([0]), meas_uid="same"),
            qp.ops.MidMeasure(qp.wires.Wires([1]), meas_uid="same"),
        ],
    ],
)
def test_qasm3_rejects_invalid_mid_measurement_ids(operations: list[qp.operation.Operator]) -> None:
    """Require stable, unique IDs for mid-circuit measurement results."""
    qdmi = StubDevice([operation("measure", 1)], [ProgramFormat.OPENQASM3], program_features=tuple(_DYNAMIC_FEATURES))
    converter = _ProgramConverter(
        qdmi,  # ty: ignore[invalid-argument-type]
        qp.wires.Wires([0, 1]),
        ProgramFormat.OPENQASM3,
        _DYNAMIC_FEATURES,
    )

    with pytest.raises(PennyLaneValidationError, match="unique nonempty measurement IDs"):
        converter.convert(qp.tape.QuantumScript(operations))


def test_qasm3_rejects_mid_measurement_without_payload_capabilities() -> None:
    """Reject mid-circuit measurement unless the selected payload supports it."""
    qdmi = StubDevice([operation("measure", 1)], [ProgramFormat.OPENQASM3])
    converter = _ProgramConverter(
        qdmi,  # ty: ignore[invalid-argument-type]
        qp.wires.Wires([0]),
        ProgramFormat.OPENQASM3,
    )
    tape = qp.tape.QuantumScript([qp.ops.MidMeasure(qp.wires.Wires([0]), meas_uid="measurement")])

    with pytest.raises(PennyLaneUnsupportedOperationError, match="does not support mid-circuit measurement"):
        converter.convert(tape)


def test_qasm3_prefers_and_resolves_braket_spellings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefer QASM3 and emit only spellings advertised by a Braket-style device."""
    qdmi = StubDevice(
        [
            operation("h", 1),
            operation("cnot", 2),
            operation("phaseshift", 1, 1),
            operation("xx", 2, 1),
        ],
        [ProgramFormat.OPENQASM2, ProgramFormat.OPENQASM3],
        result_factory=lambda _program, shots: ["10"] * shots,
    )
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=["left", "right"], shots=10)

    @qp.qnode(device)
    def circuit():
        qp.Hadamard("left")
        qp.CNOT(wires=["left", "right"])
        qp.PhaseShift(0.25, wires="right")
        qp.IsingXX(0.5, wires=["right", "left"])
        return qp.sample(wires=["right", "left"])

    samples = circuit()
    payload, program_format, _shots, _parameters = qdmi.submissions[0]

    assert program_format == ProgramFormat.OPENQASM3
    assert payload == (
        "OPENQASM 3.0;\n"
        "qubit[2] q;\n"
        "bit[2] c;\n"
        "h q[0];\n"
        "cnot q[0],q[1];\n"
        "phaseshift(0.25) q[1];\n"
        "xx(0.5) q[1],q[0];\n"
        "c[0] = measure q[0];\n"
        "c[1] = measure q[1];\n"
    )
    assert "include" not in payload
    assert "gate " not in payload
    assert "pragma" not in payload
    assert "inv @" not in payload
    # QDMI output slot zero is the rightmost bit. The requested measurement
    # order places slot one before slot zero.
    np.testing.assert_array_equal(samples[0], [1, 0])


def test_qasm3_resolves_ddsim_aliases_and_inverse_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve MQT Core-style aliases for controls, phases, rotations, and inverses."""
    qdmi = StubDevice(
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
        [ProgramFormat.OPENQASM3],
    )
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=5)

    @qp.qnode(device)
    def circuit():
        qp.CNOT(wires=[0, 1])
        qp.PhaseShift(-0.125, wires=1)
        qp.adjoint(qp.S)(0)
        qp.adjoint(qp.T)(1)
        qp.SX(0)
        qp.adjoint(qp.SX)(1)
        qp.IsingXX(0.1, wires=[0, 1])
        qp.IsingYY(0.2, wires=[0, 1])
        qp.IsingZZ(0.3, wires=[0, 1])
        return qp.sample(wires=[0, 1])

    circuit()
    payload = qdmi.submissions[0][0]

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
    # PennyLane's OpenQASM 2 serializer spells U3 as `u3`, and the device
    # advertises it, so a fallback to OpenQASM 2 would silently succeed. The
    # OpenQASM 3 operation table has no U3 row, so the QASM3 path must fail.
    qdmi = StubDevice([operation("u3", 1, 3)], [ProgramFormat.OPENQASM3, ProgramFormat.OPENQASM2])
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=1, shots=4)
    tape = qp.tape.QuantumScript([qp.U3(0.1, 0.2, 0.3, wires=0)], [qp.sample(wires=0)], shots=4)
    qasm2_called = False

    def fail_if_called(*_args: object, **_kwargs: object) -> str:
        nonlocal qasm2_called
        qasm2_called = True
        return ""

    monkeypatch.setattr(qp, "to_openqasm", fail_if_called)

    @qp.qnode(device)
    def circuit():
        qp.U3(0.1, 0.2, 0.3, wires=0)
        return qp.sample(wires=0)

    # Preprocessing stops the operation before conversion.
    with pytest.raises(PennyLaneUnsupportedOperationError):
        circuit()
    # Conversion refuses it as well, rather than retrying with OpenQASM 2.
    with pytest.raises(PennyLaneUnsupportedOperationError, match="OpenQASM 3"):
        device.execute(tape)
    assert not qasm2_called
    assert not qdmi.submissions


def test_qasm2_fallback_uses_pennylane_serializer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use PennyLane's QASM2 serializer only when QASM3 is unavailable."""
    qdmi = StubDevice(
        [operation("h", 1), operation("cx", 2), operation("rx", 1, 1)],
        [ProgramFormat.OPENQASM2],
    )
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=10)

    @qp.qnode(device)
    def circuit():
        qp.Hadamard(0)
        qp.CNOT(wires=[0, 1])
        qp.RX(0.25, 1)
        return qp.sample(wires=[0, 1])

    circuit()
    payload, program_format, _shots, _parameters = qdmi.submissions[0]

    assert program_format == ProgramFormat.OPENQASM2
    assert payload == (
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


def test_qasm2_rejects_non_intersection_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject an operation the serializer knows when the QDMI device does not."""
    qdmi = StubDevice([operation("h", 1)], [ProgramFormat.OPENQASM2])
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=2)
    tape = qp.tape.QuantumScript([qp.CNOT(wires=[0, 1])], [qp.sample(wires=[0, 1])], shots=2)

    @qp.qnode(device)
    def circuit():
        qp.CNOT(wires=[0, 1])
        return qp.sample(wires=[0, 1])

    # Preprocessing stops the operation before conversion.
    with pytest.raises(PennyLaneUnsupportedOperationError):
        circuit()
    # Conversion stops it again for a caller that executes a tape directly,
    # because PennyLane's OpenQASM 2 serializer cannot reject it.
    with pytest.raises(PennyLaneUnsupportedOperationError, match="CNOT"):
        device.execute(tape)
    assert not qdmi.submissions


def test_qasm2_wraps_serializer_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose serializer failures as focused translation errors."""
    qdmi = StubDevice([operation("h", 1)], [ProgramFormat.OPENQASM2])
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=1, shots=2)

    def fail(*_args: object, **_kwargs: object) -> str:
        msg = "serializer failed"
        raise ValueError(msg)

    monkeypatch.setattr(qp, "to_openqasm", fail)

    @qp.qnode(device)
    def circuit():
        qp.Hadamard(0)
        return qp.sample(wires=0)

    with pytest.raises(PennyLaneTranslationError, match="serializer failed"):
        circuit()


@pytest.mark.parametrize("parameter", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_parameters(monkeypatch: pytest.MonkeyPatch, parameter: float) -> None:
    """Reject non-finite bound parameters before submission."""
    qdmi = StubDevice([operation("rx", 1, 1)], [ProgramFormat.OPENQASM3])
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=1, shots=2)

    @qp.qnode(device)
    def circuit():
        qp.RX(parameter, 0)
        return qp.sample(wires=0)

    with pytest.raises(PennyLaneValidationError, match="non-finite"):
        circuit()
    assert not qdmi.submissions


def test_validates_operation_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    """Honor operation-specific QDMI site pairs."""
    qdmi = StubDevice(
        [operation("cx", 2, site_pairs=[(0, 1)])],
        [ProgramFormat.OPENQASM3],
        qubits=3,
    )
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=3, shots=2)

    @qp.qnode(device)
    def circuit():
        qp.CNOT(wires=[1, 0])
        return qp.sample(wires=[0, 1])

    with pytest.raises(PennyLaneValidationError, match=r"not advertised on device wires \(1, 0\)"):
        circuit()
    assert not qdmi.submissions


def test_reuses_session_contract_checks_without_skipping_input_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reuse valid gate locations across tapes, not inputs or other sessions."""
    rx = operation("rx", 1, 1, sites=[0])
    qdmi = StubDevice([rx], [ProgramFormat.OPENQASM3])
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=2)
    query_sites = Mock(wraps=rx.sites)
    monkeypatch.setattr(rx, "sites", query_sites)

    def tape(angle: float, wire: int = 0):
        return qp.tape.QuantumScript([qp.RX(angle, wires=wire)], [qp.sample(wires=[0, 1])], shots=2)

    device.execute((tape(0.1), tape(0.2)))
    device.execute(tape(0.3))
    query_sites.assert_called_once()
    assert len({program for program, *_ in qdmi.submissions}) == 3
    with pytest.raises(PennyLaneValidationError, match="non-finite"):
        device.execute(tape(float("nan")))
    with pytest.raises(PennyLaneValidationError, match="not advertised on device wire 1"):
        device.execute(tape(0.4, wire=1))
    assert len(qdmi.submissions) == 3

    query_sites.reset_mock()
    QDMIDevice("fake.qdmi", wires=2, shots=2).execute(tape(0.5))
    query_sites.assert_called_once()


def test_rejects_operation_on_an_unadvertised_site(monkeypatch: pytest.MonkeyPatch) -> None:
    """Honor the single-qubit sites a QDMI operation advertises."""
    qdmi = StubDevice([operation("h", 1, sites=[0])], [ProgramFormat.OPENQASM3])
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=2)

    @qp.qnode(device)
    def circuit():
        qp.Hadamard(0)
        qp.Hadamard(1)
        return qp.sample(wires=[0, 1])

    with pytest.raises(PennyLaneValidationError, match="not advertised on device wire 1"):
        circuit()
    assert not qdmi.submissions


def test_falls_back_to_the_device_coupling_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the device topology when an operation advertises no site pairs."""
    qdmi = StubDevice(
        [operation("h", 1), operation("cx", 2)],
        [ProgramFormat.OPENQASM3],
        qubits=3,
        coupling_map=[(0, 1)],
        result_factory=lambda _program, shots: ["000"] * shots,
    )
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=3, shots=2)

    @qp.qnode(device)
    def connected():
        qp.CNOT(wires=[1, 0])
        return qp.sample(wires=[0, 1])

    @qp.qnode(device)
    def disconnected():
        qp.CNOT(wires=[0, 2])
        return qp.sample(wires=[0, 2])

    # The coupling map is undirected, so the reversed edge is connected too.
    connected()
    assert len(qdmi.submissions) == 1
    with pytest.raises(PennyLaneValidationError, match=r"does not connect wires \(0, 2\)"):
        disconnected()
    assert len(qdmi.submissions) == 1


def test_accepts_advertised_site_pairs_and_wider_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept an advertised site pair, and skip loci checks above two wires."""
    qdmi = StubDevice(
        [operation("h", 1), operation("cx", 2, site_pairs=[(0, 1)]), operation("ccx", 3)],
        [ProgramFormat.OPENQASM3],
        qubits=3,
        result_factory=lambda _program, shots: ["000"] * shots,
    )
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=3, shots=2)

    @qp.qnode(device)
    def circuit():
        qp.CNOT(wires=[0, 1])
        qp.Toffoli(wires=[0, 1, 2])
        return qp.sample(wires=[0, 1, 2])

    circuit()
    payload = qdmi.submissions[0][0]

    assert "cx q[0],q[1];" in payload
    assert "ccx q[0],q[1],q[2];" in payload


@pytest.mark.parametrize(
    ("advertised", "expected"),
    [
        (operation("rx", 2, 1), "advertises 2 wires"),
        (operation("rx", 1, 0), "advertises 0 parameters"),
    ],
)
def test_rejects_contradictory_qdmi_metadata(
    monkeypatch: pytest.MonkeyPatch, advertised: object, expected: str
) -> None:
    """Reject a QDMI operation whose arity contradicts the operation table."""
    qdmi = StubDevice([advertised], [ProgramFormat.OPENQASM3])  # ty: ignore[invalid-argument-type]
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=1, shots=2)

    @qp.qnode(device)
    def circuit():
        qp.RX(0.5, 0)
        return qp.sample(wires=0)

    with pytest.raises(PennyLaneValidationError, match=expected):
        circuit()
    assert not qdmi.submissions


def test_measurement_without_wires_samples_every_device_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sample every device wire when the measurement names none."""
    qdmi = StubDevice(
        [operation("x", 1)],
        [ProgramFormat.OPENQASM3],
        result_factory=lambda _program, shots: ["10"] * shots,
    )
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=4)

    @qp.qnode(device)
    def circuit():
        qp.PauliX(1)
        return qp.sample()

    samples = circuit()

    assert samples.shape == (4, 2)
    np.testing.assert_array_equal(samples[0], [0, 1])


def test_converts_an_unpreprocessed_tape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert a tape handed straight to the device, without preprocessing."""
    qdmi = StubDevice(
        [operation("h", 1)],
        [ProgramFormat.OPENQASM3],
        result_factory=lambda _program, shots: ["00"] * shots,
    )
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=2)

    # Preprocessing always names the measured wires. A tape that skips it may
    # carry no measurement, or one that names none; both sample every wire.
    for measurements in ([], [qp.sample()]):
        samples = np.asarray(device.execute(qp.tape.QuantumScript([qp.Hadamard(0)], measurements, shots=2)))
        assert samples.shape == (2, 2)

    with pytest.raises(PennyLaneValidationError, match="not a device wire"):
        device.execute(qp.tape.QuantumScript([qp.Hadamard(5)], [qp.sample(wires=0)], shots=2))
