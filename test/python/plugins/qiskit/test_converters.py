# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for circuit format converters."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit import QuantumCircuit

from mqt.core.plugins.qiskit import UnsupportedOperationError, qiskit_to_iqm_json

if TYPE_CHECKING:
    from test.python.plugins.qiskit.conftest import MockQDMIDevice


def test_qiskit_to_iqm_json_simple_circuit(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test conversion of a simple circuit to IQM JSON."""
    device = mock_qdmi_device_factory(
        name="IQM Device",
        num_qubits=2,
        operations=["r", "cz", "measure", "barrier"],
    )

    qc = QuantumCircuit(2, 2)
    qc.r(1.5708, 0.0, 0)
    qc.cz(0, 1)
    qc.measure_all()

    json_str = qiskit_to_iqm_json(qc, device)  # type: ignore[arg-type]
    program = json.loads(json_str)

    assert "name" in program
    assert "metadata" in program
    assert "instructions" in program
    assert isinstance(program["instructions"], list)
    assert len(program["instructions"]) == 5  # r, cz, barrier, measure, measure
    instr_names = [instr["name"] for instr in program["instructions"]]
    assert instr_names == ["prx", "cz", "barrier", "measure", "measure"]


def test_qiskit_to_iqm_json_prx_parameters(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that R gates are converted to PRX with correct parameters."""
    device = mock_qdmi_device_factory(num_qubits=1, operations=["r", "measure"])

    angle = np.pi / 2
    phase = np.pi / 4
    qc = QuantumCircuit(1, 1)
    qc.r(angle, phase, 0)
    qc.measure_all()

    json_str = qiskit_to_iqm_json(qc, device)  # type: ignore[arg-type]
    program = json.loads(json_str)

    prx_instr = program["instructions"][0]
    assert prx_instr["name"] == "prx"
    assert "args" in prx_instr
    assert "angle_t" in prx_instr["args"]
    assert "phase_t" in prx_instr["args"]

    expected_angle_t = angle / (2 * np.pi)
    expected_phase_t = phase / (2 * np.pi)
    assert abs(prx_instr["args"]["angle_t"] - expected_angle_t) < 1e-10
    assert abs(prx_instr["args"]["phase_t"] - expected_phase_t) < 1e-10


def test_qiskit_to_iqm_json_barrier(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that barriers are correctly converted."""
    device = mock_qdmi_device_factory(num_qubits=3, operations=["barrier"])

    qc = QuantumCircuit(3)
    qc.barrier([0, 1, 2])

    json_str = qiskit_to_iqm_json(qc, device)  # type: ignore[arg-type]
    program = json.loads(json_str)

    barrier_instr = program["instructions"][0]
    assert barrier_instr["name"] == "barrier"
    assert len(barrier_instr["qubits"]) == 3
    assert barrier_instr["args"] == {}


def test_qiskit_to_iqm_json_cz_gate(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that CZ gates are correctly converted."""
    device = mock_qdmi_device_factory(num_qubits=2, operations=["cz"])

    qc = QuantumCircuit(2)
    qc.cz(0, 1)

    json_str = qiskit_to_iqm_json(qc, device)  # type: ignore[arg-type]
    program = json.loads(json_str)

    cz_instr = program["instructions"][0]
    assert cz_instr["name"] == "cz"
    assert len(cz_instr["qubits"]) == 2
    assert cz_instr["args"] == {}


def test_qiskit_to_iqm_json_measure_keys(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that measurements generate correct keys."""
    device = mock_qdmi_device_factory(num_qubits=2, operations=["measure"])

    qc = QuantumCircuit(2, 2)
    qc.measure_all()

    json_str = qiskit_to_iqm_json(qc, device)  # type: ignore[arg-type]
    program = json.loads(json_str)

    barr = program["instructions"][0]
    meas0 = program["instructions"][1]
    meas1 = program["instructions"][2]

    assert barr["name"] == "barrier"
    assert barr["args"] == {}
    assert meas0["name"] == "measure"
    assert "key" in meas0["args"]
    assert meas1["name"] == "measure"
    assert "key" in meas1["args"]
    assert meas0["args"]["key"] != meas1["args"]["key"]


def test_qiskit_to_iqm_json_unsupported_operation(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that unsupported operations raise UnsupportedOperationError."""
    device = mock_qdmi_device_factory(num_qubits=1, operations=[])

    qc = QuantumCircuit(1)
    qc.h(0)

    with pytest.raises(UnsupportedOperationError, match="not supported in IQM JSON format"):
        qiskit_to_iqm_json(qc, device)  # type: ignore[arg-type]


def test_qiskit_to_iqm_json_circuit_name(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that circuit name is preserved in IQM JSON."""
    device = mock_qdmi_device_factory(num_qubits=1, operations=["measure"])

    qc = QuantumCircuit(1, 1, name="my_test_circuit")
    qc.measure_all()

    json_str = qiskit_to_iqm_json(qc, device)  # type: ignore[arg-type]
    program = json.loads(json_str)

    assert program["name"] == "my_test_circuit"
