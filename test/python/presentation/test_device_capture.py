# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check the boundaries between reported hardware and local compiler models."""

from __future__ import annotations

import json
import runpy
from math import pi
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator

from mqt.core.mlir import CompilerTarget, PayloadFormat, PayloadSpecification, QCProgram, TargetEnvironment

SCRIPT = Path(__file__).resolve().parents[3] / "presentations/mqsf2026/capture_devices.py"


def test_emerald_disables_stale_calibration_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reported calibration must not add an edge absent from current topology."""
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    capture = runpy.run_path(str(SCRIPT))["braket_target"]
    capabilities = {
        "paradigm": {
            "qubitCount": 3,
            "nativeGateSet": ["prx", "cz"],
            "connectivity": {"connectivityGraph": {"1": ["2"], "2": ["1"], "3": []}},
        },
        "provider": {
            "properties": {
                "one_qubit": {"1": {"f1Q_simultaneous_RB": 0.99, "fRO": 0.98}},
                "two_qubit": {"1-2": {"fCZ": 0.97}, "2-3": {"fCZ": 0.96}},
            }
        },
        "service": {"updatedAt": "2026-10-09"},
    }
    raw = {
        "deviceCapabilities": json.dumps(capabilities),
        "deviceName": "Emerald",
        "deviceArn": "public-device-id",
        "deviceStatus": "ONLINE",
    }
    target = capture(raw)
    assert target["metadata"]["edges"] == [(0, 1)]
    assert target["metadata"]["sites"][0] == {"id": 0, "name": "QB1", "provider_id": 1}
    assert target["compiler_model"]["operations"][1]["siteOverrides"] == [{"sites": [0, 1], "fidelity": 0.97}]
    assert target["provenance"]["raw"] == raw


def test_metrics_follow_wire_dependencies_and_reject_control_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Independent single-qubit gates occupy one layer; nested loops need a different metric."""
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    metrics = runpy.run_path(str(SCRIPT))["circuit_metrics"]
    circuit: dict[str, Any] = {
        "qubits": [{"id": 0}, {"id": 1}],
        "operations": [{"name": "h", "qubits": [0]}, {"name": "h", "qubits": [1]}, {"name": "cx", "qubits": [0, 1]}],
    }
    assert metrics(circuit) == {
        "operations": 3,
        "counts": {"h": 2, "cx": 1},
        "depth": 2,
        "active_qubits": 2,
        "two_qubit_operations": 1,
    }
    with pytest.raises(ValueError, match="straight-line"):
        metrics({
            **circuit,
            "operations": [{"name": "for_loop", "qubits": [0], "blocks": [[{"name": "x", "qubits": [0]}]]}],
        })


def test_ionq_native_export_preserves_phase_and_gate_matrices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native definitions use radians and retain GPi's phase relative to R(pi, phi)."""
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    module = runpy.run_path(str(SCRIPT))
    export = module["ionq_openqasm3"]
    circuit = QuantumCircuit(2, global_phase=0.37)
    for name, angle, phi, qubit in [("gpi", pi, -7.0, 0), ("gpi2", pi / 2, 13.31, 1)]:
        gate = Gate(name, 1, [phi])
        gate.definition = QuantumCircuit(1, global_phase=pi / 2 if name == "gpi" else 0)
        gate.definition.r(angle, phi, 0)
        circuit.append(gate, [qubit])
    circuit.rzz(0.43, 0, 1)
    native = module["normalize_ionq_phases"](circuit)
    assert all(-pi <= float(op.operation.params[0]) <= pi for op in native.data)
    assert np.allclose(Operator(native).data, Operator(circuit).data)
    source = export(native)
    restored = QCProgram.from_openqasm_str(source).to_qiskit()
    assert np.allclose(Operator(restored).data, Operator(circuit).data)
    assert "gpi(" in source
    assert "gpi2(" in source


def test_ionq_derived_target_synthesizes_only_native_pulses(monkeypatch: pytest.MonkeyPatch) -> None:
    """The recorded QDMI topology is retained while the compiler enforces pulse bounds."""
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    derive = runpy.run_path(str(SCRIPT))["ionq_compiler_target"]
    snapshot = CompilerTarget(
        2,
        connectivity=CompilerTarget.Connectivity([(0, 1)]),
        native_operations=CompilerTarget.NativeOperations([
            CompilerTarget.OperationCapability(name, arity, 1) for name, arity in [("gpi", 1), ("gpi2", 1), ("rzz", 2)]
        ]),
    )
    target = derive(snapshot)
    assert target.couplings == snapshot.couplings
    assert target.synthesis_basis.single_qubit == CompilerTarget.SingleQubitBasis.R_FIXED
    source = QuantumCircuit(2)
    source.u(0.37, 0.42, -0.31, 0)
    source.cx(0, 1)
    source.rx(-0.73, 1)
    program = QCProgram.from_qiskit(source).to_qco()
    program.compile_for_target(TargetEnvironment(target, PayloadSpecification(PayloadFormat("openqasm", "3.0"))))
    native = program.to_qiskit(target=target)
    assert set(native.count_ops()) == {"gpi", "gpi2", "rzz"}
    assert all(0 <= float(op.operation.params[0]) <= pi / 2 for op in native.data if op.operation.name == "rzz")
    assert Operator(native).equiv(Operator(source))
