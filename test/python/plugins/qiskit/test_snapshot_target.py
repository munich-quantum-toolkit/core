# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Qiskit projections of the shared immutable compiler target."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from qiskit.circuit import Gate, QuantumCircuit

from mqt.core.mlir import CompilerTarget, PayloadFormat, PayloadSpecification, ProgramCapability, TargetEnvironment
from mqt.core.plugins.qiskit import QDMIBackend
from mqt.core.plugins.qiskit.exceptions import UnsupportedDeviceError, UnsupportedOperationError
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.builtin_driver import open_device


def backend_for_snapshot(monkeypatch: pytest.MonkeyPatch, target: CompilerTarget) -> QDMIBackend:
    """Construct a backend from a prepared snapshot without another device query.

    Returns:
        A backend using the supplied target and OpenQASM payload contract.
    """
    environment = TargetEnvironment(target, PayloadSpecification.from_program_format(ProgramFormat.QASM3))
    monkeypatch.setattr(TargetEnvironment, "from_device", lambda _device: environment)
    device = SimpleNamespace(name=lambda: "snapshot", version=lambda: "1.0", operations=list)
    return QDMIBackend(device)  # ty: ignore[invalid-argument-type] Only identity and snapshot lookup are used.


def test_snapshot_properties_variants_and_opaque_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calibration, aliases, constraints and physical holes survive projection."""
    capability = CompilerTarget.OperationCapability
    site_tuple = CompilerTarget.SiteTuple
    target = CompilerTarget(
        [CompilerTarget.Site(0, t1=100), CompilerTarget.Site(2, t2=200), CompilerTarget.Site(5)],
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations([
            capability("prx", 1, 2, [[0]]),
            capability("r", 1, 2, [[2]]),
            capability("gphase", 0, 1),
            capability("measure", 1, 0, [[0], [2], [5]]),
            capability("mcx", CompilerTarget.OperationArity.variadic(2), 0),
            capability("rzz", 2, 1, [site_tuple([0, 2], duration=20, fidelity=0.99)], fixed_parameters=[0.5]),
            capability("rzz", 2, 1, [[2, 0]], fixed_parameters=[0.8]),
            capability("custom_native", 2, 1, [[0, 2]], parameter_bounds=[(-1.0, 1.0)]),
            capability("custom_native", 1, 1, [[0]], parameter_bounds=[(-1.0, 1.0)]),
        ]),
        duration_unit=CompilerTarget.DurationUnit("ns", 2.0),
    )
    backend = backend_for_snapshot(monkeypatch, target)
    assert "global_phase" in backend.target.operation_names
    assert backend.num_qubits == 3
    assert set(backend.target["r"]) == {(0,), (2,)}
    assert backend.target["rzz"][0, 2].duration == pytest.approx(40e-9)
    assert backend.target["rzz"][0, 2].error == pytest.approx(0.01)
    coherence = backend.target.qubit_properties
    assert coherence is not None
    assert coherence[0].t1 == pytest.approx(200e-9)
    assert coherence[2].t2 == pytest.approx(400e-9)
    assert backend.coupling_map is None
    circuit = QuantumCircuit(3)
    circuit.rzz(0.8, 2, 0)
    circuit.append(Gate("prx", 1, [0.5, 0.25]), [0])
    circuit.append(Gate("custom_native", 2, [0.2]), [0, 2])
    circuit.append(Gate("custom_native", 1, [0.2]), [0])
    backend.validate_circuit(circuit, native=True)
    circuit.append(Gate("custom_native", 2, [2.0]), [0, 2])
    with pytest.raises(UnsupportedOperationError, match="custom_native"):
        backend.validate_circuit(circuit, native=True)
    invalid = QuantumCircuit(3)
    invalid.append(Gate("mcx", 3, []), [0, 1, 2])
    with pytest.raises(UnsupportedOperationError, match="mcx"):
        backend.validate_circuit(invalid, native=True)


def test_snapshot_star_target_requires_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resonators never become Qiskit placement sites or native two-qubit edges."""
    capability = CompilerTarget.OperationCapability
    target = CompilerTarget(
        3,
        connectivity=CompilerTarget.Connectivity([(0, 1), (2, 1)]),
        native_operations=CompilerTarget.NativeOperations([
            capability("r", 1, 2, [[0], [2]]),
            capability("measure", 1, 0, [[0], [2]]),
            capability("move", 2, 0, [[0, 1], [2, 1]]),
            capability("cz", 2, 0, [[0, 1], [2, 1]]),
        ]),
    )
    backend = backend_for_snapshot(monkeypatch, target)
    assert {"r", "measure", "barrier"} <= backend.target.operation_names
    assert not {"cz", "move"} & backend.target.operation_names
    circuit = QuantumCircuit(3)
    circuit.cz(0, 2)
    with pytest.raises(UnsupportedOperationError, match=r"backend\.compile"):
        backend.validate_circuit(circuit, native=True)


def test_snapshot_global_operations_stay_global(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wide all-to-all devices retain global support without enumerating pairs."""
    capability = CompilerTarget.OperationCapability
    target = CompilerTarget(
        65535,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations([
            capability("x", CompilerTarget.OperationArity.variadic(1), 0),
            capability("measure", 1, 0),
        ]),
    )
    backend = backend_for_snapshot(monkeypatch, target)
    assert backend.target["cx"] == {None: backend.target["cx"][None]}
    assert {"x", "cx", "ccx", "mcx"} <= backend.target.operation_names
    assert backend.coupling_map is None
    assert backend.num_qubits == 65535


def test_snapshot_without_placeable_sites_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measurement alone does not make a site computational."""
    target = CompilerTarget(
        1,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations([CompilerTarget.OperationCapability("measure", 1, 0)]),
    )
    with pytest.raises(UnsupportedDeviceError, match="no sites"):
        backend_for_snapshot(monkeypatch, target)


def test_snapshot_control_flow_uses_payload_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    """A payload's forward branching enables if_else without loop support."""
    target = CompilerTarget(
        1,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations([
            CompilerTarget.OperationCapability("x", 1, 0),
            CompilerTarget.OperationCapability("measure", 1, 0),
        ]),
    )
    environment = TargetEnvironment(
        target,
        PayloadSpecification(
            PayloadFormat("openqasm", version="3.0"),
            [ProgramCapability(ProgramCapability.FORWARD_BRANCHING)],
            optional_capabilities_known=True,
        ),
    )
    monkeypatch.setattr(TargetEnvironment, "from_device", lambda _device: environment)
    device = SimpleNamespace(name=lambda: "snapshot", version=lambda: "1.0")
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type] Prepared snapshot fixture.
    assert "if_else" in backend.target.operation_names
    assert "store" not in backend.target.operation_names
    assert not {"while_loop", "for_loop", "switch_case"} & backend.target.operation_names


def test_snapshot_fixed_mcx_keeps_its_arity(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fixed native MCX does not advertise arbitrary control counts."""
    capability = CompilerTarget.OperationCapability
    target = CompilerTarget(
        4,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations([
            capability("x", 1, 0),
            capability("measure", 1, 0),
            capability("mcx", 3, 0),
        ]),
    )
    backend = backend_for_snapshot(monkeypatch, target)
    circuit = QuantumCircuit(4)
    circuit.append(Gate("mcx", 3, []), [0, 1, 2])
    backend.validate_circuit(circuit, native=True)
    circuit.append(Gate("mcx", 4, []), [0, 1, 2, 3])
    with pytest.raises(UnsupportedOperationError, match="mcx"):
        backend.validate_circuit(circuit, native=True)


def test_device_snapshot_retains_opaque_native_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real QDMI descriptors preserve provider gates and exact ordered placements."""
    signatures = [("gpi", 1, [0]), ("gpi2", 1, [1]), ("ms", 3, [0, 1]), ("xy", 1, [1, 0]), ("custom_native", 0, [0])]
    config = {
        "schema-version": 1,
        "name": "Opaque native gates",
        "numQubits": 2,
        "durationUnit": {"unit": "ns", "scaleFactor": 1},
        "qubitProperties": {"defaults": {}, "overrides": []},
        "couplings": [[0, 1], [1, 0]],
        "operations": [
            {"name": name, "numQubits": len(sites), "numParameters": parameters, "sites": [sites]}
            for name, parameters, sites in signatures
        ]
        + [{"name": "measure", "numQubits": 1, "numParameters": 0}],
    }
    device = open_device("mqt.sc.default", device_config=json.dumps(config))
    target = CompilerTarget.from_device(device)
    backend = backend_for_snapshot(monkeypatch, target)
    circuit = QuantumCircuit(2)
    for name, parameters, sites in signatures:
        assert name in backend.target.operation_names
        assert set(backend.target[name]) == {tuple(sites)}
        circuit.append(Gate(name, len(sites), [0.25] * parameters), sites)
    backend.validate_circuit(circuit, native=True)
    circuit.append(Gate("ms", 2, [0.25] * 3), [1, 0])
    with pytest.raises(UnsupportedOperationError, match="ms"):
        backend.validate_circuit(circuit, native=True)
