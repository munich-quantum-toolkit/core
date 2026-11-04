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
from qiskit import transpile
from qiskit.circuit import Gate, QuantumCircuit

from mqt.core.mlir import (
    CompilerTarget,
    PayloadFormat,
    PayloadSpecification,
    ProgramCapability,
    TargetEnvironment,
    serialize_program,
)
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


@pytest.mark.parametrize("declared", [False, True])
def test_snapshot_control_flow_uses_payload_capabilities(monkeypatch: pytest.MonkeyPatch, *, declared: bool) -> None:
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
            [ProgramCapability(ProgramCapability.FORWARD_BRANCHING)] if declared else [],
            optional_capabilities_known=True,
        ),
    )
    monkeypatch.setattr(TargetEnvironment, "from_device", lambda _device: environment)
    device = SimpleNamespace(name=lambda: "snapshot", version=lambda: "1.0")
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type] Prepared snapshot fixture.
    assert ("if_else" in backend.target.operation_names) is declared
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
        + [
            {"name": "measure", "numQubits": 1, "numParameters": 0},
            {"name": "x", "numQubits": 1, "numParameters": 0},
        ],
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
    program = serialize_program(
        circuit, target=TargetEnvironment(target, PayloadSpecification.from_program_format(ProgramFormat.QASM3))
    )
    assert isinstance(program.payload, str)
    assert "gpi(0.25) $0;" in program.payload
    assert "ms(0.25, 0.25, 0.25) $0, $1;" in program.payload
    assert "xy(0.25) $1, $0;" in program.payload
    assert "custom_native $0;" in program.payload
    circuit.append(Gate("ms", 2, [0.25] * 3), [1, 0])
    with pytest.raises(UnsupportedOperationError, match="ms"):
        backend.validate_circuit(circuit, native=True)


def test_snapshot_variant_names_preserve_native_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixed-parameter variants cannot shadow a device's real native gate."""
    capability = CompilerTarget.OperationCapability
    target = CompilerTarget(
        2,
        connectivity=CompilerTarget.Connectivity([(0, 1)]),
        native_operations=CompilerTarget.NativeOperations([
            capability("x", 1, 0),
            capability("measure", 1, 0),
            capability("rx", 1, 1, [[0]], fixed_parameters=[0.5]),
            capability("rx", 1, 1, [[1]], fixed_parameters=[0.8]),
            capability("rx_3", 1, 0, [[0]]),
            capability("cx", 2, 0, [[0, 1]]),
            capability("barrier", CompilerTarget.OperationArity.variadic(1), 0),
            capability("h", 2, 0, [[0, 1]]),
            capability("zz", 2, 0, [[0, 1]]),
        ]),
    )
    backend = backend_for_snapshot(monkeypatch, target)
    assert backend.target.operation_from_name("rx_3").params == []
    assert backend.target.operation_from_name("h").num_qubits == 2
    assert backend.target.operation_from_name("rzz").params == []
    assert "barrier" in backend.target.operation_names
    circuit = QuantumCircuit(2)
    circuit.rx(0.5, 0)
    circuit.rx(0.8, 1)
    circuit.append(Gate("rx_3", 1, []), [0])
    circuit.append(Gate("zz", 2, []), [0, 1])
    backend.validate_circuit(circuit, native=True)
    program = serialize_program(
        circuit, target=TargetEnvironment(target, PayloadSpecification.from_program_format(ProgramFormat.QASM3))
    )
    assert isinstance(program.payload, str)
    assert "zz $0, $1;" in program.payload
    coupling = backend.coupling_map
    assert coupling is not None
    assert set(coupling.get_edges()) == {(0, 1)}


def test_snapshot_unknown_variadic_gate_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An opaque signature cannot advertise an unbounded Qiskit gate arity."""
    capability = CompilerTarget.OperationCapability
    target = CompilerTarget(
        2,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations([
            capability("x", 1, 0),
            capability("measure", 1, 0),
            capability("custom", CompilerTarget.OperationArity.variadic(1), 0),
        ]),
    )
    with pytest.raises(UnsupportedOperationError, match="variadic native operation 'custom'"):
        backend_for_snapshot(monkeypatch, target)


def test_snapshot_timing_requires_physical_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calibration values cannot be presented as seconds without physical units."""
    capability = CompilerTarget.OperationCapability
    target = CompilerTarget(
        1,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations([
            capability("x", 1, 0, [CompilerTarget.SiteTuple([0], duration=20)]),
            capability("measure", 1, 0),
        ]),
        duration_unit=CompilerTarget.DurationUnit("dt", 1.0),
    )
    with pytest.raises(UnsupportedOperationError, match="supported duration unit"):
        backend_for_snapshot(monkeypatch, target)


def test_braket_aliases_global_duration_and_native_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native Braket spellings share compiler meaning and global calibration."""
    capability = CompilerTarget.OperationCapability
    target = CompilerTarget(
        2,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations([
            capability("cnot", 2, 0),
            capability("v", 1, 0, duration=12),
            capability("phaseshift", 1, 1),
            capability("measure", 1, 0),
        ]),
        duration_unit=CompilerTarget.DurationUnit("ns", 1.0),
    )
    backend = backend_for_snapshot(monkeypatch, target)
    assert {"cx", "sx", "p"} <= backend.target.operation_names
    assert backend.target["sx"][None].duration == pytest.approx(12e-9)
    circuit = QuantumCircuit(2)
    circuit.sx(0)
    circuit.p(0.3, 0)
    circuit.cx(0, 1)
    backend.validate_circuit(circuit, native=True)


def test_ddsim_target_names_and_native_mcphase() -> None:
    """The complete target vocabulary preserves Qiskit's variadic phase gate."""
    backend = QDMIBackend(open_device("mqt.ddsim.default"))
    assert set(backend.target.operation_names) == {
        "barrier",
        "break_loop",
        "c3sx",
        "ccx",
        "ccz",
        "ch",
        "continue_loop",
        "cp",
        "crx",
        "cry",
        "crz",
        "cs",
        "csdg",
        "cswap",
        "csx",
        "cu1",
        "cu3",
        "cx",
        "cy",
        "cz",
        "dcx",
        "ecr",
        "for_loop",
        "global_phase",
        "h",
        "id",
        "if_else",
        "iswap",
        "mcphase",
        "mcx",
        "measure",
        "p",
        "r",
        "rccx",
        "reset",
        "rx",
        "rxx",
        "ry",
        "ryy",
        "rz",
        "rzx",
        "rzz",
        "s",
        "sdg",
        "store",
        "swap",
        "switch_case",
        "sx",
        "sxdg",
        "t",
        "tdg",
        "u",
        "u2",
        "while_loop",
        "x",
        "xx_minus_yy",
        "xx_plus_yy",
        "y",
        "z",
    }
    circuit = QuantumCircuit(4)
    circuit.mcp(0.3, [0, 1, 2], 3)
    assert transpile(circuit, backend, optimization_level=0).count_ops() == {"mcphase": 1}


def test_sparse_global_gates_do_not_enumerate_placements(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing computational site does not expand global three-qubit support."""
    capability = CompilerTarget.OperationCapability
    target = CompilerTarget(
        127,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations([
            capability("x", 1, 0),
            capability("measure", 1, 0, [[site] for site in range(126)]),
            capability("ccx", 3, 0),
        ]),
    )
    backend = backend_for_snapshot(monkeypatch, target)
    assert set(backend.target["ccx"]) == {None}
    circuit = QuantumCircuit(127)
    circuit.ccx(0, 1, 126)
    with pytest.raises(UnsupportedOperationError, match="not advertised"):
        backend.validate_circuit(circuit, native=True)


def test_parameter_variant_names_are_local_and_keep_unrestricted_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding unrelated calibrated gates does not rename an RX parameter variant."""
    capability = CompilerTarget.OperationCapability
    names = []
    for unrelated in [[], [capability("p", 1, 1, fixed_parameters=[0.2])]]:
        target = CompilerTarget(
            1,
            connectivity=CompilerTarget.Connectivity.all_to_all(),
            native_operations=CompilerTarget.NativeOperations([
                *unrelated,
                capability("rx", 1, 1, fixed_parameters=[0.5]),
                capability("rx", 1, 1),
                capability("measure", 1, 0),
            ]),
        )
        backend = backend_for_snapshot(monkeypatch, target)
        names.append({name for name in backend.target.operation_names if name.startswith("rx")})
        assert backend.target.instruction_supported("rx", (0,), parameters=[0.8])
        assert backend.target.operation_from_name("rx_1").params == [0.5]
    assert names == [{"rx", "rx_1"}, {"rx", "rx_1"}]
