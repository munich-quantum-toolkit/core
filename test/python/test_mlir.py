# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the MLIR compiler Python bindings."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest
import qiskit
from packaging import version
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit import Gate, library
from qiskit.quantum_info import Operator

from mqt.core.ir import QuantumComputation
from mqt.core.mlir import (
    CompilerTarget,
    JeffProgram,
    OpenQASMProgram,
    OutputFormat,
    QCOProgram,
    QCProgram,
    QIRProfile,
    QIRProgram,
    compile_program,
)
from mqt.core.qdmi.driver import open_device

requires_qiskit_translation = pytest.mark.skipif(
    not (
        version.parse("2.5") <= version.parse(qiskit.__version__) < version.parse("2.6")
        or qiskit.__version__ == os.environ.get("MQT_QISKIT_TEST_CANDIDATE_VERSION")
    ),
    reason=f"no Qiskit translation is registered for {qiskit.__version__}",
)

MLIR_STRING = r"""module {
  func.func @main() -> memref<2xi1> attributes {mqt.entry_point} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<2x!qc.qubit>
    %0 = memref.load %alloc[%c0] : memref<2x!qc.qubit>
    qc.h %0 : !qc.qubit
    %1 = memref.load %alloc[%c1] : memref<2x!qc.qubit>
    qc.ctrl(%0) targets (%arg0 = %1) {
      qc.x %arg0 : !qc.qubit
      qc.yield
    } : {!qc.qubit}, {!qc.qubit}
    %alloc_0 = memref.alloc() : memref<2xi1>
    %2 = qc.measure %0 : !qc.qubit -> i1
    memref.store %2, %alloc_0[%c0] : memref<2xi1>
    %3 = qc.measure %1 : !qc.qubit -> i1
    memref.store %3, %alloc_0[%c1] : memref<2xi1>
    memref.dealloc %alloc : memref<2x!qc.qubit>
    return %alloc_0 : memref<2xi1>
  }
}
"""

QASM_STRING = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
bit[2] c = measure q;
"""


def _assert_bell_program(program: QCProgram, *, measured: bool = False) -> None:
    """Check the semantics of a translated Bell-state program."""
    assert program.is_valid
    ir = program.ir
    assert "memref<2x!qc.qubit>" in ir
    assert ir.count("qc.h ") == 1
    assert ir.count("qc.ctrl(") == 1
    assert ir.count("qc.x ") == 1

    if not measured:
        assert "func.func @main() -> i64" in ir
        assert "qc.measure" not in ir
        return

    assert "func.func @main() -> !cbit.reg<2>" in ir or "func.func @main() -> (!cbit.reg<2>" in ir
    assert "cbit.alloc" in ir
    assert ir.count("cbit.store") == 2
    assert ir.count("qc.measure") == 2


def test_compile_program_jeff_file() -> None:
    """Compile a ``.jeff`` file."""
    path = Path(__file__).parent.parent / "circuits" / "bell.jeff"

    result = compile_program(path)
    assert isinstance(result, QCProgram)
    _assert_bell_program(result)


def test_compile_program_mlir_string() -> None:
    """Compile an MLIR string."""
    result = compile_program(MLIR_STRING)
    assert isinstance(result, QCProgram)
    assert result.ir == MLIR_STRING


def test_compile_program_mlir_string_with_leading_whitespace() -> None:
    """Compile a whitespace-prefixed single-line MLIR string."""
    source = " module { %0 = qc.alloc : !qc.qubit qc.dealloc %0 : !qc.qubit }"

    result = compile_program(source)

    assert isinstance(result, QCProgram)
    assert result.ir.startswith("module")


def test_compile_program_mlir_file(tmp_path: Path) -> None:
    """Compile a ``.mlir`` file."""
    path = tmp_path / "program.mlir"
    path.write_text(MLIR_STRING, encoding="utf-8")

    result = compile_program(path)
    assert isinstance(result, QCProgram)
    assert result.ir == MLIR_STRING


def test_compile_program_mlir_file_named_module(tmp_path: Path) -> None:
    """Compile an MLIR file whose name begins with ``module``."""
    path = tmp_path / "module.mlir"
    path.write_text(MLIR_STRING, encoding="utf-8")

    result = compile_program(path)

    assert isinstance(result, QCProgram)
    assert result.ir == MLIR_STRING


def test_compile_program_rejects_unsupported_file(tmp_path: Path) -> None:
    """Reject an existing file with an unsupported extension."""
    path = tmp_path / "program.txt"
    path.write_text(MLIR_STRING, encoding="utf-8")

    with pytest.raises(RuntimeError, match="unsupported extension"):
        compile_program(path)


def test_compile_program_qasm_string() -> None:
    """Compile an OpenQASM string."""
    result = compile_program(QASM_STRING)
    assert isinstance(result, QCProgram)
    _assert_bell_program(result, measured=True)


def test_compile_program_single_line_qasm_string() -> None:
    """Compile a single-line OpenQASM source string."""
    result = compile_program(QASM_STRING.replace("\n", " "))

    assert isinstance(result, QCProgram)
    assert "qc.h" in result.ir


def test_compile_program_qasm_file(tmp_path: Path) -> None:
    """Compile a ``.qasm`` file."""
    path = tmp_path / "program.qasm"
    path.write_text(QASM_STRING, encoding="utf-8")

    result = compile_program(path)
    assert isinstance(result, QCProgram)
    _assert_bell_program(result, measured=True)


def test_compile_program_rejects_quantum_computation() -> None:
    """Reject the removed legacy compiler input."""
    with pytest.raises(RuntimeError, match="is not supported"):
        compile_program(QuantumComputation(1))  # ty: ignore[invalid-argument-type]


@requires_qiskit_translation
def test_compile_program_qiskit_quantum_circuit() -> None:
    """Compile a ``QuantumCircuit``."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(range(2), range(2))

    result = compile_program(qc)
    assert isinstance(result, QCProgram)
    _assert_bell_program(result, measured=True)


@requires_qiskit_translation
def test_compile_program_qiskit_quantum_circuit_subclass() -> None:
    """Compile a user-defined Qiskit ``QuantumCircuit`` subclass."""

    class CustomQuantumCircuit(QuantumCircuit):
        """A user-defined circuit type."""

    qc = CustomQuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(range(2), range(2))

    result = compile_program(qc)

    assert isinstance(result, QCProgram)
    _assert_bell_program(result, measured=True)


def test_jeff_program_round_trip(tmp_path: Path) -> None:
    """Store and load a ``JeffProgram`` through bytes and a file."""
    path = tmp_path / "program.jeff"
    result = compile_program(QASM_STRING, output=OutputFormat.JEFF)
    assert isinstance(result, JeffProgram)

    path.write_bytes(result.to_bytes())
    loaded = JeffProgram.from_file(path)
    restored = compile_program(loaded, output=OutputFormat.QC)
    assert isinstance(restored, QCProgram)
    _assert_bell_program(restored, measured=True)


def test_compile_program_jeff_input_runs_from_qco(tmp_path: Path) -> None:
    """Compile a serialized Jeff program through the QCO pipeline entry point."""
    path = tmp_path / "program.jeff"
    compile_program(QASM_STRING, output=OutputFormat.JEFF).write(path)

    result = compile_program(path, output=OutputFormat.QCO)

    assert isinstance(result, QCOProgram)
    assert "qco." in result.ir


def test_program_conversions_are_composable() -> None:
    """Compose frontend, cleanup, conversion, and optimization stages."""
    source = QCProgram.from_qasm_str(QASM_STRING)
    qco = source.to_qco(copy=True)
    assert source.is_valid
    assert isinstance(qco, QCOProgram)

    qco.cleanup()
    qco.merge_single_qubit_rotation_gates()
    result = qco.to_qc()
    assert not qco.is_valid
    result.cleanup()
    _assert_bell_program(result, measured=True)


def test_openqasm_program_direct_and_pipeline_output(tmp_path: Path) -> None:
    """Emit OpenQASM directly from QC and through the optimized pipeline."""
    source = QCProgram.from_qasm_str(QASM_STRING)
    direct = source.to_openqasm3()

    assert isinstance(direct, OpenQASMProgram)
    assert source.is_valid
    assert direct.source.startswith("OPENQASM 3.1;")
    assert str(direct) == direct.source

    path = tmp_path / "program.qasm"
    direct.write(path)
    assert path.read_text(encoding="utf-8") == direct.source
    _assert_bell_program(QCProgram.from_qasm_file(path), measured=True)

    optimized = compile_program(QASM_STRING, output=OutputFormat.OPENQASM3)
    assert isinstance(optimized, OpenQASMProgram)
    assert "output bit[2] c;" in optimized.source
    _assert_bell_program(QCProgram.from_qasm_str(optimized.source), measured=True)

    imported = compile_program(direct, output=OutputFormat.QC_IMPORT)
    assert isinstance(imported, QCProgram)
    _assert_bell_program(imported, measured=True)

    compiled = compile_program(direct, output=OutputFormat.QIR_ADAPTIVE)
    assert isinstance(compiled, QIRProgram)


@pytest.mark.parametrize(
    "gate",
    [
        library.SXdgGate(),
        library.RGate(0.1, 0.2),
        library.U2Gate(0.2, 0.3),
        library.UGate(0.1, 0.2, 0.3),
        library.iSwapGate(),
        library.DCXGate(),
        library.ECRGate(),
        library.RXXGate(0.1),
        library.RYYGate(0.2),
        library.RZXGate(0.3),
        library.RZZGate(0.4),
        library.XXPlusYYGate(0.5, 0.6),
        library.XXMinusYYGate(0.7, 0.8),
        library.RCCXGate(),
    ],
)
@requires_qiskit_translation
def test_openqasm_helper_gate_matrix(gate: Gate) -> None:
    """Preserve complete helper-gate matrices, including global phase."""
    circuit = QuantumCircuit(gate.num_qubits)
    circuit.append(gate, range(gate.num_qubits))

    source = QCProgram.from_qiskit(circuit).to_openqasm3().source
    round_tripped = qasm3.loads(source)

    assert np.allclose(Operator(round_tripped).data, Operator(circuit).data)


def test_compile_program_convert_to_qir() -> None:
    """Compile with the QIR Base Profile output format."""
    result = compile_program(QASM_STRING, output=OutputFormat.QIR_BASE)

    assert isinstance(result, QIRProgram)
    assert "; ModuleID" in result.llvm_ir
    assert "@__quantum__qis__h__body" in result.llvm_ir
    bitcode = result.to_bitcode()
    assert bitcode.startswith(b"BC\xc0\xde")


def test_qir_program_writes_bitcode(tmp_path: Path) -> None:
    """Write generated LLVM bitcode to a file."""
    result = compile_program(QASM_STRING, output=OutputFormat.QIR_BASE)
    path = tmp_path / "program.bc"

    result.write_bitcode(path)

    assert path.read_bytes() == result.to_bitcode()


def test_compile_program_output_format_convert_to_qir() -> None:
    """Lower a QC program directly to the QIR Adaptive Profile."""
    result = QCProgram.from_qasm_str(QASM_STRING).to_qir(QIRProfile.ADAPTIVE)

    assert isinstance(result, QIRProgram)
    assert result.profile == QIRProfile.ADAPTIVE
    assert "@__quantum__qis__h__body" in result.llvm_ir


def test_compile_program_qc_import_output() -> None:
    """Expose QC directly after the frontend translation."""
    result = compile_program(QASM_STRING, output=OutputFormat.QC_IMPORT)

    assert isinstance(result, QCProgram)
    assert "qc.h" in result.ir


def test_compile_program_exposes_raw_and_optimized_qco() -> None:
    """Expose QCO before and after the configured optimization pipeline."""
    qasm = QASM_STRING.replace("h q[0];", "rz(1.0) q[0];\nrx(1.0) q[0];")

    raw = compile_program(qasm, output=OutputFormat.QCO)
    optimized = compile_program(qasm, output=OutputFormat.QCO_OPTIMIZED)

    assert isinstance(raw, QCOProgram)
    assert isinstance(optimized, QCOProgram)
    assert raw.ir != optimized.ir


@pytest.fixture(scope="module")
def garnet_target() -> CompilerTarget:
    """Snapshot the bundled IQM Garnet device.

    Returns:
        The detached compiler target.
    """
    return CompilerTarget.from_device_id("mqt.sc.iqm.garnet")


def test_compile_program_for_qdmi_target(garnet_target: CompilerTarget) -> None:
    """Compile through the canonical target pipeline for a QDMI device."""
    result = compile_program(
        QASM_STRING,
        output=OutputFormat.QCO_OPTIMIZED,
        target=garnet_target,
    )

    assert isinstance(result, QCOProgram)
    static_sites = {int(site) for site in re.findall(r"qco\.static (\d+)", result.ir)}
    assert len(static_sites) == 2
    assert static_sites <= {site.id for site in garnet_target.sites}
    assert "qco.r(" in result.ir
    assert "qco.ctrl" in result.ir
    assert "qco.z " in result.ir
    assert result.ir.count("qco.measure") == 2
    assert "qco.rx" not in result.ir
    assert "qco.ry" not in result.ir


def test_qco_program_compiles_for_direct_sparse_target() -> None:
    """Expose direct target construction and typed QCO compilation."""
    target = CompilerTarget(
        "sparse target",
        [CompilerTarget.Site(10), CompilerTarget.Site(20)],
        couplings=[(10, 20)],
        operations=[
            CompilerTarget.Operation("u", 1, 3),
            CompilerTarget.Operation("cz", 2, 0),
            CompilerTarget.Operation("measure", 1, 0),
        ],
    )
    assert target.name == "sparse target"
    assert [site.id for site in target.sites] == [10, 20]
    assert target.couplings == [(10, 20)]
    assert target.synthesis_basis is not None
    assert target.synthesis_basis.single_qubit == CompilerTarget.SingleQubitBasis.U
    assert target.synthesis_basis.entangler == CompilerTarget.GateKind.CZ

    qco = compile_program(QASM_STRING, output=OutputFormat.QCO)
    assert isinstance(qco, QCOProgram)

    qco.compile_for_target(target)

    assert {int(site) for site in re.findall(r"qco\.static (\d+)", qco.ir)} == {10, 20}
    assert "qco.u(" in qco.ir
    assert "qco.ctrl" in qco.ir
    assert "qco.z " in qco.ir
    assert qco.ir.count("qco.measure") == 2


@requires_qiskit_translation
def test_target_compilation_exports_canonical_physical_qiskit_circuit() -> None:
    """Export a mapped program with the complete compiler target."""
    target = CompilerTarget(5)
    mapped = compile_program(
        QASM_STRING,
        output=OutputFormat.QCO_OPTIMIZED,
        target=target,
    )
    assert isinstance(mapped, QCOProgram)
    assert 0 < mapped.ir.count("qco.static") < target.num_qubits

    qc = mapped.to_qc(copy=True)
    restored = qc.to_qiskit(target=target)

    assert mapped.is_valid
    assert restored.num_qubits == 5
    assert [(register.name, len(register)) for register in restored.qregs] == [("q", 5)]
    assert restored.layout is None


def test_compiler_target_constructors_preserve_python_api() -> None:
    """Construct every target metadata type and target overload."""
    duration_unit = CompilerTarget.DurationUnit("ns", 1.0)
    sites = [
        CompilerTarget.Site(10, "q0", 100, 200),
        CompilerTarget.Site(20, "q1"),
    ]
    site_tuple = CompilerTarget.SiteTuple([10, 20], duration=10, fidelity=0.99)
    operation = CompilerTarget.Operation("cx", 2, 0, site_tuples=[site_tuple], duration=20, fidelity=0.98)

    targets = [
        CompilerTarget(2, duration_unit=duration_unit),
        CompilerTarget("dense", 2, duration_unit=duration_unit),
        CompilerTarget(sites, operations=[operation], duration_unit=duration_unit),
        CompilerTarget("sparse", sites, operations=[operation], duration_unit=duration_unit),
    ]

    assert [target.num_qubits for target in targets] == [2, 2, 2, 2]
    assert targets[1].name == "dense"
    assert targets[3].name == "sparse"
    assert sites[0].name == "q0"
    assert sites[0].t1 == 100
    assert sites[0].t2 == 200
    assert site_tuple.sites == [10, 20]
    assert len(operation.site_tuples) == 1
    assert operation.site_tuples[0].sites == [10, 20]
    assert duration_unit.unit == "ns"


def test_compiler_target_classical_control_is_explicit_and_canonical() -> None:
    """Keep runtime classical control opt-in and queryable."""
    capability = CompilerTarget.ClassicalControl
    target = CompilerTarget(
        2,
        classical_control=[
            capability.MULTIWAY_BRANCH,
            capability.CONDITIONAL,
            capability.MULTIWAY_BRANCH,
        ],
    )

    assert target.classical_control == [capability.CONDITIONAL, capability.MULTIWAY_BRANCH]
    assert target.supports_classical_control(capability.CONDITIONAL)
    assert not target.supports_classical_control(capability.ITERATION)
    assert CompilerTarget(2).classical_control == []


def test_compiler_target_construction_preserves_validation_errors() -> None:
    """Translate explicit C++ construction errors to Python ``ValueError``."""
    for _ in range(2):
        with pytest.raises(ValueError, match="must contain at least one site"):
            CompilerTarget(0)
    with pytest.raises(ValueError, match="site ID must be nonnegative"):
        CompilerTarget.Site(-1)
    with pytest.raises(ValueError, match="contains a duplicate site"):
        CompilerTarget.SiteTuple([0, 0])
    with pytest.raises(ValueError, match="duration unit must not be empty"):
        CompilerTarget.DurationUnit("", 1.0)
    with pytest.raises(ValueError, match="operation qubit count must be positive"):
        CompilerTarget.Operation("x", 0, 0)


def test_compiler_target_snapshots_qdmi_device(garnet_target: CompilerTarget) -> None:
    """Retain IQM topology and calibration independently of the live device."""
    target = garnet_target

    assert target.name == "IQM Garnet"
    assert target.num_qubits == 20
    assert len(target.couplings) == 30
    assert target.sites[0].name == "QB1"
    assert target.sites[0].t1 == 26626
    assert target.sites[0].t2 == 8376
    assert target.duration_unit is not None
    assert target.duration_unit.unit == "us"
    assert target.duration_unit.scale_factor == pytest.approx(0.001)
    assert target.supports_operation("r", 1, 2)
    assert target.supports_operation("cz", 2, 0)
    assert target.supports_operation("measure", 1, 0)
    assert not target.supports_operation("rx", 1, 1)
    assert target.synthesis_basis is not None
    assert target.synthesis_basis.single_qubit == CompilerTarget.SingleQubitBasis.R
    assert target.synthesis_basis.entangler == CompilerTarget.GateKind.CZ
    assert [operation.name for operation in target.operations] == ["r", "cz", "measure"]
    assert [len(operation.site_tuples) for operation in target.operations] == [20, 30, 20]
    assert all(
        site_tuple.fidelity is not None for operation in target.operations for site_tuple in operation.site_tuples
    )
    assert all(site_tuple.duration is None for operation in target.operations for site_tuple in operation.site_tuples)


def _compiler_target_metadata(target: CompilerTarget) -> dict[str, object]:
    """Return all metadata exposed by an immutable compiler target."""
    duration_unit = target.duration_unit
    synthesis_basis = target.synthesis_basis
    return {
        "name": target.name,
        "duration_unit": None if duration_unit is None else (duration_unit.unit, duration_unit.scale_factor),
        "num_qubits": target.num_qubits,
        "sites": [(site.id, site.name, site.t1, site.t2) for site in target.sites],
        "has_explicit_topology": target.has_explicit_topology,
        "couplings": target.couplings,
        "has_explicit_operations": target.has_explicit_operations,
        "operations": [
            (
                operation.name,
                operation.canonical_name,
                operation.num_qubits,
                operation.num_parameters,
                operation.duration,
                operation.fidelity,
                [(site_tuple.sites, site_tuple.duration, site_tuple.fidelity) for site_tuple in operation.site_tuples],
            )
            for operation in target.operations
        ],
        "supported_gates": target.supported_gates,
        "synthesis_basis": (
            None if synthesis_basis is None else (synthesis_basis.single_qubit, synthesis_basis.entangler)
        ),
    }


def test_compiler_target_from_device_id_matches_opened_device() -> None:
    """Stable-ID construction produces the same detached DDSIM target."""
    direct = CompilerTarget.from_device(open_device("mqt.ddsim.default"))
    by_id = CompilerTarget.from_device_id("mqt.ddsim.default", custom1="value")

    assert _compiler_target_metadata(by_id) == _compiler_target_metadata(direct)


def test_compiler_target_from_device_id_preserves_open_and_conversion_errors() -> None:
    """Stable-ID construction retains registry and target compatibility errors."""
    with pytest.raises(IndexError, match="Unknown QDMI device ID"):
        CompilerTarget.from_device_id("unknown.device")
    with pytest.raises(ValueError, match="mutually exclusive"):
        CompilerTarget.from_device_id(
            "mqt.ddsim.default",
            device_config="{}",
            device_config_file=Path("device.json"),
        )


def test_qco_program_runs_textual_pipeline() -> None:
    """Run registered QCO passes through MLIR textual pipeline syntax."""
    qco = compile_program(QASM_STRING, output=OutputFormat.QCO)
    assert isinstance(qco, QCOProgram)

    qco.run_pass_pipeline("mqt-qco-default")
    qco.lift_hadamards()

    with pytest.raises(RuntimeError, match="MLIR operation failed"):
        qco.run_pass_pipeline("not-a-pass")


def test_qco_program_reuses_qubits() -> None:
    """Expose the raw and composite qubit-reuse flows."""
    independent_qubits = """
module {
  func.func @main() attributes {mqt.entry_point} {
    %q0 = qco.alloc : !qco.qubit
    %q1 = qco.alloc : !qco.qubit
    %q0_h = qco.h %q0 : !qco.qubit -> !qco.qubit
    %q1_h = qco.h %q1 : !qco.qubit -> !qco.qubit
    %q0_m, %c0 = qco.measure %q0_h : !qco.qubit
    %q1_m, %c1 = qco.measure %q1_h : !qco.qubit
    qco.sink %q0_m : !qco.qubit
    qco.sink %q1_m : !qco.qubit
    return
  }
}
"""
    raw = QCOProgram.from_mlir_str(independent_qubits)
    assert raw.ir.count("qco.alloc") == 2
    raw.reuse_qubits()
    assert raw.ir.count("qco.alloc") == 1
    assert "qco.reset" in raw.ir

    composite = QCOProgram.from_mlir_str(independent_qubits)
    assert composite.ir.count("qco.alloc") == 2
    composite.run_qubit_reuse_pipeline()
    assert composite.ir.count("qco.alloc") == 1
    assert composite.ir.count("qco.sink") == 1
    assert "qco.h" not in composite.ir
    assert "qco.measure" not in composite.ir
    assert "qco.reset" not in composite.ir


def test_typed_programs_normalize_global_phases() -> None:
    """Normalize QC and QCO phases through the typed Python APIs."""
    qc = QCProgram.from_mlir_str(
        """module {
          func.func @test(%q: !qc.qubit) {
            %a = arith.constant 0.25 : f64
            qc.gphase(%a)
            qc.x %q : !qc.qubit
            %b = arith.constant 0.5 : f64
            qc.gphase(%b)
            return
          }
        }"""
    )
    qc.normalize_global_phases()
    assert qc.ir.count("qc.gphase") == 1

    qco = QCOProgram.from_mlir_str(
        """module {
          func.func @test(%q: !qco.qubit) -> !qco.qubit {
            %a = arith.constant 0.25 : f64
            qco.gphase(%a)
            %q1 = qco.x %q : !qco.qubit -> !qco.qubit
            %b = arith.constant 0.5 : f64
            qco.gphase(%b)
            return %q1 : !qco.qubit
          }
        }"""
    )
    qco.normalize_global_phases()
    assert qco.ir.count("qco.gphase") == 1
    once = qco.ir
    qco.normalize_global_phases()
    assert qco.ir == once


def test_qco_program_decomposes_multi_controlled() -> None:
    """Decompose multi-controlled gates through the typed QCOProgram API."""
    qco = compile_program(
        'OPENQASM 3.0; include "stdgates.inc"; qubit[3] q; ctrl(2) @ x q[0], q[1], q[2];',
        output=OutputFormat.QCO,
    )
    assert isinstance(qco, QCOProgram)
    before = qco.ir
    assert "qco.ctrl" in before

    retained = qco.copy()
    retained.decompose_multi_controlled(min_qubits=4)
    assert "controls_out:2" in retained.ir

    qco.decompose_multi_controlled()
    assert qco.ir != before
    assert "controls_out:2" not in qco.ir

    with pytest.raises(RuntimeError, match="MLIR operation failed"):
        qco.decompose_multi_controlled(min_qubits=2)


def test_compile_program_fails_for_missing_file() -> None:
    """A missing known input file extension raises an error."""
    with pytest.raises(RuntimeError, match="does not exist"):
        compile_program("missing_program.qasm")
