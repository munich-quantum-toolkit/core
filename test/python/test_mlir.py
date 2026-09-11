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
import sys
from pathlib import Path
from threading import Event, Thread

import numpy as np
import pytest
import qiskit
from packaging import version
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, library
from qiskit.quantum_info import Operator

from mqt.core.mlir import (
    CompilerTarget,
    JeffProgram,
    OpenQASMProgram,
    OutputFormat,
    PayloadEncoding,
    PayloadFormat,
    PayloadSpecification,
    ProgramCapability,
    ProgramConstraint,
    QCOProgram,
    QCProgram,
    QIRProfile,
    QIRProgram,
    TargetEnvironment,
    compile_program,
)
from mqt.core.qdmi import ProgramFormat
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


def _test_payload_specification() -> PayloadSpecification:
    """Return one explicit selected payload contract for target tests."""
    return PayloadSpecification(
        PayloadFormat("qir", "2.1.0", "base", PayloadEncoding.BINARY),
        [
            ProgramCapability(
                ProgramCapability.FORWARD_BRANCHING, 0, [ProgramConstraint(ProgramConstraint.MAX_NESTING_DEPTH, 8)]
            )
        ],
        optional_capabilities_known=True,
    )


def _test_target_environment(target: CompilerTarget) -> TargetEnvironment:
    """Pair a compiler target with the test payload specification.

    Returns:
        The complete target environment.
    """
    return TargetEnvironment(target, _test_payload_specification())


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
    source = (
        " module { func.func @main() attributes {mqt.entry_point} {"
        " %0 = qc.alloc : !qc.qubit qc.dealloc %0 : !qc.qubit return } }"
    )

    result = compile_program(source)

    assert isinstance(result, QCProgram)
    assert result.ir.startswith("module")


@pytest.mark.parametrize("version_header", ["", "OPENQASM 2.0;", "OPENQASM 3.0;", "OPENQASM 3.1;"])
def test_openqasm_import_versions(version_header: str, tmp_path: Path) -> None:
    """Both OpenQASM factories accept every supported version."""
    source = f'{version_header}\ninclude "qelib1.inc"; qreg q[1]; h q[0];'
    path = tmp_path / "program.qasm"
    path.write_text(source, encoding="utf-8")

    for program in (QCProgram.from_openqasm_str(source), QCProgram.from_openqasm_file(path)):
        assert program.is_valid
        assert program.num_gates() == 1
        assert compile_program(program, output=OutputFormat.QCO).is_valid


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
    """Compile a serialized jeff program through the QCO pipeline entry point."""
    path = tmp_path / "program.jeff"
    compile_program(QASM_STRING, output=OutputFormat.JEFF).write(path)

    result = compile_program(path, output=OutputFormat.QCO)

    assert isinstance(result, QCOProgram)
    assert "qco." in result.ir


def test_program_conversions_are_composable() -> None:
    """Compose frontend, cleanup, conversion, and optimization stages."""
    source = QCProgram.from_openqasm_str(QASM_STRING)
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
    source = QCProgram.from_openqasm_str(QASM_STRING)
    direct = source.to_openqasm3()

    assert isinstance(direct, OpenQASMProgram)
    assert source.is_valid
    assert direct.source.startswith("OPENQASM 3.1;")
    assert str(direct) == direct.source

    path = tmp_path / "program.qasm"
    direct.write(path)
    assert path.read_text(encoding="utf-8") == direct.source
    _assert_bell_program(QCProgram.from_openqasm_file(path), measured=True)

    optimized = compile_program(QASM_STRING, output=OutputFormat.OPENQASM3)
    assert isinstance(optimized, OpenQASMProgram)
    assert "output bit[2] c;" in optimized.source
    _assert_bell_program(QCProgram.from_openqasm_str(optimized.source), measured=True)

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
    round_tripped = QCProgram.from_openqasm_str(source).to_qiskit()

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
    result = QCProgram.from_openqasm_str(QASM_STRING).to_qir(QIRProfile.ADAPTIVE)

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


@requires_qiskit_translation
def test_empty_compiled_program_round_trips_through_mlir() -> None:
    """Reload empty compiled IR through both typed program APIs."""
    circuit = QuantumCircuit(0)
    program = QCProgram.from_qiskit(circuit).to_qco()
    target = CompilerTarget(
        1,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations.unrestricted(),
    )
    program.compile_for_target(_test_target_environment(target))
    assert "qco." not in program.ir

    qc = QCOProgram.from_mlir_str(program.ir).to_qc()
    restored = QCProgram.from_mlir_str(qc.ir).to_qiskit()

    assert restored.num_qubits == 0
    assert restored.num_clbits == 0
    assert Operator(restored) == Operator(circuit)


@pytest.fixture(scope="module")
def garnet_target() -> CompilerTarget:
    """Snapshot the bundled IQM Garnet device.

    Returns:
        The detached compiler target.
    """
    return CompilerTarget.from_device_id("mqt.sc.iqm.garnet")


def test_compile_program_for_qdmi_target(garnet_target: CompilerTarget) -> None:
    """Compile through the canonical target pipeline for a QDMI device."""
    result = compile_program(QASM_STRING, target=garnet_target, output=OutputFormat.QIR_BASE)

    assert isinstance(result, QIRProgram)
    assert result.profile == QIRProfile.BASE

    mapped = compile_program(QASM_STRING, output=OutputFormat.QCO)
    assert isinstance(mapped, QCOProgram)
    mapped.compile_for_target(_test_target_environment(garnet_target))
    static_sites = {int(site) for site in re.findall(r"qco\.static (\d+)", mapped.ir)}
    assert len(static_sites) == 2
    assert static_sites <= {site.id for site in garnet_target.sites}
    assert "qco.r(" in mapped.ir
    assert "qco.ctrl" in mapped.ir
    assert "qco.z " in mapped.ir
    assert mapped.ir.count("qco.measure") == 2
    assert "qco.rx" not in mapped.ir
    assert "qco.ry" not in mapped.ir


def test_compile_program_rejects_unsupported_target_payload_without_consuming_input() -> None:
    """Reject an unsupported selected payload before consuming typed input."""
    program = compile_program(QASM_STRING, output=OutputFormat.QCO)
    assert isinstance(program, QCOProgram)
    target = CompilerTarget(
        2,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations.unrestricted(),
    )
    with pytest.raises(ValueError, match="executable output"):
        compile_program(program, target=target, output=OutputFormat.QCO, inplace=True)  # ty: ignore[no-matching-overload]

    assert program.is_valid


def test_qco_program_compiles_for_direct_sparse_target() -> None:
    """Expose direct target construction and typed QCO compilation."""
    target = CompilerTarget(
        "sparse target",
        [CompilerTarget.Site(10), CompilerTarget.Site(20)],
        connectivity=CompilerTarget.Connectivity([(10, 20)]),
        native_operations=CompilerTarget.NativeOperations([
            CompilerTarget.OperationCapability("u", 1, 3),
            CompilerTarget.OperationCapability("cz", 2, 0),
            CompilerTarget.OperationCapability("measure", 1, 0),
        ]),
    )
    assert target.name == "sparse target"
    assert [site.id for site in target.sites] == [10, 20]
    assert target.couplings == [(10, 20)]
    assert target.synthesis_basis is not None
    assert target.synthesis_basis.single_qubit == CompilerTarget.SingleQubitBasis.U
    assert target.synthesis_basis.entangler == CompilerTarget.GateKind.CZ

    qco = compile_program(QASM_STRING, output=OutputFormat.QCO)
    assert isinstance(qco, QCOProgram)

    qco.compile_for_target(_test_target_environment(target))

    assert {int(site) for site in re.findall(r"qco\.static (\d+)", qco.ir)} == {10, 20}
    assert "qco.u(" in qco.ir
    assert "qco.ctrl" in qco.ir
    assert "qco.z " in qco.ir
    assert qco.ir.count("qco.measure") == 2


@requires_qiskit_translation
@pytest.mark.parametrize("num_sites", [1, 2])
def test_target_compiles_single_qubit_gates_without_entangler(num_sites: int) -> None:
    """Compile a non-native rotation without inventing a two-qubit capability."""
    target = CompilerTarget(
        num_sites,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations([
            CompilerTarget.OperationCapability("sx", 1, 0),
            CompilerTarget.OperationCapability("x", 1, 0),
            CompilerTarget.OperationCapability("rz", 1, 1),
            CompilerTarget.OperationCapability("gphase", 0, 1),
        ]),
    )
    assert target.synthesis_basis is not None
    assert target.synthesis_basis.single_qubit == CompilerTarget.SingleQubitBasis.ZSXX
    assert target.synthesis_basis.entangler is None
    source = QuantumCircuit(num_sites)
    for site in range(num_sites):
        source.ry(0.123, site)
    program = QCProgram.from_qiskit(source).to_qco()

    program.compile_for_target(_test_target_environment(target))

    assert program.is_valid
    result = program.to_qc().to_qiskit(target=target)
    assert set(result.count_ops()) <= {"sx", "x", "rz"}
    assert np.allclose(Operator(result).data, Operator(source).data)


@requires_qiskit_translation
def test_target_compilation_exports_canonical_physical_qiskit_circuit() -> None:
    """Export a mapped program with the complete compiler target."""
    target = CompilerTarget(
        5,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations.unrestricted(),
    )
    mapped = compile_program(QASM_STRING, output=OutputFormat.QCO)
    assert isinstance(mapped, QCOProgram)
    mapped.compile_for_target(_test_target_environment(target))
    assert 0 < mapped.ir.count("qco.static") < target.num_sites

    source_ir = mapped.ir
    restored = mapped.to_qiskit(target=target)

    assert mapped.is_valid
    assert mapped.ir == source_ir
    assert restored == mapped.to_qc(copy=True).to_qiskit(target=target)
    assert restored.num_qubits == 5
    assert [(register.name, len(register)) for register in restored.qregs] == [("q", 5)]
    assert restored.layout is None


@requires_qiskit_translation
def test_target_synthesis_decomposes_without_routing() -> None:
    """Synthesize a controlled rotation through the typed basis-only API."""
    target = CompilerTarget(
        3,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations([
            CompilerTarget.OperationCapability("sx", 1, 0),
            CompilerTarget.OperationCapability("x", 1, 0),
            CompilerTarget.OperationCapability("rz", 1, 1),
            CompilerTarget.OperationCapability("cz", 2, 0),
            CompilerTarget.OperationCapability("gphase", 0, 1),
        ]),
    )
    source = QuantumCircuit(3)
    source.h(0)
    source.append(library.RYGate(0.7).control(2, annotated=True), [0, 1, 2])
    program = QCProgram.from_qiskit(source).to_qco()

    program.synthesize_for_target(_test_target_environment(target))

    result = program.to_qiskit(target=target)
    assert set(result.count_ops()) <= {"sx", "x", "rz", "cz"}
    assert np.allclose(Operator(result).data, Operator(source).data)

    sparse = CompilerTarget(
        3,
        connectivity=CompilerTarget.Connectivity([(0, 1), (1, 2)]),
        native_operations=CompilerTarget.NativeOperations.unrestricted(),
    )
    with pytest.raises(RuntimeError, match="all-to-all connectivity"):
        program.synthesize_for_target(_test_target_environment(sparse))


@requires_qiskit_translation
def test_qco_qiskit_export_preserves_program() -> None:
    """Reuse QC export without consuming QCO, including when export fails."""
    source = QuantumCircuit(2)
    source.h(0)
    source.cx(0, 1)
    program = QCProgram.from_qiskit(source).to_qco()
    source_ir = program.ir

    assert np.allclose(Operator(program.to_qiskit()).data, Operator(source).data)
    assert program.ir == source_ir

    target = CompilerTarget(
        2,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations.unrestricted(),
    )
    with pytest.raises(RuntimeError, match="requires statically mapped qubits"):
        program.to_qiskit(target=target)
    assert program.ir == source_ir

    program.to_qc()
    with pytest.raises(RuntimeError, match="already been consumed"):
        program.to_qiskit()


@pytest.mark.parametrize("kind", ["qc", "qco", "jeff"])
@pytest.mark.parametrize("action", ["copy", "cleanup", "compile", "compile_inplace"])
def test_consumed_program_operations_raise(kind: str, action: str) -> None:
    """Report consumed program use as a Python exception across binding paths."""
    program: QCProgram | QCOProgram | JeffProgram = QCProgram.from_openqasm_str(QASM_STRING)
    if kind != "qc":
        program = program.to_qco()
        if kind == "jeff":
            program = program.to_jeff()
    if isinstance(program, QCOProgram):
        program.to_qc()
    else:
        program.to_qco()
    assert not program.is_valid

    operation = {
        "copy": program.copy,
        "cleanup": program.cleanup,
        "compile": lambda: compile_program(program),
        "compile_inplace": lambda: compile_program(program, inplace=True),
    }[action]
    with pytest.raises(RuntimeError, match="already been consumed"):
        operation()


def test_consumed_jeff_write_raises(tmp_path: Path) -> None:
    """Guard const member adapters before entering the native writer."""
    program = QCProgram.from_openqasm_str(QASM_STRING).to_qco().to_jeff()
    program.to_qco()
    with pytest.raises(RuntimeError, match="already been consumed"):
        program.write(tmp_path / "consumed.jeff")


@pytest.mark.parametrize("capability_id", [None, "forward-branching-typo"])
def test_target_compilation_preserves_diagnostics(capability_id: str | None, capfd: pytest.CaptureFixture[str]) -> None:
    """Keep native control-flow legality errors in the Python exception."""
    program = QCProgram.from_openqasm_str("""OPENQASM 3.0;
include "stdgates.inc";
qubit q;
bit c;
h q;
c = measure q;
if (c) { x q; }
""").to_qco()
    target = CompilerTarget(
        1,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations.unrestricted(),
    )
    payload = PayloadSpecification(
        PayloadFormat("openqasm", "3.1"),
        [ProgramCapability(capability_id)] if capability_id is not None else [],
    )
    valid = program.copy()
    with pytest.raises(RuntimeError, match=r"Target compilation failed.*qco\.if"):
        program.compile_for_target(TargetEnvironment(target, payload))

    # A copy shares the context, whose diagnostic handler must be restored.
    capfd.readouterr()
    with pytest.raises(RuntimeError):
        valid.run_pass_pipeline("not-a-pass")
    assert "failed to parse pass pipeline" in capfd.readouterr().err
    valid.compile_for_target(_test_target_environment(target))
    valid.to_qc()
    with pytest.raises(RuntimeError, match="already been consumed"):
        valid.compile_for_target(TargetEnvironment(target, payload))


def test_compiler_target_constructors_preserve_python_api() -> None:
    """Construct every target metadata type and target overload."""
    duration_unit = CompilerTarget.DurationUnit("ns", 1.0)
    sites = [
        CompilerTarget.Site(10, "q0", 100, 200),
        CompilerTarget.Site(20, "q1"),
    ]
    site_tuple = CompilerTarget.SiteTuple([10, 20], duration=10, fidelity=0.99)
    operation = CompilerTarget.OperationCapability(
        "cx",
        2,
        0,
        site_tuples=[site_tuple],
        duration=20,
        fidelity=0.98,
    )
    fixed_zero = CompilerTarget.OperationArity.fixed(0)
    variadic = CompilerTarget.OperationArity.variadic(2)
    global_phase = CompilerTarget.OperationCapability("gphase", fixed_zero, 1)
    multi_controlled_x = CompilerTarget.OperationCapability("x", variadic, 0)
    connectivity = CompilerTarget.Connectivity.all_to_all()
    unrestricted = CompilerTarget.NativeOperations.unrestricted()

    targets = [
        CompilerTarget(2, connectivity=connectivity, native_operations=unrestricted, duration_unit=duration_unit),
        CompilerTarget(
            "dense", 2, connectivity=connectivity, native_operations=unrestricted, duration_unit=duration_unit
        ),
        CompilerTarget(
            sites,
            connectivity=connectivity,
            native_operations=CompilerTarget.NativeOperations([operation]),
            duration_unit=duration_unit,
        ),
        CompilerTarget(
            "sparse",
            sites,
            connectivity=connectivity,
            native_operations=CompilerTarget.NativeOperations([operation]),
            duration_unit=duration_unit,
        ),
    ]

    assert [target.num_sites for target in targets] == [2, 2, 2, 2]
    assert targets[1].name == "dense"
    assert targets[3].name == "sparse"
    assert sites[0].name == "q0"
    assert sites[0].t1 == 100
    assert sites[0].t2 == 200
    assert site_tuple.sites == [10, 20]
    assert len(operation.site_tuples) == 1
    assert operation.site_tuples[0].sites == [10, 20]
    assert not CompilerTarget.OperationCapability("x", 1, 0).site_tuples
    assert targets[0].supports_operation("ecr", 2, sites=[0, 1])
    assert not targets[2].supports_operation("ecr", 2, sites=[10, 20])
    assert targets[2].supports_operation("cx", 2, sites=[10, 20])
    assert not targets[2].supports_operation("cx", 2, sites=[20, 10])
    assert operation.arity.kind == CompilerTarget.OperationArityKind.FIXED
    assert operation.arity.value == 2
    assert global_phase.arity.kind == CompilerTarget.OperationArityKind.FIXED
    assert global_phase.arity.value == 0
    assert fixed_zero.accepts(0)
    assert not fixed_zero.accepts(1)
    assert multi_controlled_x.arity.kind == CompilerTarget.OperationArityKind.VARIADIC
    assert multi_controlled_x.arity.value == 2
    assert not variadic.accepts(1)
    assert variadic.accepts(2)
    assert variadic.accepts(5)
    assert duration_unit.unit == "ns"


@pytest.mark.parametrize("arity", [2, CompilerTarget.OperationArity.fixed(2)])
def test_compiler_target_accepts_plain_site_tuples(arity: int | CompilerTarget.OperationArity) -> None:
    """Mix plain placements and calibrated tuples without widening support."""
    operation = CompilerTarget.OperationCapability(
        "cx", arity, 0, site_tuples=[(1, 0), [1, 2], CompilerTarget.SiteTuple([2, 0], fidelity=0.99)]
    )
    target = CompilerTarget(
        3,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations([operation]),
    )
    assert [entry.sites for entry in operation.site_tuples] == [[1, 0], [1, 2], [2, 0]]
    assert [entry.fidelity for entry in operation.site_tuples] == [None, None, 0.99]
    assert target.supports_operation("cx", 2, sites=[1, 0])
    assert not target.supports_operation("cx", 2, sites=[0, 1])
    with pytest.raises(ValueError, match="site tuple does not match its arity"):
        CompilerTarget.OperationCapability("cx", arity, 0, site_tuples=[(0,)])


def test_payload_specification_preserves_python_api() -> None:
    """Construct and validate one context-free selected payload contract."""
    payload_format = PayloadFormat("qir", "2.1.0", "base", PayloadEncoding.BINARY)
    constraint = ProgramConstraint(ProgramConstraint.MAX_NESTING_DEPTH, 8)
    capability = ProgramCapability(ProgramCapability.FORWARD_BRANCHING, 0, [constraint])
    environment = PayloadSpecification(
        payload_format,
        [capability],
        optional_capabilities_known=True,
    )

    assert environment.format.format_id == "qir"
    assert environment.format.version == "2.1.0"
    assert environment.format.profile == "base"
    assert environment.format.encoding == PayloadEncoding.BINARY
    assert environment.capabilities[0].capability_id == "forward-branching"
    assert environment.capabilities[0].value == 0
    assert environment.capabilities[0].constraints[0].constraint_id == "max-control-flow-nesting-depth"
    assert environment.capabilities[0].constraints[0].value == 8
    assert environment.optional_capabilities_known
    assert ProgramCapability.COUNTED_ITERATION == "counted-iteration"
    assert ProgramCapability.CONDITIONAL_LOOP == "conditional-loop"
    assert ProgramCapability.MULTIWAY_BRANCHING == "multiway-branching"
    assert ProgramConstraint.MAX_ITERATION_COUNT == "max-iteration-count"
    assert ProgramConstraint.MAX_CASE_COUNT == "max-case-count"

    payload_format.version = "9.9.9"
    exposed_descriptor = environment.format
    exposed_descriptor.version = "8.8.8"
    capability.value = 1
    assert environment.format.version == "2.1.0"
    assert environment.capabilities[0].value == 0

    with pytest.raises(ValueError, match=r"major\[\.minor\[\.patch\]\]"):
        PayloadSpecification(PayloadFormat("qir", "2.1.0.1", "base"))


@pytest.mark.parametrize(
    ("format_id", "version", "profile", "expected_version", "expected_type"),
    [("qir", "2.1", "base", "2.1.0", QIRProgram), ("openqasm", "3.1", "", "3.1.0", OpenQASMProgram)],
)
def test_target_compilation_accepts_exact_version_shorthand(
    format_id: str, version: str, profile: str, expected_version: str, expected_type: type
) -> None:
    """Normalize a shortened version before selecting the compiler output."""
    payload = PayloadSpecification(PayloadFormat(format_id, version, profile))
    assert payload.format.version == expected_version
    target = CompilerTarget(
        2,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations.unrestricted(),
    )
    program = compile_program(QASM_STRING, output=OutputFormat.QCO)
    program.compile_for_target(TargetEnvironment(target, payload))
    qc = program.to_qc()
    result = qc.to_qir(QIRProfile.BASE) if format_id == "qir" else qc.to_openqasm3()
    assert isinstance(result, expected_type)


def test_compiler_target_construction_preserves_validation_errors() -> None:
    """Translate explicit C++ construction errors to Python ``ValueError``."""
    with pytest.raises(TypeError):
        CompilerTarget(1)  # ty: ignore[no-matching-overload]
    for _ in range(2):
        with pytest.raises(ValueError, match="must contain at least one site"):
            CompilerTarget(
                0,
                connectivity=CompilerTarget.Connectivity.all_to_all(),
                native_operations=CompilerTarget.NativeOperations.unrestricted(),
            )
    with pytest.raises(ValueError, match="site ID must be nonnegative"):
        CompilerTarget.Site(-1)
    with pytest.raises(ValueError, match="contains a duplicate site"):
        CompilerTarget.SiteTuple([0, 0])
    with pytest.raises(ValueError, match="duration unit must not be empty"):
        CompilerTarget.DurationUnit("", 1.0)
    with pytest.raises(ValueError, match="zero-arity operation cannot contain site tuples"):
        CompilerTarget.OperationCapability(
            "gphase",
            CompilerTarget.OperationArity.fixed(0),
            1,
            site_tuples=[CompilerTarget.SiteTuple([])],
        )
    with pytest.raises(ValueError, match="variadic minimum must be positive"):
        CompilerTarget.OperationCapability("x", CompilerTarget.OperationArity.variadic(0), 0)
    with pytest.raises(ValueError, match="variadic operation cannot contain site tuples"):
        CompilerTarget.OperationCapability(
            "x",
            CompilerTarget.OperationArity.variadic(2),
            0,
            site_tuples=[CompilerTarget.SiteTuple([0, 1])],
        )
    with pytest.raises(ValueError, match="site tuple does not match its arity"):
        CompilerTarget.OperationCapability("cx", 2, 0, site_tuples=[CompilerTarget.SiteTuple([0])])


def test_compiler_target_snapshots_qdmi_device(garnet_target: CompilerTarget) -> None:
    """Retain IQM topology and calibration independently of the live device."""
    target = garnet_target

    assert target.name == "IQM Garnet"
    assert target.num_sites == 20
    assert target.connectivity_kind == CompilerTarget.ConnectivityKind.EXPLICIT
    assert target.native_operations_kind == CompilerTarget.NativeOperationsKind.EXPLICIT
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
        "num_sites": target.num_sites,
        "sites": [(site.id, site.name, site.t1, site.t2) for site in target.sites],
        "connectivity_kind": target.connectivity_kind,
        "couplings": target.couplings,
        "native_operations_kind": target.native_operations_kind,
        "operations": [
            (
                operation.name,
                operation.canonical_name,
                (operation.arity.kind, operation.arity.value),
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

    with pytest.raises(RuntimeError, match="Compiler action failed"):
        qco.run_pass_pipeline("not-a-pass")


def test_qco_program_runs_pauli_twirling_pass() -> None:
    """Run default and seeded Pauli twirling on copied QCO programs."""
    source = compile_program(QASM_STRING, output=OutputFormat.QCO)
    assert isinstance(source, QCOProgram)

    twirled = source.copy()
    twirled.run_pass_pipeline("pauli-twirl-2q-gates{seed=6}")

    default_twirled = source.copy()
    default_twirled.run_pass_pipeline("pauli-twirl-2q-gates")
    seeded_twirled = source.copy()
    seeded_twirled.run_pass_pipeline("pauli-twirl-2q-gates{seed=42}")

    assert default_twirled.ir == seeded_twirled.ir
    assert source.ir.count("qco.id ") == 0
    assert twirled.ir.count("qco.id ") == 4
    assert twirled.ir.count("qco.ctrl(") == 1


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


@pytest.mark.parametrize("gate", ["x", "y", "rx(0.73)", "ry(0.73)", "rz(0.73)"])
def test_qco_program_decomposes_multi_controlled(gate: str) -> None:
    """Decompose multi-controlled gates through the typed QCOProgram API."""
    qco = compile_program(
        f'OPENQASM 3.0; include "stdgates.inc"; qubit[3] q; ctrl(2) @ {gate} q[0], q[1], q[2];',
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

    with pytest.raises(RuntimeError, match="Compiler action failed"):
        qco.decompose_multi_controlled(min_qubits=2)


def test_compile_program_fails_for_missing_file() -> None:
    """A missing known input file extension raises an error."""
    with pytest.raises(RuntimeError, match="does not exist"):
        compile_program("missing_program.qasm")


def test_qc_program_num_gates() -> None:
    """Expose gate counts to Python."""
    program = QCProgram.from_openqasm_str(QASM_STRING)
    assert program.num_gates() == 2
    assert program.num_single_qubit_gates() == 1
    assert program.num_two_qubit_gates() == 1


@pytest.mark.parametrize("mode", ["targetless", "target_output", "target_payload", "source", "path"])
def test_native_compilation_releases_gil(tmp_path: Path, mode: str) -> None:
    """Python threads progress during native parsing and each compilation overload."""
    source = 'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q;\n' + (
        "rx(0.1) q[0]; cx q[0],q[1]; rz(0.2) q[1];\n" * 2000
    )
    target = CompilerTarget(
        2,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations.unrestricted(),
    )
    program = compile_program(source, output=OutputFormat.QCO)
    invalid_source = source + "unknown_gate q[0];"
    path = tmp_path / "invalid.qasm"
    path.write_text(invalid_source, encoding="utf-8")
    start = Event()
    progress = Event()

    def worker() -> None:
        start.wait()
        progress.set()

    thread = Thread(target=worker)
    interval = sys.getswitchinterval()
    try:
        # Prevent interpreter time slices around the native call from passing the check.
        sys.setswitchinterval(60)
        thread.start()
        start.set()
        # ponytail: ten calls allow scheduling; add a native barrier if this remains flaky.
        for _ in range(10):
            if mode == "targetless":
                compile_program(program, output=OutputFormat.OPENQASM3)
            elif mode == "target_output":
                compile_program(program, target=target, output=OutputFormat.OPENQASM3)
            elif mode == "target_payload":
                compile_program(program, target=target, program_format=ProgramFormat.QASM3)
            else:
                # Parsing fails before the compilation release scope can be reached.
                with pytest.raises(RuntimeError, match="Compiler action failed"):
                    compile_program(path if mode == "path" else invalid_source, output=OutputFormat.QCO)
            if progress.is_set():
                break
        assert progress.is_set(), "native parsing or compilation held the GIL"
    finally:
        start.set()
        thread.join(timeout=5)
        sys.setswitchinterval(interval)
    assert not thread.is_alive()
    assert program.is_valid
