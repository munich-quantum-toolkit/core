# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Native command argument forwarding and exit status."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import distribution
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from qiskit import qpy
from qiskit.quantum_info import Statevector

from mqt.core.mlir import CompilerTarget, OutputFormat, QCProgram, compile_program


def run_command(arguments: list[str], setup: str = "") -> subprocess.CompletedProcess[bytes]:
    """Run the one-shot compiler command in its own process.

    Returns:
        The exit status and captured binary output.
    """
    script = setup + "\nfrom mqt.core.__main__ import main; main()"
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - No shell; isolated LLVM CLI state.
        [sys.executable, "-c", script, *arguments], capture_output=True, check=False
    )


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_native_command(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    """Forward arguments intact and preserve a failing native tool's exit code."""
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys, "argv", ["mqt-core", "bench", "input with spaces.qasm", "--help"])
    execute = Mock(side_effect=SystemExit(7))
    run = Mock(return_value=Mock(returncode=7))
    monkeypatch.setattr(os, "execv", execute)
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(SystemExit) as error:
        distribution("mqt-core").entry_points["mqt-core"].load()()
    assert error.value.code == 7
    suffix = ".exe" if platform == "win32" else ""
    executable = Path(str(distribution("mqt-core").locate_file(f"mqt/core/bin/mqt-core{suffix}")))
    arguments = [str(executable), "bench", "input with spaces.qasm", "--help"]
    if platform == "win32":
        run.assert_called_once_with(arguments, check=False)
    else:
        execute.assert_called_once_with(executable, arguments)


@pytest.mark.parametrize("destination", ["file", "stdout", "missing-directory"])
def test_qpy_export(
    tmp_path: Path,
    destination: str,
) -> None:
    """Run the native compiler and independently decode QPY or report an output failure."""
    source = tmp_path / "input.qasm"
    source.write_text("OPENQASM 3; qubit q; x q;")
    output = tmp_path / "output.qpy"
    arguments = ["compile", str(source), "-o", str(output), "--qdmi-list-devices=false"]
    if destination == "stdout":
        arguments = ["compile", str(source), "--emit=qpy"]
    elif destination == "missing-directory":
        arguments[arguments.index("-o") + 1] = str(tmp_path / "absent" / "output.qpy")
    error = run_command(arguments)
    if destination == "missing-directory":
        assert error.returncode != 0
        assert error.stderr
    else:
        assert error.returncode == 0
        data = error.stdout if destination == "stdout" else output.read_bytes()
        assert qpy.get_qpy_version(BytesIO(data)) == 17
        circuit = qpy.load(BytesIO(data))[0]
        assert Statevector.from_instruction(circuit).probabilities_dict() == {"1": 1.0}


@pytest.mark.parametrize("failure", ["compiler", "serializer"])
def test_qpy_export_failure(tmp_path: Path, failure: str) -> None:
    """Compilation and serialization failures leave an existing destination intact."""
    source = tmp_path / "input.qasm"
    source.write_text("not OpenQASM" if failure == "compiler" else "OPENQASM 3; qubit q; x q;")
    setup = ""
    if failure == "serializer":
        setup = """
from qiskit import qpy
from qiskit.qpy.exceptions import QpyError
def fail(*args, **kwargs):
    raise QpyError("Cannot serialize circuit")
qpy.dump = fail
"""
    output = tmp_path / "output.qpy"
    output.write_bytes(b"existing")
    assert run_command(["compile", str(source), "-o", str(output)], setup).returncode != 0
    assert output.read_bytes() == b"existing"


@pytest.mark.parametrize("input_format", ["openqasm", "qc", "qco", "jeff"])
def test_qpy_input_formats(tmp_path: Path, input_format: str) -> None:
    """Load each supported frontend, including an explicit extension override."""
    source = (
        tmp_path / {"openqasm": "circuit.txt", "qc": "qc.mlir", "qco": "qco.mlir", "jeff": "circuit.jeff"}[input_format]
    )
    qasm = "OPENQASM 3; qubit q; x q;"
    if input_format == "jeff":
        compile_program(qasm, output=OutputFormat.JEFF).write(source)
    elif input_format == "openqasm":
        source.write_text(qasm)
    else:
        program = QCProgram.from_openqasm_str(qasm)
        source.write_text(program.to_qco().ir if input_format == "qco" else program.ir)
    output = tmp_path / "output.qpy"
    arguments = ["compile", str(source), "-o", str(output)]
    if input_format == "openqasm":
        arguments.append("--input-format=openqasm")
    assert run_command(arguments).returncode == 0
    assert Statevector(qpy.load(BytesIO(output.read_bytes()))[0]).probabilities_dict() == {"1": 1.0}


@pytest.mark.parametrize(
    ("arguments", "width"),
    [
        (["--decompose-multi-controlled"], 2),
        (["--decompose-multi-controlled=false"], 3),
        (["--decompose-multi-controlled", "--decompose-multi-controlled-min-qubits=4"], 3),
    ],
)
def test_qpy_decomposition(tmp_path: Path, arguments: list[str], width: int) -> None:
    """Honor the decomposition threshold while retaining the circuit's action."""
    source = tmp_path / "input.qasm"
    source.write_text("OPENQASM 3; qubit[3] q; x q[0]; x q[1]; ctrl(2) @ x q[0],q[1],q[2];")
    output = tmp_path / "output.qpy"
    assert run_command(["compile", *arguments, str(source), "-o", str(output)]).returncode == 0
    circuit = qpy.load(BytesIO(output.read_bytes()))[0]
    assert max(item.operation.num_qubits for item in circuit.data) == width
    assert Statevector(circuit).data == pytest.approx(Statevector.from_label("111").data, abs=1e-14)


def test_qpy_custom_pipeline(tmp_path: Path) -> None:
    """Run an explicitly requested pipeline through its CLI alias."""
    source = tmp_path / "input.qasm"
    source.write_text(
        'OPENQASM 3; include "stdgates.inc"; input float unused; qubit[8] q; '
        "if (false) { rx(unused) q[7]; } gphase(0.37); x q[2];"
    )
    output = tmp_path / "output.qpy"
    error = run_command([
        "compile",
        str(source),
        "-o",
        str(output),
        "--passes=builtin.module(drop-global-phases)",
        "--mlir-print-ir-after-all",
    ])
    assert error.returncode == 0
    assert Statevector(qpy.load(BytesIO(output.read_bytes()))[0]).data == pytest.approx([0, 1], abs=1e-14)


@pytest.mark.parametrize(
    ("gate_name", "payload_format", "input_format"),
    [
        ("u3", "qir", "openqasm"),
        ("device_u", "qpy", "openqasm"),
        ("device_u", "qpy", "qc"),
        ("device_u", "qpy", "qco"),
    ],
)
def test_qpy_target_export(tmp_path: Path, gate_name: str, input_format: str, payload_format: str) -> None:
    """Retain native gate names, sparse device addresses, and logical layout."""
    target = CompilerTarget(
        "sparse device",
        [CompilerTarget.Site(2), CompilerTarget.Site(5)],
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations([
            CompilerTarget.OperationCapability(gate_name, 1, 3, canonical_name="u"),
            CompilerTarget.OperationCapability("gphase", 0, 1),
            CompilerTarget.OperationCapability("measure", 1, 0),
        ]),
    )
    format_attribute = (
        'id = "qir", version = "2.1.0", profile = "base"'
        if payload_format == "qir"
        else 'id = "qpy", version = "17", profile = ""'
    )
    payload = (
        f"#mqt.payload_spec<format = <{format_attribute}, encoding = binary>, "
        "capabilities = [], optional_capabilities_known = true>"
    )
    source = tmp_path / ("input.qasm" if input_format == "openqasm" else "input.mlir")
    qasm = "OPENQASM 3; qubit q; x q;"
    arguments = []
    if input_format == "openqasm":
        source.write_text(qasm)
        arguments = [
            f"--target={target}",
            f"--payload-spec={payload}",
            "--seed=42",
            "--mapping-trials=1",
            "--mapping-iterations=0",
            "--mapping-lookahead=0",
            "--mapping-search-memory-limit=0",
        ]
    else:
        program = compile_program(qasm, target=target, output=OutputFormat.QC)
        source.write_text(program.to_qco().ir if input_format == "qco" else program.ir)
    output = tmp_path / "output.qpy"
    assert run_command(["compile", str(source), "-o", str(output), *arguments]).returncode == 0
    with output.open("rb") as stream:
        restored = qpy.load(stream)[0]
    assert restored.count_ops() == {gate_name: 1}
    assert restored.num_qubits == 6
    physical_qubit = restored.find_bit(restored.data[0].qubits[0]).index
    assert physical_qubit in {2, 5}
    if input_format == "openqasm":
        assert restored.layout is not None
        assert restored.layout.initial_index_layout(filter_ancillas=True) == [physical_qubit]
    expected = Statevector.from_int(1 << physical_qubit, 2**restored.num_qubits)
    assert Statevector.from_instruction(restored).data == pytest.approx(expected.data, abs=1e-14)


@pytest.mark.parametrize("arguments", [["--help"], ["-o", "output.qasm"], ["-o", "output.qpy"]])
def test_compile_dispatch(monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    """Every compiler output uses the same in-process command."""
    from mqt.core import mlir  # ruff: ignore[import-outside-top-level] - Patch the private command entry.

    compiler = Mock(return_value=7)
    monkeypatch.setattr(mlir, "_run_compiler", compiler)
    monkeypatch.setattr(sys, "argv", ["mqt-core", "compile", *arguments])
    with pytest.raises(SystemExit) as error:
        distribution("mqt-core").entry_points["mqt-core"].load()()
    assert error.value.code == 7
    compiler.assert_called_once_with(["compile", *arguments])


def test_qpy_requires_optional_qiskit() -> None:
    """Missing Qiskit produces an installation hint before compilation."""
    result = run_command(
        ["compile", "--emit=qpy"],
        """
import importlib.metadata
original = importlib.metadata.version
def version(name):
    if name == "qiskit":
        raise importlib.metadata.PackageNotFoundError(name)
    return original(name)
importlib.metadata.version = version
""",
    )
    assert result.returncode == 1
    assert b"install mqt-core[qiskit]" in result.stderr


def test_compiler_entry_is_one_shot(tmp_path: Path) -> None:
    """Repeated calls fail before touching LLVM's process-global options."""
    source = tmp_path / "input.qasm"
    source.write_text("OPENQASM 3; qubit q; x q;")
    script = """
from mqt.core.mlir import _run_compiler
try:
    _run_compiler([])
except ValueError:
    pass
else:
    raise AssertionError("empty arguments accepted")
assert _run_compiler(["compile", "--input-format=invalid"]) == 1
"""
    result = run_command(["compile", str(source), "--emit=qpy"], script)
    assert result.returncode != 0
    assert b"can only run once" in result.stderr
