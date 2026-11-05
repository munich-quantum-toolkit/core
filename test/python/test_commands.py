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
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from qiskit import qpy
from qiskit.qpy.exceptions import QpyError
from qiskit.quantum_info import Statevector

from mqt.core import mlir
from mqt.core.mlir import CompilerTarget, OutputFormat, QCProgram, compile_program

if TYPE_CHECKING:
    from collections.abc import Callable

    CommandRunner = Callable[[list[str]], SystemExit]


@pytest.fixture
def run_command(monkeypatch: pytest.MonkeyPatch) -> CommandRunner:
    """Return a runner that captures the installed command's SystemExit."""

    def run(arguments: list[str]) -> SystemExit:
        monkeypatch.setattr(sys, "argv", ["mqt-core", *arguments])
        with pytest.raises(SystemExit) as error:
            distribution("mqt-core").entry_points["mqt-core"].load()()
        return error.value

    return run


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_native_command(monkeypatch: pytest.MonkeyPatch, run_command: CommandRunner, platform: str) -> None:
    """Forward arguments intact and preserve a failing native tool's exit code."""
    monkeypatch.setattr(sys, "platform", platform)
    execute = Mock(side_effect=SystemExit(7))
    run = Mock(return_value=Mock(returncode=7))
    monkeypatch.setattr(os, "execv", execute)
    monkeypatch.setattr(subprocess, "run", run)

    assert run_command(["bench", "input with spaces.qasm", "--help"]).code == 7
    suffix = ".exe" if platform == "win32" else ""
    executable = Path(str(distribution("mqt-core").locate_file(f"mqt/core/bin/mqt-core{suffix}")))
    arguments = [str(executable), "bench", "input with spaces.qasm", "--help"]
    if platform == "win32":
        run.assert_called_once_with(arguments, check=False)
    else:
        execute.assert_called_once_with(executable, arguments)


@pytest.mark.parametrize("destination", ["file", "stdout", "missing-directory"])
def test_qpy_export(
    monkeypatch: pytest.MonkeyPatch,
    run_command: CommandRunner,
    tmp_path: Path,
    capsysbinary: pytest.CaptureFixture[bytes],
    destination: str,
) -> None:
    """Compile in-process and independently decode QPY or report an output failure."""
    source = tmp_path / "input.qasm"
    source.write_text("OPENQASM 3; qubit q; x q;")
    output = tmp_path / "output.qpy"
    arguments = ["compile", str(source), "-o", str(output)]
    if destination == "stdout":
        arguments = ["compile", str(source), "--emit=qpy"]
    elif destination == "missing-directory":
        arguments[-1] = str(tmp_path / "absent" / "output.qpy")
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=AssertionError("QPY must compile in-process")))
    error = run_command(arguments)
    if destination == "missing-directory":
        assert "QPY export failed" in str(error)
    else:
        assert error.code == 0
        data = capsysbinary.readouterr().out if destination == "stdout" else output.read_bytes()
        assert qpy.get_qpy_version(BytesIO(data)) == 17
        circuit = qpy.load(BytesIO(data))[0]
        assert Statevector.from_instruction(circuit).probabilities_dict() == {"1": 1.0}


@pytest.mark.parametrize("failure", ["compiler", "serializer"])
def test_qpy_export_failure(
    monkeypatch: pytest.MonkeyPatch, run_command: CommandRunner, tmp_path: Path, failure: str
) -> None:
    """Compilation and serialization failures leave an existing destination intact."""
    source = tmp_path / "input.qasm"
    source.write_text("not OpenQASM" if failure == "compiler" else "OPENQASM 3; qubit q; x q;")
    if failure == "serializer":
        monkeypatch.setattr(qpy, "dump", Mock(side_effect=QpyError("Cannot serialize circuit")))
    output = tmp_path / "output.qpy"
    output.write_bytes(b"existing")
    assert run_command(["compile", str(source), "-o", str(output)]).code != 0
    assert output.read_bytes() == b"existing"


@pytest.mark.parametrize("input_format", ["openqasm", "qc", "qco", "jeff"])
def test_qpy_input_formats(run_command: CommandRunner, tmp_path: Path, input_format: str) -> None:
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
    assert run_command(arguments).code == 0
    assert Statevector(qpy.load(BytesIO(output.read_bytes()))[0]).probabilities_dict() == {"1": 1.0}


@pytest.mark.parametrize(
    ("arguments", "width"),
    [
        (["--decompose-multi-controlled"], 2),
        (["--decompose-multi-controlled=True"], 2),
        (["--decompose-multi-controlled="], 2),
        (["--decompose-multi-controlled=false"], 3),
        (["--decompose-multi-controlled", "--decompose-multi-controlled-min-qubits=4"], 3),
    ],
)
def test_qpy_decomposition(run_command: CommandRunner, tmp_path: Path, arguments: list[str], width: int) -> None:
    """Honor the decomposition threshold while retaining the circuit's action."""
    source = tmp_path / "input.qasm"
    source.write_text("OPENQASM 3; qubit[3] q; x q[0]; x q[1]; ctrl(2) @ x q[0],q[1],q[2];")
    output = tmp_path / "output.qpy"
    assert run_command(["compile", *arguments, str(source), "-o", str(output)]).code == 0
    circuit = qpy.load(BytesIO(output.read_bytes()))[0]
    assert max(item.operation.num_qubits for item in circuit.data) == width
    assert Statevector(circuit).data == pytest.approx(Statevector.from_label("111").data, abs=1e-14)


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("42", 42),
        ("0x2a", 42),
        ("0o52", 42),
        ("052", 42),
        ("0b101010", 42),
        (str(2**64 - 1), 2**64 - 1),
        ("08", None),
        ("+1", None),
        ("-1", None),
        ("1_0", None),
        (str(2**64), None),
    ],
)
def test_qpy_unsigned_options(
    monkeypatch: pytest.MonkeyPatch,
    run_command: CommandRunner,
    tmp_path: Path,
    text: str,
    value: int | None,
) -> None:
    """Retain LLVM's numeric spelling and range for seeds and mapping controls."""
    source = tmp_path / "input.qasm"
    source.write_text("OPENQASM 3; qubit q; x q;")
    compile_mock = Mock(return_value=b"qpy")
    monkeypatch.setattr(mlir, "compile_program", compile_mock)
    error = run_command(["compile", str(source), f"--seed={text}", "-o", str(tmp_path / "output.qpy")])
    if value is None:
        assert error.code != 0
        compile_mock.assert_not_called()
    else:
        assert error.code == 0
        assert compile_mock.call_args.kwargs["options"].seed == value


def test_qpy_custom_pipeline(run_command: CommandRunner, tmp_path: Path) -> None:
    """Run an explicitly requested pipeline through its CLI alias."""
    source = tmp_path / "input.qasm"
    source.write_text("OPENQASM 3; qubit q; gphase(0.37); x q;")
    output = tmp_path / "output.qpy"
    error = run_command(["compile", str(source), "-o", str(output), "--passes=builtin.module(drop-global-phases)"])
    assert error.code == 0
    assert Statevector(qpy.load(BytesIO(output.read_bytes()))[0]).data == pytest.approx([0, 1], abs=1e-14)


def test_qpy_rejects_native_debug_option(
    run_command: CommandRunner, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reject unsupported native debugging options before touching output."""
    output = tmp_path / "output.qpy"
    output.write_bytes(b"existing")
    assert run_command(["compile", "--mlir-print-ir-after-all", "-o", str(output)]).code != 0
    assert "unrecognized arguments: --mlir-print-ir-after-all" in capsys.readouterr().err
    assert output.read_bytes() == b"existing"


@pytest.mark.parametrize("gate_name", ["u3", "device_u"])
@pytest.mark.parametrize("input_format", ["openqasm", "qc", "qco"])
def test_qpy_target_export(run_command: CommandRunner, tmp_path: Path, gate_name: str, input_format: str) -> None:
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
    payload = (
        '#mqt.payload_spec<format = <id = "qir", version = "2.1.0", profile = "base", encoding = binary>, '
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
    assert run_command(["compile", str(source), "-o", str(output), *arguments]).code == 0
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


@pytest.mark.parametrize("arguments", [["--help"], ["-o", "output.qasm"]])
def test_native_output_passthrough(
    monkeypatch: pytest.MonkeyPatch, run_command: CommandRunner, arguments: list[str]
) -> None:
    """Native output and help bypass the optional QPY exporter."""
    monkeypatch.setattr(os, "execv", Mock(side_effect=SystemExit(7)))
    monkeypatch.setattr(subprocess, "run", Mock(return_value=Mock(returncode=7)))
    assert run_command(["compile", *arguments]).code == 7


def test_qpy_requires_optional_qiskit(
    monkeypatch: pytest.MonkeyPatch,
    run_command: CommandRunner,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing Qiskit produces an installation hint before invoking the compiler."""
    monkeypatch.setitem(sys.modules, "qiskit", None)
    assert run_command(["compile", "--emit=qpy"]).code == 1
    assert "install mqt-core[qiskit]" in capsys.readouterr().err
