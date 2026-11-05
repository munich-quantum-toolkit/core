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

from mqt.core.mlir import CompilerTarget


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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsysbinary: pytest.CaptureFixture[bytes], destination: str
) -> None:
    """Run the real compiler and independently decode QPY or report an output failure."""
    source = tmp_path / "input.qasm"
    source.write_text("OPENQASM 3; qubit q; x q;")
    output = tmp_path / "output.qpy"
    arguments = ["mqt-core", "compile", str(source), "-o", str(output)]
    if destination == "stdout":
        arguments = ["mqt-core", "compile", str(source), "--emit=qpy"]
    elif destination == "missing-directory":
        arguments[-1] = str(tmp_path / "absent" / "output.qpy")
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as error:
        distribution("mqt-core").entry_points["mqt-core"].load()()
    if destination == "missing-directory":
        assert "QPY export failed" in str(error.value)
    else:
        assert error.value.code == 0
        data = capsysbinary.readouterr().out if destination == "stdout" else output.read_bytes()
        assert qpy.get_qpy_version(BytesIO(data)) == 17
        circuit = qpy.load(BytesIO(data))[0]
        assert Statevector.from_instruction(circuit).probabilities_dict() == {"1": 1.0}


@pytest.mark.parametrize("failure", ["compiler", "serializer"])
def test_qpy_export_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str) -> None:
    """Compilation and serialization failures leave an existing destination intact."""
    source = tmp_path / "input.qasm"
    source.write_text("not OpenQASM" if failure == "compiler" else "OPENQASM 3; qubit q; x q;")
    if failure == "serializer":
        monkeypatch.setattr(qpy, "dump", Mock(side_effect=ValueError("Cannot serialize circuit")))
    output = tmp_path / "output.qpy"
    output.write_bytes(b"existing")
    monkeypatch.setattr(sys, "argv", ["mqt-core", "compile", str(source), "-o", str(output)])
    with pytest.raises(SystemExit) as error:
        distribution("mqt-core").entry_points["mqt-core"].load()()
    assert error.value.code != 0
    assert output.read_bytes() == b"existing"


@pytest.mark.parametrize("gate_name", ["u3", "device_u"])
def test_qpy_target_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gate_name: str) -> None:
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
    source = tmp_path / "input.qasm"
    source.write_text("OPENQASM 3; qubit q; x q;")
    output = tmp_path / "output.qpy"
    monkeypatch.setattr(
        sys,
        "argv",
        ["mqt-core", "compile", str(source), f"--target={target}", f"--payload-spec={payload}", "-o", str(output)],
    )
    with pytest.raises(SystemExit) as error:
        distribution("mqt-core").entry_points["mqt-core"].load()()
    assert error.value.code == 0
    with output.open("rb") as stream:
        restored = qpy.load(stream)[0]
    assert restored.count_ops() == {gate_name: 1}
    assert restored.num_qubits == 6
    assert restored.layout is not None
    placement = restored.layout.initial_index_layout(filter_ancillas=True)
    assert len(placement) == 1
    assert placement[0] in {2, 5}
    assert restored.find_bit(restored.data[0].qubits[0]).index == placement[0]
    expected = Statevector.from_int(1 << placement[0], 2**restored.num_qubits)
    assert Statevector.from_instruction(restored).data == pytest.approx(expected.data, abs=1e-14)


@pytest.mark.parametrize("arguments", [["--help"], ["-o", "output.qasm"]])
def test_native_output_passthrough(monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    """Native output and help bypass the optional QPY exporter."""
    monkeypatch.setattr(sys, "argv", ["mqt-core", "compile", *arguments])
    monkeypatch.setattr(os, "execv", Mock(side_effect=SystemExit(7)))
    monkeypatch.setattr(subprocess, "run", Mock(return_value=Mock(returncode=7)))
    with pytest.raises(SystemExit) as error:
        distribution("mqt-core").entry_points["mqt-core"].load()()
    assert error.value.code == 7


def test_qpy_requires_optional_qiskit(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Missing Qiskit produces an installation hint before invoking the compiler."""
    monkeypatch.setitem(sys.modules, "qiskit", None)
    monkeypatch.setattr(sys, "argv", ["mqt-core", "compile", "--emit=qpy"])
    with pytest.raises(SystemExit) as error:
        distribution("mqt-core").entry_points["mqt-core"].load()()
    assert error.value.code == 1
    assert "install mqt-core[qiskit]" in capsys.readouterr().err
