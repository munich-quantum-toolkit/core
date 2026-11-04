# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the mqt-core CLI."""

from __future__ import annotations

import json
import shutil
import sys
from importlib.metadata import distribution
from pathlib import Path
from subprocess import check_output
from typing import TYPE_CHECKING

import pytest
from qiskit import qpy
from qiskit.quantum_info import Statevector

from mqt.core import __version__ as mqt_core_version
from mqt.core.mlir import CompilerTarget, QCProgram

if TYPE_CHECKING:
    from pytest_console_scripts import ScriptRunner

pytestmark = pytest.mark.script_launch_mode("subprocess")


@pytest.mark.parametrize("arguments", [[], ["--help"]])
def test_cli_help(script_runner: ScriptRunner, arguments: list[str]) -> None:
    """Show available options with no arguments or explicit help."""
    ret = script_runner.run(["mqt-core", *arguments])
    assert ret.success
    assert "mqt-core" in ret.stdout
    assert "--version" in ret.stdout
    assert "--include-dir" in ret.stdout
    assert "--cmake-dir" in ret.stdout


def test_cli_version(script_runner: ScriptRunner) -> None:
    """Test running the CLI with the --version argument."""
    ret = script_runner.run(["mqt-core", "--version"])
    assert ret.success
    assert mqt_core_version in ret.stdout


@pytest.mark.parametrize(
    ("option", "marker"), [("include-dir", "mqt_ddsim_qdmi"), ("cmake-dir", "mqt-core/mqt-core-config.cmake")]
)
def test_cli_installation_dir(script_runner: ScriptRunner, option: str, marker: str) -> None:
    """Locate the installed headers and CMake package."""
    ret = script_runner.run(["mqt-core", f"--{option}"])
    assert ret.success
    assert (Path(ret.stdout.strip()) / marker).exists()


def test_cli_relocated_installation(script_runner: ScriptRunner, tmp_path: Path) -> None:
    """Resolve installation paths after moving the package, and diagnose missing files."""
    package = Path(str(distribution("mqt-core").locate_file("mqt/core")))
    relocated = tmp_path / "relocated package"
    shutil.copytree(package, relocated)
    executable = relocated / "bin" / ("mqt-core.exe" if sys.platform == "win32" else "mqt-core")
    for option, relative in [("include-dir", "include/mqt-core"), ("cmake-dir", "share/cmake")]:
        path = relocated / relative
        result = script_runner.run([str(executable), f"--{option}"])
        assert result.success
        assert Path(result.stdout.strip()) == path
        shutil.rmtree(path)
        result = script_runner.run([str(executable), f"--{option}"])
        assert result.returncode == 1
        assert "Installation directory not found:" in result.stderr


def test_cli_execute_module() -> None:
    """Test running the CLI by executing the mqt-core module."""
    output = check_output([sys.executable, "-m", "mqt.core", "--version"])
    assert mqt_core_version in output.decode()


@pytest.mark.parametrize("subcommand", [[], ["compile"]])
def test_compiler_cli(script_runner: ScriptRunner, tmp_path: Path, subcommand: list[str]) -> None:
    """Compile OpenQASM with the compiler bundled in the wheel."""
    source = tmp_path / "bell.qasm"
    source.write_text('OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; h q[0]; cx q[0], q[1];')
    ret = script_runner.run(["mqt-core", *subcommand, str(source), "--emit=qco"])
    assert ret.success
    assert "qco.h" in ret.stdout


def test_compiler_cli_serialized_target(script_runner: ScriptRunner, tmp_path: Path) -> None:
    """Compile with a Python target serialized into a CLI argument."""
    target = CompilerTarget(
        'saved "target"',
        2,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations.unrestricted(),
    )
    source = tmp_path / "bell.qasm"
    source.write_text('OPENQASM 3.1; include "stdgates.inc"; qubit[2] q; h q[0]; cx q[0], q[1];')
    ret = script_runner.run([
        "mqt-core",
        str(source),
        f"--target={target}",
        (
            '--payload-spec=#mqt.payload_spec<format = <id = "openqasm", version = "3.1.0", '
            'profile = "", encoding = text>, capabilities = [<id = "forward-branching", '
            "value = 0, constraints = []>], optional_capabilities_known = false>"
        ),
    ])
    assert ret.success
    assert ret.stdout.startswith("OPENQASM 3;")
    assert QCProgram.from_openqasm_str(ret.stdout).is_valid


def test_benchmark_cli(script_runner: ScriptRunner) -> None:
    """Run the bundled benchmark driver through its console script."""
    ret = script_runner.run(["mqt-core", "bench", "list"])
    assert ret.success
    assert '"modular-multiplier"' in ret.stdout
    assert '"ghz"' in ret.stdout
    assert '"grover"' in ret.stdout
    assert '"multiplexer"' in ret.stdout
    assert '"qft-adder"' in ret.stdout
    assert '"qpe"' in ret.stdout
    assert '"repeat-until-success"' in ret.stdout
    assert '"teleportation"' in ret.stdout


@pytest.mark.parametrize(
    "input_format", ["openqasm2", "openqasm3", "qir-base.ll", "qir-base.bc", "qir-adaptive.ll", "qir-adaptive.bc"]
)
def test_simulate_cli(script_runner: ScriptRunner, tmp_path: Path, input_format: str) -> None:
    """Sample OpenQASM and both QIR profiles through the isolated DDSIM worker."""
    source = tmp_path / "bell.qasm"
    if input_format == "openqasm2":
        source.write_text(
            'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;'
        )
    else:
        source.write_text(
            'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; bit[2] c; h q[0]; cx q[0],q[1]; c = measure q;'
        )
    if input_format.startswith("qir"):
        profile, extension = input_format.split(".")
        output = tmp_path / f"bell.{extension}"
        result = script_runner.run(["mqt-core", "compile", str(source), f"--emit={profile}", "-o", str(output)])
        assert result.success
        source = output
    arguments = ["mqt-core", "simulate", str(source), "--shots=128", "--seed=42"]
    result = script_runner.run(arguments)
    assert result.success
    counts = json.loads(result.stdout)["counts"]
    assert set(counts) == {"00", "11"}
    assert sum(counts.values()) == 128
    assert script_runner.run(arguments).stdout == result.stdout


@pytest.mark.parametrize("source_text", ["not LLVM IR", 'define void @main() "entry_point" { ret void }'])
def test_simulate_cli_errors(script_runner: ScriptRunner, tmp_path: Path, source_text: str) -> None:
    """Report invalid shots and invalid input with a failing exit status."""
    source = tmp_path / "invalid.ll"
    source.write_text(source_text)
    for arguments in [["--shots=0"], []]:
        result = script_runner.run(["mqt-core", "simulate", str(source), *arguments])
        assert not result.success
        assert result.stderr


def test_run_installed_provider(script_runner: ScriptRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Discover a provider from wheel metadata and compile/run through its device ID."""
    package = Path(str(distribution("mqt-core").locate_file("mqt/core")))
    manifest = next(package.rglob("*ddsim*.qdmi.json"))
    device = json.loads(manifest.read_text())["qdmi"]["devices"][0]
    device["id"] = "cli.test.device"
    device["library"] = str((manifest.parent / device["library"]).resolve())
    provider = tmp_path / "cli_provider"
    provider.mkdir()
    (provider / "__init__.py").write_text("raise RuntimeError('provider must not be imported')")
    (provider / "device.qdmi.json").write_text(json.dumps({"schema-version": 1, "qdmi": {"devices": [device]}}))
    metadata = tmp_path / "cli_provider-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: cli-provider\nVersion: 1.0\n")
    (metadata / "entry_points.txt").write_text("[mqt.core.qdmi.manifests]\ncli = cli_provider\n")
    (metadata / "RECORD").write_text("cli_provider/device.qdmi.json,,\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    result = script_runner.run(["mqt-core", "compile", "--qdmi-list-devices"])
    assert result.success
    assert "cli.test.device" in result.stdout
    source = tmp_path / "bell.qasm"
    source.write_text('OPENQASM 3; include "stdgates.inc"; qubit[2] q; bit[2] c; h q[0]; cx q[0],q[1]; c=measure q;')
    assert not script_runner.run(["mqt-core", "run", str(source)]).success
    compiled = tmp_path / "bell.bc"
    result = script_runner.run([
        "mqt-core",
        "compile",
        str(source),
        "--qdmi-device=cli.test.device",
        "-o",
        str(compiled),
    ])
    assert result.success
    for program in [source, compiled]:
        result = script_runner.run(["mqt-core", "run", str(program), "--qdmi-device=cli.test.device", "--shots=128"])
        assert result.success
        counts = json.loads(result.stdout)["counts"]
        assert set(counts) == {"00", "11"}
        assert sum(counts.values()) == 128


def test_compile_qpy_and_jeff(script_runner: ScriptRunner, tmp_path: Path) -> None:
    """Export QPY with Qiskit and execute a jeff round trip."""
    source = tmp_path / "bell.qasm"
    source.write_text('OPENQASM 3; include "stdgates.inc"; qubit[2] q; bit[2] c; h q[0]; cx q[0],q[1]; c=measure q;')
    output = tmp_path / "bell.qpy"
    result = script_runner.run(["mqt-core", "compile", str(source), "-o", str(output)])
    assert result.success
    with output.open("rb") as stream:
        circuit = qpy.load(stream)[0]
    circuit.remove_final_measurements()
    probabilities = Statevector.from_instruction(circuit).probabilities_dict()
    assert probabilities == pytest.approx({"00": 0.5, "11": 0.5})
    output = tmp_path / "bell.jeff"
    result = script_runner.run(["mqt-core", "compile", str(source), "-o", str(output)])
    assert result.success
    result = script_runner.run(["mqt-core", "simulate", str(output), "--shots=128"])
    assert result.success
    assert sum(json.loads(result.stdout)["counts"].values()) == 128


@pytest.mark.parametrize("suffix", ["ll", "bc", "qasm"])
@pytest.mark.parametrize("target", [[], ["--qdmi-device=mqt.ddsim.default"]])
def test_inferred_output(script_runner: ScriptRunner, tmp_path: Path, suffix: str, target: list[str]) -> None:
    """Infer executable formats and preserve the requested QIR encoding."""
    source = tmp_path / "input.qasm"
    source.write_text('OPENQASM 3; include "stdgates.inc"; qubit q; bit c; x q; c=measure q;')
    output = tmp_path / f"output.{suffix}"
    result = script_runner.run(["mqt-core", "compile", str(source), *target, "-o", str(output)])
    assert result.success
    if suffix == "bc":
        assert output.read_bytes().startswith(b"BC\xc0\xde")
    elif suffix == "ll":
        assert '"qir_profiles"="adaptive_profile"' in output.read_text()
    else:
        assert output.read_text().startswith("OPENQASM 3")
    result = script_runner.run(["mqt-core", "simulate", str(output), "--shots=16"])
    assert result.success
    assert json.loads(result.stdout)["counts"] == {"1": 16}


def test_simulate_logical_readout(script_runner: ScriptRunner, tmp_path: Path) -> None:
    """Restore reordered and repeated classical outputs after source compilation."""
    source = tmp_path / "readout.qasm"
    source.write_text(
        "OPENQASM 3; qubit[2] q; bit[3] c; x q[0]; c[2]=measure q[0]; c[0]=measure q[0]; c[1]=measure q[1];"
    )
    result = script_runner.run(["mqt-core", "simulate", str(source), "--shots=16"])
    assert result.success
    assert json.loads(result.stdout)["counts"] == {"101": 16}
