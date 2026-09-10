# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Device-directed compilation and submission contracts."""

from __future__ import annotations

import gc
import subprocess
import sys
from fractions import Fraction
from typing import TYPE_CHECKING

import pytest

from mqt.core.bench import qpe, repeat_until_success
from mqt.core.mlir import CompiledProgram, CompilerTarget, OutputFormat, compile_program, submit_program
from mqt.core.qdmi import Job, ProgramFormat
from mqt.core.qdmi.driver import open_device

if TYPE_CHECKING:
    from pathlib import Path

BELL = 'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; bit[2] c; h q[0]; cx q[0],q[1]; c = measure q;'


@pytest.mark.parametrize("method", [qpe.Method.STANDARD, qpe.Method.ITERATIVE])
def test_qpe_device_execution(method: qpe.Method) -> None:
    """Compile structured QPE and recover its exact phase through QIR execution."""
    benchmark = qpe.QPE(qpe.Options(precision=8, phase=Fraction(3, 8), method=method))
    device = open_device("mqt.ddsim.default")
    compiled = compile_program(benchmark.generate(), target=device)
    job = submit_program(compiled, target=device, num_shots=32, custom1=17)
    job.wait()
    # QIR records register bits in increasing index order; benchmarks use big endian.
    counts = {bits[::-1]: count for bits, count in job.get_counts().items()}
    assert counts == {"01100000": 32}
    assert benchmark.evaluate(counts).total_variation_distance == pytest.approx(0)


@pytest.mark.parametrize("data_qubits", [1, 4])
def test_repeat_until_success_device_execution(data_qubits: int) -> None:
    """Preserve phase-sensitive RUS results through target placement and QIR."""
    benchmark = repeat_until_success.RepeatUntilSuccess(repeat_until_success.Options(data_qubits=data_qubits))
    device = open_device("mqt.ddsim.default")
    compiled = compile_program(benchmark.generate(), target=device)
    job = submit_program(compiled, target=device, num_shots=4096, custom1=17)
    job.wait()
    assert benchmark.evaluate(job.get_counts()).total_variation_distance < 0.02


@pytest.mark.parametrize("form", ["artifact", "source", "device_id"])
def test_submission_forms(form: str) -> None:
    """All forms default to 1024 shots and preserve seeded samples."""
    device = open_device("mqt.ddsim.default")
    if form == "artifact":
        program = compile_program(BELL, target=device)
        assert isinstance(program, CompiledProgram)
        assert program.program_format == ProgramFormat.QIR_ADAPTIVE_MODULE
        job = submit_program(program, target=device, custom1=7)
    elif form == "source":
        job = submit_program(BELL, target=device, custom1=7)
    else:
        job = submit_program(BELL, target="mqt.ddsim.default", custom1=7)
    assert isinstance(job, Job)
    del device
    gc.collect()
    job.wait()
    shots = job.get_shots()
    assert len(shots) == 1024
    assert set(shots) <= {"00", "11"}
    again = submit_program(BELL, target="mqt.ddsim.default", custom1=7)
    again.wait()
    assert again.get_shots() == shots


@pytest.mark.parametrize(
    "program_format",
    [
        ProgramFormat.QIR_ADAPTIVE_MODULE,
        ProgramFormat.QIR_ADAPTIVE_STRING,
        ProgramFormat.QASM3,
        ProgramFormat.QIR_BASE_MODULE,
        ProgramFormat.QIR_BASE_STRING,
    ],
)
def test_compiled_formats(program_format: ProgramFormat) -> None:
    """Compiled programs keep their selected format when the device is reopened."""
    device = open_device("mqt.ddsim.default")
    by_device = compile_program(BELL, target=device, program_format=program_format)
    by_id = compile_program(BELL, target="mqt.ddsim.default", program_format=program_format)
    assert by_id.payload == by_device.payload
    assert by_id.program_format == program_format
    if program_format == ProgramFormat.QASM3:
        assert by_id.payload_specification.format.version == "3.1.0"
        assert isinstance(by_id.payload, str)
        assert by_id.payload.startswith("OPENQASM 3.1;")
    with pytest.raises(AttributeError):
        by_id.payload = b"changed"  # ty: ignore[invalid-assignment]
    del device
    gc.collect()
    job = submit_program(by_id, target=open_device("mqt.ddsim.default"), num_shots=0)
    job.wait()
    assert job.get_dense_statevector() == pytest.approx([2**-0.5, 0, 0, 2**-0.5])


def test_rejects_invalid_submission_options() -> None:
    """Reject invalid counts and formats before consuming a typed program."""
    device = open_device("mqt.ddsim.default")
    source = compile_program(BELL, output=OutputFormat.QCO)
    with pytest.raises(ValueError, match="nonnegative"):
        submit_program(source, target=device, num_shots=-1)
    with pytest.raises(ValueError, match="cannot emit"):
        compile_program(source, target=device, program_format=ProgramFormat.QASM2, inplace=True)
    assert source.is_valid
    compiled = compile_program(source, target=device)
    assert source.is_valid
    with pytest.raises(ValueError, match="conflicts"):
        submit_program(compiled, target=device, program_format=ProgramFormat.QASM3)


def test_explicit_target_requires_output_and_matching_contract() -> None:
    """Hardware snapshots need a chosen output and must match before submission."""
    device = open_device("mqt.ddsim.default")
    target = CompilerTarget.from_device(device)
    with pytest.raises(ValueError, match="requires output"):
        compile_program(BELL, target=target)  # ty: ignore[invalid-argument-type]
    matching = compile_program(BELL, target=target, program_format=ProgramFormat.QASM3)
    job = submit_program(matching, target=device, num_shots=8)
    job.wait()
    assert len(job.get_shots()) == 8
    other = CompilerTarget(
        2,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations.unrestricted(),
    )
    mismatch = compile_program(BELL, target=other, program_format=ProgramFormat.QASM3)
    with pytest.raises(ValueError, match="recompile"):
        submit_program(mismatch, target=device)


def test_qdmi_does_not_import_compiler() -> None:
    """Raw QDMI submission does not load the compiler module."""
    script = """
import sys
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device

assert "mqt.core.mlir" not in sys.modules
device = open_device("mqt.ddsim.default")
source = "OPENQASM 3.0; qubit q; bit c = measure q;"
raw = device.submit_job(source, ProgramFormat.QASM3, num_shots=1)
assert raw.wait()
assert "mqt.core.mlir" not in sys.modules
from mqt.core.mlir import submit_program
job = submit_program(source, target=device, num_shots=2)
assert job.wait()
assert job.get_counts() == {"0": 2}
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # ruff: ignore[subprocess-without-shell-equals-true]


def test_source_path_is_read_once(tmp_path: Path) -> None:
    """Source submission imports once; artifact resubmission needs no source."""
    path = tmp_path / "bell.qasm"
    path.write_text(BELL)

    class Source:
        calls = 0

        def __fspath__(self) -> str:
            self.calls += 1
            assert self.calls == 1
            return str(path)

    source = Source()
    device = open_device("mqt.ddsim.default")
    job = submit_program(source, target=device, num_shots=2)
    job.wait()
    assert source.calls == 1
    compiled = compile_program(path, target=device)
    path.unlink()
    job = submit_program(compiled, target=device, num_shots=2)
    job.wait()
    assert len(job.get_shots()) == 2


@pytest.mark.parametrize("program_format", [ProgramFormat.QASM3, ProgramFormat.QIR_ADAPTIVE_MODULE])
def test_payload_forward_branching(program_format: ProgramFormat) -> None:
    """Guaranteed forward branching permits feedback and Base rejects it."""
    source = BELL.replace("c = measure q;", "c[0] = measure q[0]; if (c[0]) { x q[1]; } c[1] = measure q[1];")
    device = open_device("mqt.ddsim.default")
    compiled = compile_program(source, target=device, program_format=program_format)
    job = submit_program(compiled, target=device, num_shots=32)
    job.wait()
    assert set(job.get_counts()) <= ({"00", "01"} if program_format == ProgramFormat.QASM3 else {"00", "10"})
    with pytest.raises(RuntimeError, match="Not supported"):
        job.get_dense_statevector()
    assert len(job.get_shots()) == 32
    with pytest.raises(ValueError, match="Compilation failed"):
        compile_program(source, target=device, program_format=ProgramFormat.QIR_BASE_MODULE)


@pytest.mark.parametrize("program_format", [ProgramFormat.QASM3, ProgramFormat.QIR_BASE_MODULE])
def test_compiled_terminal_sampling_retains_state(program_format: ProgramFormat) -> None:
    """Eligible compiled payloads expose all lazy results without changing shots."""
    job = submit_program(BELL, target="mqt.ddsim.default", num_shots=64, program_format=program_format, custom1=7)
    job.wait()
    shots = job.get_shots()
    for _ in range(2):
        assert job.get_dense_statevector() == pytest.approx([2**-0.5, 0, 0, 2**-0.5])
        assert job.get_sparse_statevector() == pytest.approx({"00": 2**-0.5, "11": 2**-0.5})
        assert job.get_dense_probabilities() == pytest.approx([0.5, 0, 0, 0.5])
        assert job.get_sparse_probabilities() == pytest.approx({"00": 0.5, "11": 0.5})
    assert job.get_shots() == shots


def test_adaptive_payload_supports_optional_computations() -> None:
    """DDSIM accepts optional computations in the preferred Adaptive payload."""
    source = (
        'OPENQASM 3.0; include "stdgates.inc"; qubit q; bit c; h q; c = measure q; '
        "if (true) { int[32] k = int[32](c); if (k + 1 == 2) { x q; } } c = measure q;"
    )
    device = open_device("mqt.ddsim.default")
    compiled = compile_program(source, target=device)
    assert compiled.program_format == ProgramFormat.QIR_ADAPTIVE_MODULE
    assert "qir.int-computations" in {c.capability_id for c in compiled.payload_specification.capabilities}
    job = submit_program(compiled, target=device, num_shots=16)
    job.wait()
    assert job.get_counts() == {"0": 16}


def test_loop_is_supported_by_default_payload() -> None:
    """The preferred Adaptive payload permits the compiler to retain loops."""
    source = 'OPENQASM 3.0; include "stdgates.inc"; qubit q; bit c; for int i in [0:2] { x q; } c = measure q;'
    job = submit_program(source, target="mqt.ddsim.default", num_shots=8)
    job.wait()
    assert job.get_counts() == {"1": 8}


def test_default_payload_supports_measurement_controlled_loop() -> None:
    """Conditional loops use the explicitly advertised Adaptive capability."""
    source = (
        'OPENQASM 3.0; include "stdgates.inc"; qubit q; bit c; x q; c = measure q; while (c) { x q; c = measure q; }'
    )
    job = submit_program(source, target="mqt.ddsim.default", num_shots=8)
    job.wait()
    assert job.get_counts() == {"0": 8}


def test_qir_artifact_rejects_unsupported_entry_result() -> None:
    """Keep scalar program outputs distinct from the QDMI QIR status return."""
    source = "OPENQASM 3.0; qubit q; bit c = measure q; int[32] k = int[32](c);"
    with pytest.raises(ValueError, match=r"i64 \(\) entry point"):
        compile_program(source, target="mqt.ddsim.default")


@pytest.mark.parametrize("num_shots", [0, 8])
def test_empty_source_has_a_successful_entry_point(num_shots: int) -> None:
    """An output-free source receives a success status without inventing outputs."""
    job = submit_program("OPENQASM 3.0;", target="mqt.ddsim.default", num_shots=num_shots)
    job.wait()
    assert job.check() == Job.Status.DONE
    assert job.get_dense_statevector() == [1 + 0j]


def test_openqasm31_switch() -> None:
    """The OpenQASM payload supports measurement-controlled switch statements."""
    source = """OPENQASM 3.1;
include "stdgates.inc";
qubit q;
bit c;
h q;
c = measure q;
switch (int(c)) { case 1 { x q; } default {} }
c = measure q;
"""
    job = submit_program(source, target="mqt.ddsim.default", program_format=ProgramFormat.QASM3, num_shots=16)
    job.wait()
    assert job.get_counts() == {"0": 16}
