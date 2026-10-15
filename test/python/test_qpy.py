# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QPY interoperability through real Qiskit circuits and Core programs."""

from __future__ import annotations

import subprocess
import sys
from io import BytesIO, StringIO
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import pytest
import qiskit
from qiskit import QuantumCircuit
from qiskit import qpy as qiskit_qpy
from qiskit.circuit import IfElseOp, ParameterVector
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.qpy import formats
from qiskit.quantum_info import Operator
from qiskit_support import supports_qiskit_translation

from mqt.core import qpy
from mqt.core.mlir import CompilerTarget, QCProgram

if TYPE_CHECKING:
    from pathlib import Path

requires_translation = pytest.mark.skipif(
    not supports_qiskit_translation(), reason=f"No registered Qiskit adapter for {qiskit.__version__}"
)


@requires_translation
@pytest.mark.parametrize("dialect", ["qc", "qco"])
def test_qpy_paths_preserve_parameters_phase_and_sources(tmp_path: Path, dialect: str) -> None:
    """Qiskit can read exported programs and bind the original parameter identities."""
    angles = ParameterVector("angles", 2)
    circuit = QuantumCircuit(2, global_phase=0.25)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(angles[0] + 2 * angles[1], 1)
    input_path = tmp_path / "input.qpy"
    with input_path.open("wb") as stream:
        qiskit_qpy.dump(circuit, stream)
    program = qpy.load(str(input_path))[0]
    if dialect == "qco":
        program = program.to_qco()
    original_ir = program.ir
    output_path = tmp_path / "output.qpy"
    qpy.dump(program, output_path)
    assert program.is_valid
    assert program.ir == original_ir
    with output_path.open("rb") as stream:
        restored = qiskit_qpy.load(stream)[0]
    bindings = {angles[0]: 0.125, angles[1]: -0.75}
    assert set(restored.parameters) == set(circuit.parameters)
    np.testing.assert_allclose(
        Operator(restored.assign_parameters(bindings)).data,
        Operator(circuit.assign_parameters(bindings)).data,
        atol=1e-12,
    )


@requires_translation
def test_qpy_streams_preserve_order_control_flow_and_ownership() -> None:
    """Multiple circuits survive without closing streams or dropping dynamic control."""
    first = QuantumCircuit(1)
    first.h(0)
    dynamic = QuantumCircuit(2, 1)
    dynamic.measure(0, 0)
    with dynamic.if_test((dynamic.clbits[0], 1)):
        dynamic.x(1)
    source = BytesIO()
    qiskit_qpy.dump([first, dynamic], source)
    prefixed = BytesIO(b"prefix" + source.getvalue())
    prefixed.seek(6)
    programs = qpy.load(prefixed)
    assert not prefixed.closed
    assert len(programs) == 2
    output = BytesIO(b"prefix")
    output.seek(6)
    qpy.dump(iter(programs), output)
    assert not output.closed
    assert output.getvalue().startswith(b"prefixQISKIT")
    restored = qiskit_qpy.load(BytesIO(output.getvalue()[6:]))
    np.testing.assert_allclose(Operator(restored[0]).data, Operator(first).data, atol=1e-12)
    assert restored[1].count_ops() == {"measure": 1, "if_else": 1}
    branch = restored[1].data[-1].operation
    assert isinstance(branch, IfElseOp)
    assert branch.blocks[0].count_ops() == {"x": 1}
    assert all(program.is_valid for program in programs)


def test_empty_archive() -> None:
    """An empty archive remains empty rather than creating a placeholder circuit."""
    stream = BytesIO()
    qpy.dump([], stream)
    stream.seek(0)
    assert qpy.load(stream) == []


@pytest.mark.parametrize("payload", [b"", b"truncated", b"QISKIT\xff", b"garbage" * 10])
def test_invalid_qpy_is_rejected(payload: bytes) -> None:
    """Malformed data and future format versions produce a QPY-specific diagnostic."""
    with pytest.raises(ValueError, match="Invalid or unsupported QPY input"):
        qpy.load(BytesIO(payload))


def test_invalid_qpy_program_type_is_rejected() -> None:
    """A corrupt program type in an otherwise valid archive is malformed input."""
    stream = BytesIO()
    qiskit_qpy.dump(QuantumCircuit(1), stream, version=13)
    payload = bytearray(stream.getvalue())
    payload[formats.FILE_HEADER_V10_SIZE] = ord("z")
    with pytest.raises(ValueError, match="Invalid or unsupported QPY input") as raised:
        qpy.load(BytesIO(payload))
    assert isinstance(raised.value.__cause__, TypeError)


@pytest.mark.parametrize("version", [12, 13, qiskit_qpy.QPY_VERSION])
@pytest.mark.parametrize("corruption", ["qubit_index", "gate_name"])
def test_corrupt_qpy_instruction_is_rejected(version: int, corruption: str) -> None:
    """Invalid operands and instructions fail in both Python and native QPY readers."""
    circuit = QuantumCircuit(1, name="corrupt_instruction")
    circuit.x(0)
    stream = BytesIO()
    qiskit_qpy.dump(circuit, stream, version=max(version, 13))
    payload = bytearray(stream.getvalue())
    # This parameter-free circuit has the same layout in QPY 12 and 13.
    payload[6] = version
    assert qiskit_qpy.load(BytesIO(payload)) == [circuit]
    original, replacement = (
        (b"XGateq\x00\x00\x00\x00", b"XGateq\x00\x00\x00\x01") if corruption == "qubit_index" else (b"XGate", b"Zzzzz")
    )
    assert payload.count(original) == 1
    source = BytesIO(payload.replace(original, replacement))
    with pytest.raises(ValueError, match="Invalid or unsupported QPY input"):
        qpy.load(source)
    assert not source.closed


@pytest.mark.parametrize("stage", ["decode", "translate"])
def test_native_qpy_panic_is_recoverable(monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    """PyO3 panics become ordinary errors at either native boundary, retaining their cause."""
    panic_type = type("PanicException", (BaseException,), {"__module__": "pyo3_runtime"})
    error = panic_type("native Qiskit failure")
    source = BytesIO()
    qiskit_qpy.dump(QuantumCircuit(1), source, version=13)
    source.seek(0)

    def fail(value: object) -> NoReturn:  # ruff: ignore[unused-function-argument]
        raise error

    if stage == "decode":
        monkeypatch.setattr(qiskit_qpy, "load", fail)
    else:
        monkeypatch.setattr(QCProgram, "from_qiskit", fail)
    with pytest.raises(ValueError, match="Invalid or unsupported QPY input") as raised:
        qpy.load(source)
    assert raised.value.__cause__ is error
    assert not source.closed


@pytest.mark.parametrize("stage", ["decode", "translate"])
@pytest.mark.parametrize(
    "error_type",
    [
        KeyboardInterrupt,
        SystemExit,
        GeneratorExit,
        MemoryError,
        RuntimeError,
        OSError,
        ImportError,
        type("PanicException", (BaseException,), {"__module__": __name__}),
    ],
)
def test_unrelated_qpy_errors_propagate(
    monkeypatch: pytest.MonkeyPatch, stage: str, error_type: type[BaseException]
) -> None:
    """The native panic guard must not relabel interrupts, resource errors, or other failures."""
    error = error_type("not malformed QPY")
    source = BytesIO()
    qiskit_qpy.dump(QuantumCircuit(1), source, version=13)
    source.seek(0)

    def fail(value: object) -> NoReturn:  # ruff: ignore[unused-function-argument]
        raise error

    if stage == "decode":
        monkeypatch.setattr(qiskit_qpy, "load", fail)
    else:
        monkeypatch.setattr(QCProgram, "from_qiskit", fail)
    with pytest.raises(error_type) as raised:
        qpy.load(source)
    assert raised.value is error
    assert not source.closed


@pytest.mark.parametrize("error_type", [ValueError, TypeError, LookupError, AttributeError])
def test_qpy_translation_errors_are_preserved(monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]) -> None:
    """Decoder-specific error normalization does not change ordinary compiler errors."""
    error = error_type("circuit translation failed")
    source = BytesIO()
    qiskit_qpy.dump(QuantumCircuit(1), source, version=13)
    source.seek(0)

    def fail(value: object) -> NoReturn:  # ruff: ignore[unused-function-argument]
        raise error

    monkeypatch.setattr(QCProgram, "from_qiskit", fail)
    with pytest.raises(error_type) as raised:
        qpy.load(source)
    assert raised.value is error
    assert not source.closed


def test_text_stream_is_rejected_as_caller_error() -> None:
    """A text stream is an API type error, not a malformed binary archive."""
    with pytest.raises(TypeError):
        qpy.load(StringIO("not binary"))  # ty: ignore[invalid-argument-type] Invalid public input.


@requires_translation
def test_unsupported_circuit_in_archive_fails() -> None:
    """A valid archive with a later unsupported circuit must not return a partial list."""
    supported = QuantumCircuit(1)
    supported.h(0)
    unsupported = QuantumCircuit(1)
    unsupported.delay(10, 0)
    stream = BytesIO()
    qiskit_qpy.dump([supported, unsupported], stream)
    stream.seek(0)
    with pytest.raises(RuntimeError, match=r"[Dd]elay|unsupported"):
        qpy.load(stream)
    assert not stream.closed


@requires_translation
@pytest.mark.parametrize("version", range(qiskit_qpy.QPY_COMPATIBILITY_VERSION, qiskit_qpy.QPY_VERSION + 1))
def test_explicit_qpy_version(version: int) -> None:
    """The requested supported format version reaches Qiskit."""
    program = QCProgram.from_qiskit(QuantumCircuit(1))
    stream = BytesIO()
    qpy.dump(program, stream, version=version)
    stream.seek(0)
    assert qiskit_qpy.get_qpy_version(stream) == version
    assert len(qpy.load(stream)) == 1


@requires_translation
def test_qpy_export_maps_target_sites() -> None:
    """Target-aware export maps sparse physical site IDs into dense Qiskit indices."""
    target = CompilerTarget(
        "sparse target",
        [CompilerTarget.Site(10), CompilerTarget.Site(20)],
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations.unrestricted(),
    )
    program = QCProgram.from_mlir_str("""module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qc.static 20 : !qc.qubit
    qc.x %q : !qc.qubit
    return
  }
}""")
    stream = BytesIO()
    qpy.dump(program, stream, target=target)
    stream.seek(0)
    restored = qiskit_qpy.load(stream)[0]
    expected = QuantumCircuit(2)
    expected.x(1)
    np.testing.assert_allclose(Operator(restored).data, Operator(expected).data, atol=1e-12)


@requires_translation
def test_unsupported_qiskit_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """QPY loading cannot bypass the compiler bridge's version guard."""
    stream = BytesIO()
    qiskit_qpy.dump(QuantumCircuit(1), stream)
    stream.seek(0)
    monkeypatch.setattr(qiskit, "__version__", "2.6.0")
    with pytest.raises(RuntimeError, match=r"installed version '2\.6\.0'"):
        qpy.load(stream)


@pytest.mark.parametrize("version", [qiskit_qpy.QPY_COMPATIBILITY_VERSION, qiskit_qpy.QPY_VERSION])
def test_truncated_circuit_payload(version: int) -> None:
    """A valid header cannot hide a truncated circuit payload."""
    stream = BytesIO()
    qiskit_qpy.dump(QuantumCircuit(2), stream, version=version)
    with pytest.raises(ValueError, match="Invalid or unsupported QPY input"):
        qpy.load(BytesIO(stream.getvalue()[:-1]))


def test_invalid_export_type_does_not_write() -> None:
    """Reject objects outside the program API before touching the output."""
    stream = BytesIO(b"keep me")
    with pytest.raises(TypeError, match="QCProgram or QCOProgram"):
        qpy.dump([object()], stream)  # ty: ignore[invalid-argument-type] Invalid public input.
    assert stream.getvalue() == b"keep me"


def test_qpy_dependency_errors_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing legacy decoder dependency must not be mislabeled as corrupt input."""
    error = MissingOptionalLibraryError("symengine", "QPY load", "pip install symengine")

    def missing_dependency(stream: object) -> NoReturn:  # ruff: ignore[unused-function-argument]
        raise error

    monkeypatch.setattr(qiskit_qpy, "load", missing_dependency)
    with pytest.raises(MissingOptionalLibraryError) as raised:
        qpy.load(BytesIO())
    assert raised.value is error


def test_short_write_is_not_reported_as_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stream that accepts only part of a payload must fail explicitly."""
    stream = BytesIO()
    monkeypatch.setattr(stream, "write", lambda data: len(data) - 1)
    with pytest.raises(OSError, match="complete QPY payload"):
        qpy.dump([], stream)


@requires_translation
def test_failed_export_preserves_destination_and_source(tmp_path: Path) -> None:
    """An invalid QPY version must not truncate an existing destination or consume IR."""
    program = QCProgram.from_qiskit(QuantumCircuit(1))
    original_ir = program.ir
    destination = tmp_path / "keep.qpy"
    destination.write_bytes(b"keep me")
    with pytest.raises(ValueError, match="version"):
        qpy.dump(program, destination, version=qiskit_qpy.QPY_VERSION + 1)
    assert destination.read_bytes() == b"keep me"
    assert program.ir == original_ir


@requires_translation
def test_failed_translation_preserves_destination_and_sources() -> None:
    """Failure in a later program leaves earlier programs valid and writes nothing."""
    valid = QCProgram.from_qiskit(QuantumCircuit(1))
    invalid = valid.copy()
    invalid.to_qco()
    stream = BytesIO(b"keep me")
    with pytest.raises(RuntimeError, match=r"[Ii]nvalid|consumed"):
        qpy.dump([valid, invalid], stream)
    assert valid.is_valid
    assert stream.getvalue() == b"keep me"


def test_qpy_import_keeps_qiskit_optional() -> None:
    """Importing the helper module must not load Qiskit or native compiler bindings."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import mqt.core.qpy; "
                "assert 'qiskit' not in sys.modules; assert 'mqt.core.mlir' not in sys.modules; "
                "sys.modules['qiskit'] = None; "
                "from io import BytesIO; "
                "mqt.core.qpy.load(BytesIO())"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr
    assert "qiskit" in result.stderr
    assert "AssertionError" not in result.stderr
