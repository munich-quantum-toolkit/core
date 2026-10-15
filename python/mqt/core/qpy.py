# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Read and write QPY files through the optional Qiskit compiler bridge.

Only the circuit features supported by that bridge are preserved. This is not
a lossless archive of arbitrary Qiskit objects or metadata.
"""

from __future__ import annotations

__lazy_modules__ = {"io", "pathlib", "struct"}

import struct
from contextlib import contextmanager, nullcontext
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from typing import BinaryIO

    from mqt.core.mlir import CompilerTarget, QCOProgram, QCProgram

__all__ = ["dump", "load"]


@contextmanager
def _qpy_native_panics() -> Generator[None, None, None]:
    """Make Qiskit's native panics catchable without swallowing unrelated failures.

    Raises:
        ValueError: If Qiskit raises a PyO3 panic exception.
    """
    try:
        yield
    except BaseException as error:
        # PyO3 has no importable panic class; use it directly if Qiskit exposes one.
        if type(error).__module__ != "pyo3_runtime" or type(error).__name__ != "PanicException":
            raise
        msg = f"Invalid or unsupported QPY input: {error}"
        raise ValueError(msg) from error


def load(file: str | PathLike[str] | BinaryIO) -> list[QCProgram]:
    """Load every circuit in a QPY file as a QC program, in file order.

    Requires Qiskit in the compiler bridge's supported version range. Qiskit
    validates the QPY format and version; Core validates each circuit. If any
    circuit is unsupported, the call raises rather than returning a partial list.
    Native Qiskit panics become ValueError with the original exception as cause.
    Other dependency, I/O, and circuit translation errors propagate to the caller.

    Args:
        file: Path or open binary stream. Caller-owned streams remain open and
            are read from their current position.

    Returns:
        One program per circuit, including an empty list for an empty archive.

    Raises:
        ImportError: If Qiskit or a required QPY decoder dependency is missing.
        ValueError: If the QPY data is malformed, its version is unsupported,
            or Qiskit's native code panics during decoding or conversion.
    """
    from qiskit import qpy  # ruff: ignore[import-outside-top-level] Optional dependency.
    from qiskit.exceptions import QiskitError  # ruff: ignore[import-outside-top-level] Optional dependency.

    from mqt.core.mlir import QCProgram  # ruff: ignore[import-outside-top-level] Load native bindings only on use.

    with (
        Path(file).open("rb") if isinstance(file, (str, PathLike)) else nullcontext(file) as stream,
        BytesIO(stream.read()) as buffer,
        _qpy_native_panics(),
    ):
        try:
            circuits = qpy.load(buffer)
        except ImportError:
            raise
        except (QiskitError, struct.error, EOFError, ValueError, TypeError, LookupError, AttributeError) as error:
            msg = f"Invalid or unsupported QPY input: {error}"
            raise ValueError(msg) from error
    with _qpy_native_panics():
        return [QCProgram.from_qiskit(circuit) for circuit in circuits]


def dump(
    programs: QCProgram | QCOProgram | Iterable[QCProgram | QCOProgram],
    file: str | PathLike[str] | BinaryIO,
    *,
    version: int | None = None,
    target: CompilerTarget | None = None,
) -> None:
    """Write QC or QCO programs to QPY without consuming the source programs.

    All circuits are translated and serialized in memory before opening the
    destination. Translation or serialization failures leave it unchanged;
    an I/O failure during writing may leave a partial file.
    Dependency, translation, QPY version, and I/O errors propagate to the caller.

    Args:
        programs: One program or an iterable of programs in output order.
        file: Path or open binary stream. Paths are overwritten on success.
            Caller-owned streams remain open and are written at their current
            position, without truncating existing trailing bytes.
        version: QPY format version, or None for Qiskit's default. Qiskit
            validates the requested version and its supported circuit features.
        target: Optional target forwarded to ``to_qiskit`` for site mapping and
            gate names. This does not compile or route the programs.

    Raises:
        TypeError: If an input is not a QC or QCO program.
        OSError: If the output stream does not accept the complete payload.
    """
    from qiskit import qpy  # ruff: ignore[import-outside-top-level] Optional dependency.

    from mqt.core.mlir import QCOProgram, QCProgram  # ruff: ignore[import-outside-top-level] Load native bindings only on use.

    items = [programs] if isinstance(programs, (QCProgram, QCOProgram)) else list(programs)
    if any(not isinstance(program, (QCProgram, QCOProgram)) for program in items):
        msg = "QPY export requires QCProgram or QCOProgram objects."
        raise TypeError(msg)
    circuits = [program.to_qiskit(target=target, copy=True) for program in items]
    with BytesIO() as buffer:
        qpy.dump(circuits, buffer, version=qpy.QPY_VERSION if version is None else version)
        payload = buffer.getvalue()
        with Path(file).open("wb") if isinstance(file, (str, PathLike)) else nullcontext(file) as stream:
            if stream.write(payload) != len(payload):
                msg = "Could not write the complete QPY payload."
                raise OSError(msg)
