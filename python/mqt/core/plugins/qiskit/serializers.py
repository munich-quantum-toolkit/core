# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Registry of program serializers for QDMI program formats.

A QDMI device accepts a program in one or more *program formats*, listed by
:class:`~mqt.core.qdmi.ProgramFormat`. A *program serializer* turns one Qiskit
:class:`~qiskit.circuit.QuantumCircuit` into one program in one such format.

A format fixes the kind of payload it carries, so there are two signatures. A
text format takes a :class:`TextProgramSerializer`, which returns :class:`str`.
A binary format takes a :class:`BinaryProgramSerializer`, which returns
:class:`bytes`. :func:`~mqt.core.qdmi.is_binary_program_format` states which
kind a format carries.

MQT Core provides OpenQASM 2/3, QPY, IQM JSON, and Base/Adaptive QIR serializers.
Device packages can advertise their serializers through the ``mqt.core.qiskit.program_serializers``
entry point group. The entry point name is the
:class:`~mqt.core.qdmi.ProgramFormat` member name, and the value points to the
serializer:

```toml
[project.entry-points."mqt.core.qiskit.program_serializers"]
IQM_JSON = "iqm.qdmi.serializers:qiskit_to_iqm_json"
```

:func:`register_program_serializer` does the same at run time. A registration
takes precedence over an entry point for the same format. All built-in serializers are fallbacks;
registrations and entry points take precedence over them.

A device usually accepts several formats. :data:`PROGRAM_FORMAT_PREFERENCE`
records which one to use, from most to least preferred, and
:func:`preferred_program_formats` applies that order to the formats a device
reports.
"""

from __future__ import annotations

__lazy_modules__ = {"importlib.metadata", "io", "mqt.core.mlir", "warnings"}

import warnings
from collections.abc import Callable
from functools import partial
from importlib.metadata import entry_points
from io import BytesIO
from typing import TYPE_CHECKING, Protocol

from mqt.core.mlir import CompilerTarget, QCProgram, QIRProfile
from mqt.core.qdmi import ProgramFormat

if TYPE_CHECKING:
    from collections.abc import Iterable
    from importlib.metadata import EntryPoint

    from qiskit.circuit import QuantumCircuit

    from mqt.core.plugins.qiskit.backend import QDMIBackend

__all__ = [
    "ENTRY_POINT_GROUP",
    "PROGRAM_FORMAT_PREFERENCE",
    "BinaryProgramSerializer",
    "ProgramSerializer",
    "TextProgramSerializer",
    "preferred_program_formats",
    "program_serializer",
    "register_program_serializer",
    "unregister_program_serializer",
]


def __dir__() -> list[str]:
    return __all__


#: Entry point group through which a package advertises its program serializers.
ENTRY_POINT_GROUP = "mqt.core.qiskit.program_serializers"


class TextProgramSerializer(Protocol):
    """Serializes a circuit into a program format whose payload is text."""

    def __call__(self, circuit: QuantumCircuit, backend: QDMIBackend, /) -> str:
        """Serialize a circuit into a program string.

        Args:
            circuit: The circuit to serialize. It has no unbound parameters.
            backend: The backend that runs the circuit. Its ``device`` property
                provides the site names and metadata a format may need, and its
                ``target`` property provides the supported operations.

        Returns:
            The program in the serializer's format.

        Raises:
            UnsupportedOperationError: If the circuit contains an operation the
                format cannot express.
            TranslationError: If serialization fails for another reason.
        """
        ...


class BinaryProgramSerializer(Protocol):
    """Serializes a circuit into a program format whose payload is binary."""

    def __call__(self, circuit: QuantumCircuit, backend: QDMIBackend, /) -> bytes:
        """Serialize a circuit into program bytes.

        Args:
            circuit: The circuit to serialize. It has no unbound parameters.
            backend: The backend that runs the circuit. Its ``device`` property
                provides the site names and metadata a format may need, and its
                ``target`` property provides the supported operations.

        Returns:
            The program in the serializer's format.

        Raises:
            UnsupportedOperationError: If the circuit contains an operation the
                format cannot express.
            TranslationError: If serialization fails for another reason.
        """
        ...


#: A serializer for one program format, text or binary.
ProgramSerializer = Callable[["QuantumCircuit", "QDMIBackend"], str | bytes]

#: Preferred formats, in descending order: device-native formats first, then
#: QPY and OpenQASM 3 preserve device gate names. Adaptive QIR precedes
#: Base QIR; binary encoding wins ties within each profile.
PROGRAM_FORMAT_PREFERENCE: tuple[ProgramFormat, ...] = (
    ProgramFormat.IQM_JSON,
    ProgramFormat.CUSTOM1,
    ProgramFormat.CUSTOM2,
    ProgramFormat.CUSTOM3,
    ProgramFormat.CUSTOM4,
    ProgramFormat.CUSTOM5,
    ProgramFormat.QPY,
    ProgramFormat.QASM3,
    ProgramFormat.QIR_ADAPTIVE_MODULE,
    ProgramFormat.QIR_ADAPTIVE_STRING,
    ProgramFormat.QIR_BASE_MODULE,
    ProgramFormat.QIR_BASE_STRING,
    ProgramFormat.QASM2,
)


def _serialize_to_qasm3(circuit: QuantumCircuit, backend: QDMIBackend) -> str:
    """Export validated native operations and device sites through MQT Core.

    Returns:
        OpenQASM 3 with explicit classical outputs.
    """
    backend.validate_circuit(circuit, native=True)
    return QCProgram.from_qiskit(circuit, native=True).to_openqasm3().source


def _serialize_to_qasm2(circuit: QuantumCircuit, backend: QDMIBackend) -> str:
    """Serialize a circuit into an OpenQASM 2 program.

    Args:
        circuit: The circuit to serialize.
        backend: The backend whose native placements constrain the circuit.

    Returns:
        The OpenQASM 2 program.
    """
    backend.validate_circuit(circuit, native=True)
    return QCProgram.from_qiskit(circuit, native=True).to_openqasm2().source


def _serialize_to_qpy(circuit: QuantumCircuit, backend: QDMIBackend) -> bytes:
    """Serialize a circuit after validating native placements.

    Returns:
        One circuit encoded as QPY bytes, using at most version 17.
    """
    from qiskit import qpy  # ruff: ignore[import-outside-top-level] - Load QPY only when this serializer is used.

    backend.validate_circuit(circuit, native=True)
    with BytesIO() as buffer:
        # QPY 17 enables native writing and remains readable by Qiskit 2.5.
        qpy.dump(circuit, buffer, version=17)
        return buffer.getvalue()


def _serialize_to_qir(
    circuit: QuantumCircuit, backend: QDMIBackend, *, adaptive: bool = False, binary: bool = False
) -> str | bytes:
    """Export device gate calls as QIR text or bitcode.

    Returns:
        The selected QIR encoding, with circuit indices as device qubits.
    """
    backend.validate_circuit(circuit, native=True)
    program = QCProgram.from_qiskit(circuit, native=True).to_qir(QIRProfile.ADAPTIVE if adaptive else QIRProfile.BASE)
    return program.to_bitcode() if binary else program.to_llvm_ir()


def _serialize_to_iqm_json(circuit: QuantumCircuit, backend: QDMIBackend) -> str:
    """Export PRX/CZ gates and terminal measurements using device site names.

    Returns:
        An IQM circuit object in JSON.
    """
    backend.validate_circuit(circuit, native=True)
    return QCProgram.from_qiskit(circuit, native=True).to_iqm_json(CompilerTarget.from_device(backend.device)).source


_FALLBACKS: dict[ProgramFormat, ProgramSerializer] = {
    ProgramFormat.QASM2: _serialize_to_qasm2,
    ProgramFormat.QASM3: _serialize_to_qasm3,
    ProgramFormat.QPY: _serialize_to_qpy,
    ProgramFormat.IQM_JSON: _serialize_to_iqm_json,
    ProgramFormat.QIR_BASE_STRING: _serialize_to_qir,
    ProgramFormat.QIR_BASE_MODULE: partial(_serialize_to_qir, binary=True),
    ProgramFormat.QIR_ADAPTIVE_STRING: partial(_serialize_to_qir, adaptive=True),
    ProgramFormat.QIR_ADAPTIVE_MODULE: partial(_serialize_to_qir, adaptive=True, binary=True),
}


class _ProgramSerializerRegistry:
    """The serializers for one process, and the entry points behind them.

    The registry reads :data:`ENTRY_POINT_GROUP` once, on the first lookup
    rather than at import, because loading an entry point imports the package
    that advertises it. The registry does not synchronize concurrent first use.
    """

    def __init__(self, discover: Callable[[], Iterable[EntryPoint]]) -> None:
        """Initialize the built-in serializers and defer entry-point discovery.

        Args:
            discover: Returns the entry points that advertise a serializer.
                Injected so a test can supply its own without touching the
                installed distributions.
        """
        self._discover = discover
        self._discovery_started = False
        self._serializers: dict[ProgramFormat, ProgramSerializer | None] = {}

    def register(self, fmt: ProgramFormat, serializer: ProgramSerializer, *, replace: bool = False) -> None:
        """Add a serializer for one program format.

        Registration skips entry-point discovery to allow calls during import.
        Explicit registrations take precedence over discovered serializers.

        Args:
            fmt: The program format the serializer produces.
            serializer: The serializer to add.
            replace: Replace an existing serializer for the same format.

        Raises:
            ValueError: If the format already has a serializer and ``replace`` is false.
        """
        if not replace and self._serializers.get(fmt) is not None:
            msg = f"A program serializer for {fmt.name} is already registered. Pass replace=True to override it."
            raise ValueError(msg)
        self._serializers[fmt] = serializer

    def unregister(self, fmt: ProgramFormat) -> None:
        """Remove the serializer for one program format.

        Args:
            fmt: The program format whose serializer to remove. A format without
                a serializer is ignored.
        """
        self._load_entry_points()
        self._serializers[fmt] = None  # Suppress built-in fallback after removal.

    def get(self, fmt: ProgramFormat) -> ProgramSerializer | None:
        """Return the serializer for one program format.

        Args:
            fmt: The program format to look up.

        Returns:
            The serializer, or ``None`` if no package provides one.
        """
        self._load_entry_points()
        if fmt in self._serializers:
            return self._serializers[fmt]
        return _FALLBACKS.get(fmt)

    def _load_entry_points(self) -> None:
        """Read the entry points once and publish what they name.

        Imported plugins may reenter the registry. During discovery, they see
        built-ins and explicit registrations; discovered serializers are published after
        all entry points have loaded.
        """
        if self._discovery_started:
            return

        self._discovery_started = True
        discovered: dict[ProgramFormat, ProgramSerializer] = {}
        try:
            for entry_point in self._discover():
                loaded = _ProgramSerializerRegistry._load_entry_point(entry_point)
                if loaded is not None:
                    fmt, serializer = loaded
                    discovered.setdefault(fmt, serializer)
        except BaseException:
            # Allow a later lookup to retry interrupted discovery.
            self._discovery_started = False
            raise

        # A registration made before or during discovery keeps precedence.
        for fmt, serializer in discovered.items():
            self._serializers.setdefault(fmt, serializer)

    @staticmethod
    def _load_entry_point(entry_point: EntryPoint) -> tuple[ProgramFormat, ProgramSerializer] | None:
        """Resolve one entry point into a format and its serializer.

        An entry point that names an unknown program format or fails to load
        produces a warning and is skipped, so one broken package
        cannot make every other serializer unreachable.

        Args:
            entry_point: The entry point to resolve.

        Returns:
            The format and its serializer, or ``None`` if the entry point is
            unusable.
        """
        try:
            fmt = ProgramFormat[entry_point.name]
        except KeyError:
            warnings.warn(
                f"Entry point '{entry_point.name}' in group '{ENTRY_POINT_GROUP}' does not name a program format "
                f"and will be skipped.",
                UserWarning,
                stacklevel=2,
            )
            return None

        try:
            serializer = entry_point.load()
        except Exception as exc:  # ruff:ignore[blind-except] One bad package must not break the others
            warnings.warn(
                f"Failed to load the program serializer for {fmt.name} from '{entry_point.value}': {exc}",
                UserWarning,
                stacklevel=2,
            )
            return None

        return fmt, serializer


_REGISTRY = _ProgramSerializerRegistry(lambda: entry_points(group=ENTRY_POINT_GROUP))


def register_program_serializer(fmt: ProgramFormat, serializer: ProgramSerializer, *, replace: bool = False) -> None:
    """Register a serializer for one program format.

    Args:
        fmt: The program format the serializer produces.
        serializer: The serializer to register. It must return :class:`str` for
            a text format and :class:`bytes` for a binary format.
        replace: Replace an existing serializer for the same format.

    Raises:
        ValueError: If the format already has a serializer and ``replace`` is false.
    """  # ruff:ignore[docstring-extraneous-exception] The registry raises ValueError.
    _REGISTRY.register(fmt, serializer, replace=replace)


def unregister_program_serializer(fmt: ProgramFormat) -> None:
    """Remove the serializer for one program format.

    Args:
        fmt: The program format whose serializer to remove. A format without a
            serializer is ignored.
    """
    _REGISTRY.unregister(fmt)


def program_serializer(fmt: ProgramFormat) -> ProgramSerializer | None:
    """Return the serializer for one program format.

    Args:
        fmt: The program format to look up.

    Returns:
        The registered serializer, or ``None`` if no package provides one.
    """
    return _REGISTRY.get(fmt)


def preferred_program_formats(formats: Iterable[ProgramFormat]) -> list[ProgramFormat]:
    """Order the program formats a device reports by :data:`PROGRAM_FORMAT_PREFERENCE`.

    Args:
        formats: The program formats the device accepts.

    Returns:
        The given formats, most preferred first. A format that :data:`PROGRAM_FORMAT_PREFERENCE` does
        not name comes after every format it does name, in the order it was
        given.
    """
    ranks = {fmt: rank for rank, fmt in enumerate(PROGRAM_FORMAT_PREFERENCE)}
    unranked = len(PROGRAM_FORMAT_PREFERENCE)
    return sorted(formats, key=lambda fmt: ranks.get(fmt, unranked))
