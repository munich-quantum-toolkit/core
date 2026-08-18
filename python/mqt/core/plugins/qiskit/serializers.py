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
kind a format carries. A format for which
:func:`~mqt.core.qdmi.has_program_payload` is false carries no program at all
and cannot have a serializer.

MQT Core registers its own OpenQASM 2 and OpenQASM 3 serializers here. Every
other format belongs to the package that owns the device. Such a package
advertises its serializer through the ``mqt.core.qiskit.program_serializers``
entry point group. The entry point name is the
:class:`~mqt.core.qdmi.ProgramFormat` member name, and the value points to the
serializer:

```toml
[project.entry-points."mqt.core.qiskit.program_serializers"]
IQM_JSON = "iqm.qdmi.serializers:qiskit_to_iqm_json"
```

:func:`register_program_serializer` does the same at run time. A registration
takes precedence over an entry point for the same format.

A device usually accepts several formats. :data:`PROGRAM_FORMAT_PREFERENCE`
records which one to use, from most to least preferred, and
:func:`preferred_program_formats` applies that order to the formats a device
reports.
"""

from __future__ import annotations

import warnings
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Protocol

from ...qdmi import ProgramFormat, has_program_payload

if TYPE_CHECKING:
    from collections.abc import Iterable

    from qiskit.circuit import QuantumCircuit

    from .backend import QDMIBackend

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
ProgramSerializer = TextProgramSerializer | BinaryProgramSerializer

#: The program formats in the order the backend prefers them, most preferred
#: first. A device-native format comes first, because a package that registers a
#: serializer for its own device's format wants that format used. The
#: standardized formats follow in order of what a circuit may contain: the QIR
#: adaptive profile allows classical control, QPY carries a Qiskit circuit
#: without loss, and OpenQASM 3 expresses control flow, while the QIR base
#: profile forbids classical feedback and OpenQASM 2 has no control flow at all.
#: Encoding only breaks a tie within one profile, because it decides how the
#: program travels rather than what it may say. ``CALIBRATION`` and
#: ``BATCH_JOB`` are absent because a serialized circuit is not what they carry.
PROGRAM_FORMAT_PREFERENCE: tuple[ProgramFormat, ...] = (
    ProgramFormat.IQM_JSON,
    ProgramFormat.CUSTOM1,
    ProgramFormat.CUSTOM2,
    ProgramFormat.CUSTOM3,
    ProgramFormat.CUSTOM4,
    ProgramFormat.CUSTOM5,
    ProgramFormat.QIR_ADAPTIVE_MODULE,
    ProgramFormat.QIR_ADAPTIVE_STRING,
    ProgramFormat.QPY,
    ProgramFormat.QASM3,
    ProgramFormat.QIR_BASE_MODULE,
    ProgramFormat.QIR_BASE_STRING,
    ProgramFormat.QASM2,
)

_SERIALIZERS: dict[ProgramFormat, ProgramSerializer] = {}
_ENTRY_POINTS_LOADED = False


def register_program_serializer(fmt: ProgramFormat, serializer: ProgramSerializer, *, replace: bool = False) -> None:
    """Register a serializer for one program format.

    Args:
        fmt: The program format the serializer produces.
        serializer: The serializer to register. It must return :class:`str` for
            a text format and :class:`bytes` for a binary format.
        replace: Replace an existing serializer for the same format.

    Raises:
        ValueError: If the format carries no program payload, or if the format
            already has a serializer and ``replace`` is false.
    """
    if not has_program_payload(fmt):
        msg = f"{fmt.name} carries no program payload, so it cannot have a program serializer."
        raise ValueError(msg)
    # This function does not read the entry points. A registration must be able
    # to precede them, because that is what gives it precedence, and because
    # `backend.py` registers the OpenQASM formats while the adapter is still
    # importing.
    if not replace and fmt in _SERIALIZERS:
        msg = f"A program serializer for {fmt.name} is already registered. Pass replace=True to override it."
        raise ValueError(msg)
    _SERIALIZERS[fmt] = serializer


def unregister_program_serializer(fmt: ProgramFormat) -> None:
    """Remove the serializer for one program format.

    Args:
        fmt: The program format whose serializer to remove. A format without a
            serializer is ignored.
    """
    _load_entry_points()
    _SERIALIZERS.pop(fmt, None)


def program_serializer(fmt: ProgramFormat) -> ProgramSerializer | None:
    """Return the serializer for one program format.

    Args:
        fmt: The program format to look up.

    Returns:
        The registered serializer, or ``None`` if no package provides one.
    """
    _load_entry_points()
    return _SERIALIZERS.get(fmt)


def preferred_program_formats(formats: Iterable[ProgramFormat]) -> list[ProgramFormat]:
    """Order the program formats a device reports by :data:`PROGRAM_FORMAT_PREFERENCE`.

    Args:
        formats: The program formats the device accepts.

    Returns:
        Those of the given formats that carry a program payload, most preferred
        first. A format that :data:`PROGRAM_FORMAT_PREFERENCE` does not name
        comes after every format it does name, in the order it was given.
    """
    ranks = {fmt: rank for rank, fmt in enumerate(PROGRAM_FORMAT_PREFERENCE)}
    unranked = len(PROGRAM_FORMAT_PREFERENCE)
    candidates = [fmt for fmt in formats if has_program_payload(fmt)]
    return sorted(candidates, key=lambda fmt: ranks.get(fmt, unranked))


def _load_entry_points() -> None:
    """Load the serializers advertised through :data:`ENTRY_POINT_GROUP` once.

    An entry point that names an unknown program format, names a format without
    a program payload, or fails to load produces a warning and is skipped, so
    one broken package cannot make every other serializer unreachable.
    """
    global _ENTRY_POINTS_LOADED  # ruff:ignore[global-statement] Guards a one-time import side effect
    if _ENTRY_POINTS_LOADED:
        return
    # The flag guards the loop below, which imports serializer modules that may
    # call back into this module.
    _ENTRY_POINTS_LOADED = True

    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        try:
            fmt = ProgramFormat[entry_point.name]
        except KeyError:
            warnings.warn(
                f"Entry point '{entry_point.name}' in group '{ENTRY_POINT_GROUP}' does not name a program format "
                f"and will be skipped.",
                UserWarning,
                stacklevel=2,
            )
            continue

        if not has_program_payload(fmt):
            warnings.warn(
                f"Entry point '{entry_point.name}' in group '{ENTRY_POINT_GROUP}' names a program format that "
                f"carries no program payload and will be skipped.",
                UserWarning,
                stacklevel=2,
            )
            continue

        try:
            serializer = entry_point.load()
        except Exception as exc:  # ruff:ignore[blind-except] One bad package must not break the others
            warnings.warn(
                f"Failed to load the program serializer for {fmt.name} from '{entry_point.value}': {exc}",
                UserWarning,
                stacklevel=2,
            )
            continue

        # An explicit registration for this format takes precedence.
        _SERIALIZERS.setdefault(fmt, serializer)
