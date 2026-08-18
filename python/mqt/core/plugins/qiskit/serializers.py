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
kind a format carries. Two formats take no serializer at all, because a
serialized circuit is not what they carry; see :data:`NON_CIRCUIT_FORMATS`.

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
from enum import Enum, auto
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Protocol

from ...qdmi import ProgramFormat

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from importlib.metadata import EntryPoint

    from qiskit.circuit import QuantumCircuit

    from .backend import QDMIBackend

__all__ = [
    "ENTRY_POINT_GROUP",
    "NON_CIRCUIT_FORMATS",
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

#: The program formats that no program serializer can produce. A serializer
#: turns one Qiskit circuit into one program, and neither of these carries such
#: a program: ``CALIBRATION`` asks the device to run a calibration routine, and
#: ``BATCH_JOB`` carries a list of already-created jobs. This states what a
#: circuit can be serialized into, which is a question about this adapter
#: rather than about what
#: :meth:`~mqt.core.qdmi.Device.submit_job` accepts.
NON_CIRCUIT_FORMATS: frozenset[ProgramFormat] = frozenset({
    ProgramFormat.CALIBRATION,
    ProgramFormat.BATCH_JOB,
})

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


class _LoadState(Enum):
    """How far the registry has got with reading the entry points."""

    NOT_STARTED = auto()
    LOADING = auto()
    LOADED = auto()


class _ProgramSerializerRegistry:
    """The serializers for one process, and the entry points behind them.

    The registry reads :data:`ENTRY_POINT_GROUP` once, on the first lookup
    rather than at import, because loading an entry point imports the package
    that advertises it. Discovery is not thread-safe: two threads that reach a
    cold registry together both read the entry points. Import-time discovery in
    several threads is already fraught, so the registry does not lock.
    """

    def __init__(self, discover: Callable[[], Iterable[EntryPoint]]) -> None:
        """Initialize an empty registry.

        Args:
            discover: Returns the entry points that advertise a serializer.
                Injected so a test can supply its own without touching the
                installed distributions.
        """
        self._discover = discover
        self._serializers: dict[ProgramFormat, ProgramSerializer] = {}
        self._load_state = _LoadState.NOT_STARTED

    def register(self, fmt: ProgramFormat, serializer: ProgramSerializer, *, replace: bool = False) -> None:
        """Add a serializer for one program format.

        Registering does not read the entry points. A registration must be able
        to precede them, because that is what gives it precedence, and because
        ``backend.py`` registers the OpenQASM formats while the adapter is still
        importing.

        Args:
            fmt: The program format the serializer produces.
            serializer: The serializer to add.
            replace: Replace an existing serializer for the same format.

        Raises:
            ValueError: If the format does not carry a serialized circuit, or if
                the format already has a serializer and ``replace`` is false.
        """
        if fmt in NON_CIRCUIT_FORMATS:
            msg = f"{fmt.name} does not carry a serialized circuit, so it cannot have a program serializer."
            raise ValueError(msg)
        if not replace and fmt in self._serializers:
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
        self._serializers.pop(fmt, None)

    def get(self, fmt: ProgramFormat) -> ProgramSerializer | None:
        """Return the serializer for one program format.

        Args:
            fmt: The program format to look up.

        Returns:
            The serializer, or ``None`` if no package provides one.
        """
        self._load_entry_points()
        return self._serializers.get(fmt)

    def _load_entry_points(self) -> None:
        """Read the entry points once and publish what they name.

        Loading an entry point imports a third-party module, which may call back
        into this registry. Such a call sees the ``LOADING`` state and returns
        without starting a second pass, so it observes the registrations made so
        far and none of the discovery in flight. Publishing the discovered
        serializers in one step at the end keeps that observation the same
        whatever order the entry points arrive in.
        """
        if self._load_state is not _LoadState.NOT_STARTED:
            return

        self._load_state = _LoadState.LOADING
        discovered: dict[ProgramFormat, ProgramSerializer] = {}
        try:
            for entry_point in self._discover():
                loaded = _ProgramSerializerRegistry._load_entry_point(entry_point)
                if loaded is not None:
                    fmt, serializer = loaded
                    discovered.setdefault(fmt, serializer)
        except BaseException:
            # Discovery did not finish, so leave the registry cold. A later
            # lookup tries again rather than reporting an empty result forever.
            self._load_state = _LoadState.NOT_STARTED
            raise

        # A registration made before or during discovery keeps precedence.
        for fmt, serializer in discovered.items():
            self._serializers.setdefault(fmt, serializer)
        self._load_state = _LoadState.LOADED

    @staticmethod
    def _load_entry_point(entry_point: EntryPoint) -> tuple[ProgramFormat, ProgramSerializer] | None:
        """Resolve one entry point into a format and its serializer.

        An entry point that names an unknown program format, names a format in
        :data:`NON_CIRCUIT_FORMATS`, or fails to load produces a warning and is
        skipped, so one broken package cannot make every other serializer
        unreachable.

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

        if fmt in NON_CIRCUIT_FORMATS:
            warnings.warn(
                f"Entry point '{entry_point.name}' in group '{ENTRY_POINT_GROUP}' names a program format that "
                f"does not carry a serialized circuit and will be skipped.",
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
        ValueError: If the format does not carry a serialized circuit, or if the
            format already has a serializer and ``replace`` is false. Raised by
            the registry this function delegates to.
    """  # ruff:ignore[docstring-extraneous-exception] The delegate raises it, and a caller must know
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
        Those of the given formats that can carry a serialized circuit, most
        preferred first. A format that :data:`PROGRAM_FORMAT_PREFERENCE` does
        not name comes after every format it does name, in the order it was
        given.
    """
    ranks = {fmt: rank for rank, fmt in enumerate(PROGRAM_FORMAT_PREFERENCE)}
    unranked = len(PROGRAM_FORMAT_PREFERENCE)
    candidates = [fmt for fmt in formats if fmt not in NON_CIRCUIT_FORMATS]
    return sorted(candidates, key=lambda fmt: ranks.get(fmt, unranked))
