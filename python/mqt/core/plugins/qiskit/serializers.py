# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Registry of Qiskit serializers for exact QDMI payload descriptors.

An entry point in ``mqt.core.qiskit.program_serializers`` must export a
``(PayloadDescriptor, serializer)`` tuple. A direct registration takes
precedence over a discovered entry point for the same descriptor.
"""

from __future__ import annotations

import warnings
from enum import Enum, auto
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Protocol

from ...qdmi import PayloadDescriptor

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from importlib.metadata import EntryPoint

    from qiskit.circuit import QuantumCircuit

    from .backend import QDMIBackend

__all__ = [
    "ENTRY_POINT_GROUP",
    "BinaryProgramSerializer",
    "ProgramSerializer",
    "TextProgramSerializer",
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


def _format_name(fmt: PayloadDescriptor) -> str:
    """Return a concise exact payload name for diagnostics."""
    profile = f"/{fmt.profile}" if fmt.profile else ""
    return f"{fmt.format_id}/{'.'.join(map(str, fmt.version))}{profile}/{fmt.encoding.name.lower()}"


class _LoadState(Enum):
    """How far the registry has got with reading the entry points."""

    NOT_STARTED = auto()
    LOADING = auto()
    LOADED = auto()


class _ProgramSerializerRegistry:
    """The serializers for one process, and the entry points behind them.

    The registry reads :data:`ENTRY_POINT_GROUP` once, on the first lookup
    rather than at import, because loading an entry point imports the package
    that advertises it. The registry does not synchronize concurrent first use.
    """

    def __init__(self, discover: Callable[[], Iterable[EntryPoint]]) -> None:
        """Initialize an empty registry.

        Args:
            discover: Returns the entry points that advertise a serializer.
                Injected so a test can supply its own without touching the
                installed distributions.
        """
        self._discover = discover
        self._serializers: dict[PayloadDescriptor, ProgramSerializer] = {}
        self._load_state = _LoadState.NOT_STARTED

    def register(self, fmt: PayloadDescriptor, serializer: ProgramSerializer, *, replace: bool = False) -> None:
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
        if not replace and fmt in self._serializers:
            msg = (
                f"A program serializer for {_format_name(fmt)} is already registered. Pass replace=True to override it."
            )
            raise ValueError(msg)
        self._serializers[fmt] = serializer

    def unregister(self, fmt: PayloadDescriptor) -> None:
        """Remove the serializer for one program format.

        Args:
            fmt: The program format whose serializer to remove. A format without
                a serializer is ignored.
        """
        self._load_entry_points()
        self._serializers.pop(fmt, None)

    def get(self, fmt: PayloadDescriptor) -> ProgramSerializer | None:
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
        discovered: dict[PayloadDescriptor, ProgramSerializer] = {}
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
    def _load_entry_point(entry_point: EntryPoint) -> tuple[PayloadDescriptor, ProgramSerializer] | None:
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
            loaded = entry_point.load()
        except Exception as exc:  # ruff:ignore[blind-except] One bad package must not break the others
            warnings.warn(
                f"Failed to load program serializer entry point '{entry_point.value}': {exc}",
                UserWarning,
                stacklevel=2,
            )
            return None
        if not isinstance(loaded, tuple) or len(loaded) != 2 or not isinstance(loaded[0], PayloadDescriptor):
            warnings.warn(
                f"Entry point '{entry_point.name}' must export (PayloadDescriptor, serializer) and will be skipped.",
                UserWarning,
                stacklevel=2,
            )
            return None
        return loaded


_REGISTRY = _ProgramSerializerRegistry(lambda: entry_points(group=ENTRY_POINT_GROUP))


def register_program_serializer(
    fmt: PayloadDescriptor, serializer: ProgramSerializer, *, replace: bool = False
) -> None:
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


def unregister_program_serializer(fmt: PayloadDescriptor) -> None:
    """Remove the serializer for one program format.

    Args:
        fmt: The program format whose serializer to remove. A format without a
            serializer is ignored.
    """
    _REGISTRY.unregister(fmt)


def program_serializer(fmt: PayloadDescriptor) -> ProgramSerializer | None:
    """Return the serializer for one program format.

    Args:
        fmt: The program format to look up.

    Returns:
        The registered serializer, or ``None`` if no package provides one.
    """
    return _REGISTRY.get(fmt)
