# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Registration of program codecs for device-specific program formats.

A program codec converts a Qiskit :class:`~qiskit.circuit.QuantumCircuit` into
one program format that a QDMI device accepts. MQT Core implements the codecs
for OpenQASM 2 and OpenQASM 3 directly in
:class:`~mqt.core.plugins.qiskit.backend.QDMIBackend`. Every other format
belongs to the package that owns the device, which registers a codec here.

A package registers a codec through the ``mqt.core.qiskit.program_codecs``
entry point group. The entry point name is the
:class:`~mqt.core.qdmi.ProgramFormat` member name, and the value points to the
codec:

```toml
[project.entry-points."mqt.core.qiskit.program_codecs"]
IQM_JSON = "iqm.qdmi.converters:qiskit_to_iqm_json"
```

:func:`register_program_codec` does the same at run time. A registered codec
takes precedence over the built-in OpenQASM codecs.
"""

from __future__ import annotations

import warnings
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Protocol

from ...qdmi import ProgramFormat

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

    from ...qdmi import Device as QDMIDevice

__all__ = [
    "ENTRY_POINT_GROUP",
    "ProgramCodec",
    "program_codec",
    "register_program_codec",
    "unregister_program_codec",
]


def __dir__() -> list[str]:
    return __all__


#: Entry point group through which a package advertises its program codecs.
ENTRY_POINT_GROUP = "mqt.core.qiskit.program_codecs"


class ProgramCodec(Protocol):
    """Converts a circuit into one device-specific program format."""

    def __call__(self, circuit: QuantumCircuit, device: QDMIDevice, /) -> str:
        """Convert a circuit into a program string.

        Args:
            circuit: The circuit to convert. It has no unbound parameters.
            device: The device the program runs on. It provides the site names
                and metadata the format needs.

        Returns:
            The program in the codec's format.

        Raises:
            UnsupportedOperationError: If the circuit contains an operation the
                format cannot express.
            TranslationError: If the conversion fails for another reason.
        """
        ...


_CODECS: dict[ProgramFormat, ProgramCodec] = {}
_ENTRY_POINTS_LOADED = False


def register_program_codec(fmt: ProgramFormat, codec: ProgramCodec, *, replace: bool = False) -> None:
    """Register a codec for one program format.

    Args:
        fmt: The program format the codec produces.
        codec: The codec to register.
        replace: Replace an existing codec for the same format.

    Raises:
        ValueError: If the format already has a codec and ``replace`` is false.
    """
    _load_entry_points()
    if not replace and fmt in _CODECS:
        msg = f"A program codec for {fmt.name} is already registered. Pass replace=True to override it."
        raise ValueError(msg)
    _CODECS[fmt] = codec


def unregister_program_codec(fmt: ProgramFormat) -> None:
    """Remove the codec for one program format.

    Args:
        fmt: The program format whose codec to remove. A format without a codec
            is ignored.
    """
    _load_entry_points()
    _CODECS.pop(fmt, None)


def program_codec(fmt: ProgramFormat) -> ProgramCodec | None:
    """Return the codec for one program format.

    Args:
        fmt: The program format to look up.

    Returns:
        The registered codec, or ``None`` if no package provides one.
    """
    _load_entry_points()
    return _CODECS.get(fmt)


def _load_entry_points() -> None:
    """Load the codecs advertised through :data:`ENTRY_POINT_GROUP` once.

    An entry point that names an unknown program format or that fails to load
    produces a warning and is skipped, so one broken package cannot make every
    other codec unreachable.
    """
    global _ENTRY_POINTS_LOADED  # ruff:ignore[global-statement] Guards a one-time import side effect
    if _ENTRY_POINTS_LOADED:
        return
    # The flag guards the loop below, which imports codec modules that may call
    # back into this module.
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

        try:
            codec = entry_point.load()
        except Exception as exc:  # ruff:ignore[blind-except] One bad package must not break the others
            warnings.warn(
                f"Failed to load the program codec for {fmt.name} from '{entry_point.value}': {exc}",
                UserWarning,
                stacklevel=2,
            )
            continue

        # An explicit registration for this format takes precedence.
        _CODECS.setdefault(fmt, codec)
