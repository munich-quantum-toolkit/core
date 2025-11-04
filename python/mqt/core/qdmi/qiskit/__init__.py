# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QDMI Qiskit optional integration."""

from __future__ import annotations

from mqt.core._compat.optional import OptionalDependencyTester  # noqa: PLC2701

# Conditional backend import (optional dependency)
try:
    from .backend import QiskitBackend
    from .provider import QDMIProvider

    _BACKEND_AVAILABLE = True
except ImportError:
    _BACKEND_AVAILABLE = False

from .exceptions import (
    CircuitValidationError,
    JobSubmissionError,
    QDMIQiskitError,
    TranslationError,
    UnsupportedFormatError,
    UnsupportedOperationError,
)

__all__ = [
    "HAS_QISKIT",
    "CircuitValidationError",
    "JobSubmissionError",
    "QDMIQiskitError",
    "QiskitNotAvailableError",
    "TranslationError",
    "UnsupportedFormatError",
    "UnsupportedOperationError",
]
if _BACKEND_AVAILABLE:
    __all__ += [
        "QDMIProvider",
        "QiskitBackend",
    ]


class QiskitNotAvailableError(ImportError):
    """Raised when Qiskit functionality is accessed without the optional dependency."""


# Optional dependency tester for Qiskit
HAS_QISKIT = OptionalDependencyTester(
    "qiskit",
    install_msg="Install with 'pip install mqt-core[qiskit]'",
)


def __getattr__(name: str) -> object:  # pragma: no cover - dynamic fallback
    """Dynamic attribute guard for optional backend symbols.

    Raises:
        QiskitNotAvailableError: If backend requested when qiskit missing.
        AttributeError: For any other unknown attribute.
    """
    if name in {"QiskitBackend", "QDMIProvider"} and not _BACKEND_AVAILABLE:
        msg = f"'{name}' requires the optional qiskit dependency. Install with 'pip install mqt-core[qiskit]'"
        raise QiskitNotAvailableError(msg)
    raise AttributeError(name)
