# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QDMI Qiskit optional integration.

Exposes lightweight dependency guards and (in later phases) backend related
APIs for interacting with QDMI devices through Qiskit's BackendV2 interface.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Final

# Conditional backend import (optional dependency)
try:
    from .backend import QiskitBackend

    _BACKEND_AVAILABLE = True
except ImportError:
    _BACKEND_AVAILABLE = False

from .exceptions import (
    CapabilityMismatchError,
    QDMIQiskitError,
    TranslationError,
    UnsupportedOperationError,
)
from .translator import (
    InstructionContext,
    build_program_ir,
    clear_operation_translators,
    get_operation_translator,
    list_operation_translators,
    register_operation_translator,
    unregister_operation_translator,
)
from .types import IRValidationError, ProgramInstruction, ProgramIR

if TYPE_CHECKING:  # pragma: no cover
    from types import ModuleType

__all__ = [
    "IRValidationError",
    "InstructionContext",
    "ProgramIR",
    "ProgramInstruction",
    "QiskitNotAvailableError",
    "build_program_ir",
    "clear_operation_translators",
    "get_operation_translator",
    "is_available",
    "list_operation_translators",
    "register_operation_translator",
    "require_qiskit",
    "unregister_operation_translator",
]
if _BACKEND_AVAILABLE:
    __all__ += [
        "CapabilityMismatchError",
        "QDMIQiskitError",
        "QiskitBackend",
        "TranslationError",
        "UnsupportedOperationError",
    ]


class QiskitNotAvailableError(ImportError):
    """Raised when Qiskit functionality is accessed without the optional dependency."""


_QISKIT_MODULE_NAME: Final = "qiskit"


def is_available() -> bool:
    """Return True if the optional Qiskit dependency is installed."""
    try:  # pragma: no cover - trivial success path
        import_module(_QISKIT_MODULE_NAME)
    except ImportError:
        return False
    else:
        return True


def require_qiskit() -> ModuleType:
    """Ensure Qiskit is available and return the imported top-level module.

    Returns:
        The imported ``qiskit`` module.

    Raises:
        QiskitNotAvailableError: If Qiskit is not installed.
    """
    try:
        return import_module(_QISKIT_MODULE_NAME)
    except ImportError as err:
        msg = "Qiskit is not installed. Install with 'pip install mqt-core[qiskit]'"
        raise QiskitNotAvailableError(msg) from err


def __getattr__(name: str) -> object:  # pragma: no cover - dynamic fallback
    """Dynamic attribute guard for optional backend symbol.

    Raises:
        QiskitNotAvailableError: If backend requested when qiskit missing.
        AttributeError: For any other unknown attribute.
    """
    if name == "QiskitBackend" and not _BACKEND_AVAILABLE:
        msg = "'QiskitBackend' requires the optional qiskit dependency. Install with 'pip install mqt-core[qiskit]'"
        raise QiskitNotAvailableError(msg)
    raise AttributeError(name)
