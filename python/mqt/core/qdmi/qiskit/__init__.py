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

if TYPE_CHECKING:  # pragma: no cover
    from types import ModuleType

__all__ = ["QiskitNotAvailableError", "is_available", "require_qiskit"]


class QiskitNotAvailableError(ImportError):
    """Raised when Qiskit functionality is accessed without the optional dependency."""


_QISKIT_MODULE_NAME: Final = "qiskit"


def is_available() -> bool:
    """Return True if the optional Qiskit dependency is installed."""
    try:  # pragma: no cover - trivial success path
        import_module(_QISKIT_MODULE_NAME)
    except Exception:  # noqa: BLE001
        return False
    else:  # TRY300 explicit else
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
    except Exception as exc:
        msg = "Qiskit is not installed. Install with 'pip install mqt-core[qiskit]'"
        raise QiskitNotAvailableError(msg) from exc


def __getattr__(name: str) -> object:  # pragma: no cover - dynamic fallback
    """Dynamic attribute guard for phase placeholders.

    Raises:
        QiskitNotAvailableError: If a future backend symbol is accessed early.
        AttributeError: If the attribute is unknown.
    """
    if name in {"QiskitBackend", "register_operation_translator", "list_operation_translators"}:
        msg = (
            f"'{name}' is not available in Phase Q1/Q2 skeleton. Upgrade to a later version "
            "implementing the Qiskit backend."
        )
        raise QiskitNotAvailableError(msg)
    raise AttributeError(name)
