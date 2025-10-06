# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Custom exception types for QDMI Qiskit integration."""

from __future__ import annotations

__all__ = [
    "CapabilityMismatchError",
    "QDMIQiskitError",
    "TranslationError",
    "UnsupportedOperationError",
]


class QDMIQiskitError(RuntimeError):
    """Base class for QDMI Qiskit backend errors."""


class UnsupportedOperationError(QDMIQiskitError):
    """Raised when a circuit contains an operation unsupported by the backend/device."""


class TranslationError(QDMIQiskitError):
    """Raised when translation of a frontend instruction to neutral IR fails."""


class CapabilityMismatchError(QDMIQiskitError):
    """Raised when device capabilities do not match assumed backend configuration."""
