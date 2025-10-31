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
    "CircuitValidationError",
    "JobSubmissionError",
    "QDMIQiskitError",
    "TranslationError",
    "UnsupportedFormatError",
    "UnsupportedOperationError",
]


class QDMIQiskitError(RuntimeError):
    """Base class for QDMI Qiskit backend errors."""


class UnsupportedOperationError(QDMIQiskitError):
    """Raised when a circuit contains an operation unsupported by the backend/device."""


class TranslationError(QDMIQiskitError):
    """Raised when translation/conversion of a circuit to a program format fails."""


class CircuitValidationError(QDMIQiskitError):
    """Raised when a circuit fails validation (e.g., unbound parameters, invalid options)."""


class JobSubmissionError(QDMIQiskitError):
    """Raised when job submission to the QDMI device fails."""


class UnsupportedFormatError(QDMIQiskitError):
    """Raised when an unsupported program format is requested."""
