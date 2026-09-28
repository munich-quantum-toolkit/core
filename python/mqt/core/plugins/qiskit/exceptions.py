# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Custom exception types for QDMI Qiskit integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.providers import JobError

if TYPE_CHECKING:
    from .job import QDMIJob

__all__ = [
    "CircuitValidationError",
    "JobExecutionError",
    "JobSubmissionError",
    "QDMIQiskitError",
    "TranslationError",
    "UnsupportedDeviceError",
    "UnsupportedFormatError",
    "UnsupportedOperationError",
]


def __dir__() -> list[str]:
    return __all__


class QDMIQiskitError(RuntimeError):
    """Base class for QDMI Qiskit backend errors."""


class UnsupportedDeviceError(QDMIQiskitError):
    """Raised when a QDMI device cannot be represented in Qiskit's Target model."""


class UnsupportedOperationError(QDMIQiskitError):
    """Raised when a circuit contains an operation unsupported by the backend/device."""


class TranslationError(QDMIQiskitError):
    """Raised when translation/conversion of a circuit to a program format fails."""


class CircuitValidationError(QDMIQiskitError):
    """Raised when a circuit fails validation (e.g., unbound parameters, invalid options)."""


class JobSubmissionError(QDMIQiskitError):
    """Raised when job submission to the QDMI device fails."""

    def __init__(self, message: str, *, job: QDMIJob | None = None) -> None:
        """Retain the batch handle for inspection and recovery."""
        super().__init__(message)
        self.job = job


class JobExecutionError(JobError):
    """An aggregate execution failure with a recoverable batch handle."""

    def __init__(self, message: str, *, job: QDMIJob) -> None:
        """Retain the batch handle for inspection and recovery."""
        super().__init__(message)
        self.job = job


class UnsupportedFormatError(QDMIQiskitError):
    """Raised when an unsupported program format is requested."""
