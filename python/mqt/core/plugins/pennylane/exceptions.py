# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exception types for the QDMI PennyLane integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mqt.core.plugins.pennylane.job import PennyLaneJob

__all__ = [
    "PennyLaneConfigurationError",
    "PennyLaneExecutionError",
    "PennyLaneTranslationError",
    "PennyLaneValidationError",
    "QDMIPluginError",
]


def __dir__() -> list[str]:
    return __all__


class QDMIPluginError(RuntimeError):
    """Base class for QDMI PennyLane plugin errors."""


class PennyLaneConfigurationError(QDMIPluginError):
    """Raised when device or job configuration is invalid."""


class PennyLaneValidationError(QDMIPluginError):
    """Raised when a quantum program is invalid for the selected device."""


class PennyLaneTranslationError(QDMIPluginError):
    """Raised when a PennyLane program cannot be translated."""


class PennyLaneExecutionError(QDMIPluginError):
    """Raised when submission or execution of a QDMI job fails."""

    def __init__(self, message: str, *, job: PennyLaneJob | None = None) -> None:
        """Retain the batch handle for inspection and recovery."""
        super().__init__(message)
        self.job = job
