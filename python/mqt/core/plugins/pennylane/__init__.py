# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""PennyLane interface for gate-based QDMI devices."""

# ruff: file-ignore[non-empty-init-module]

from __future__ import annotations

import sys
from importlib import import_module
from typing import TYPE_CHECKING

try:
    import_module("pennylane")
except ModuleNotFoundError as error:
    if error.name != "pennylane":
        raise
    HAS_PENNYLANE = False
else:
    HAS_PENNYLANE = True

__all__ = ["HAS_PENNYLANE"]

if TYPE_CHECKING or (sys.version_info >= (3, 11) and HAS_PENNYLANE):
    from .device import DDSIMDevice, QDMIDevice
    from .exceptions import (
        PennyLaneConfigurationError,
        PennyLaneExecutionError,
        PennyLaneTranslationError,
        PennyLaneUnsupportedFormatError,
        PennyLaneUnsupportedOperationError,
        PennyLaneValidationError,
        QDMIPluginError,
    )

    __all__ += [
        "DDSIMDevice",
        "PennyLaneConfigurationError",
        "PennyLaneExecutionError",
        "PennyLaneTranslationError",
        "PennyLaneUnsupportedFormatError",
        "PennyLaneUnsupportedOperationError",
        "PennyLaneValidationError",
        "QDMIDevice",
        "QDMIPluginError",
    ]
