# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""PennyLane devices backed by gate-based QDMI providers."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ..._compat.optional import OptionalDependencyTester

HAS_PENNYLANE = OptionalDependencyTester(  # ruff:ignore[non-empty-init-module] Optional plugin
    "pennylane",
    install_msg="Install with 'pip install mqt-core[pennylane]'",
)

__all__ = ["HAS_PENNYLANE"]

if TYPE_CHECKING or (  # ruff:ignore[non-empty-init-module] Optional plugin
    sys.version_info >= (3, 11) and HAS_PENNYLANE
):
    from .converter import ConvertedProgram, convert_program
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
        "ConvertedProgram",
        "DDSIMDevice",
        "PennyLaneConfigurationError",
        "PennyLaneExecutionError",
        "PennyLaneTranslationError",
        "PennyLaneUnsupportedFormatError",
        "PennyLaneUnsupportedOperationError",
        "PennyLaneValidationError",
        "QDMIDevice",
        "QDMIPluginError",
        "convert_program",
    ]
