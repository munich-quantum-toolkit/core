# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exception types for QDMI Qiskit backend validation.

This module contains validation-related exception types used by the backend.
"""

from __future__ import annotations

__all__ = [
    "IRValidationError",
]


class IRValidationError(ValueError):
    """Raised when an IR consistency or capability validation check fails."""
