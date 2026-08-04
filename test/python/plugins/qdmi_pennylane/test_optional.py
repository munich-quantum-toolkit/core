# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test optional PennyLane dependency handling."""

from __future__ import annotations

import sys

from mqt.core.plugins import pennylane as plugin


def test_optional_dependency_matches_supported_python() -> None:
    """Keep the base Python 3.10 installation free of PennyLane."""
    assert ("QDMIDevice" in plugin.__all__) == (sys.version_info >= (3, 11))
