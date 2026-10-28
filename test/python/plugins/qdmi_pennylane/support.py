# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test selection for optional PennyLane execution modes."""

from __future__ import annotations

from importlib.util import find_spec


def supports_catalyst() -> bool:
    """Return whether Catalyst is installed for this platform and Python version."""
    return find_spec("catalyst") is not None
