# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Launch the structured benchmark driver bundled with MQT Core."""

from __future__ import annotations

from typing import NoReturn

from ._commands import run_tool


def main() -> NoReturn:
    """Replace this process with the bundled benchmark driver."""
    run_tool("mqt-core-bench")
