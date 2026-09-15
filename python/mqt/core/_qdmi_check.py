# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Launch the QDMI device checker bundled with MQT Core."""

from __future__ import annotations

import os
import sys
from importlib.metadata import distribution
from pathlib import Path
from typing import NoReturn


def main() -> NoReturn:
    """Replace this process with the bundled QDMI device checker."""
    if sys.platform == "win32":
        sys.exit("The QDMI device checker requires Linux or macOS.")
    executable = Path(str(distribution("mqt-core").locate_file("mqt/core/bin/mqt-core-qdmi-check")))
    os.execv(executable, [str(executable), *sys.argv[1:]])  # ruff: ignore[start-process-with-no-shell]
