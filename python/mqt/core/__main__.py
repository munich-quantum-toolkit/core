# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Launch the native MQT Core command."""

from __future__ import annotations

__lazy_modules__ = {"importlib.metadata", "pathlib", "subprocess", "typing"}

import os
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path
from typing import NoReturn


def main() -> NoReturn:
    """Run the bundled MQT compiler, benchmark, or simulator command."""
    suffix = ".exe" if sys.platform == "win32" else ""
    executable = Path(str(distribution("mqt-core").locate_file(f"mqt/core/bin/mqt-core{suffix}")))
    arguments = [str(executable), *sys.argv[1:]]
    if sys.platform == "win32":
        # Windows exec exits the launcher without waiting for the native tool.
        sys.exit(subprocess.run(arguments, check=False).returncode)  # ruff: ignore[subprocess-without-shell-equals-true]
    os.execv(executable, arguments)  # ruff: ignore[start-process-with-no-shell]


if __name__ == "__main__":
    main()
