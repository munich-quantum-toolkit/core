# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Launch the native MQT Core command."""

from __future__ import annotations

__lazy_modules__ = {
    "importlib.metadata",
    "mqt.core._compile_qpy",
    "mqt.core._qdmi_discovery",
    "pathlib",
    "subprocess",
    "typing",
}

import os
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path
from typing import NoReturn

from mqt.core._compile_qpy import compile_qpy
from mqt.core._qdmi_discovery import discover_qdmi_manifests


def main() -> NoReturn:
    """Run a bundled MQT Core command."""
    suffix = ".exe" if sys.platform == "win32" else ""
    executable = Path(str(distribution("mqt-core").locate_file(f"mqt/core/bin/mqt-core{suffix}")))
    arguments = [str(executable), *sys.argv[1:]]
    if len(arguments) > 1 and arguments[1] in {"compile", "run", "simulate"}:
        manifests: list[str] = []
        discover_qdmi_manifests(lambda path: manifests.append(f"--qdmi-manifest={path}"))
        arguments[2:2] = manifests
    if len(arguments) > 1 and arguments[1] == "compile":
        try:
            result = compile_qpy(arguments)
        except (ValueError, RuntimeError, OSError) as error:
            sys.exit(f"QPY export failed: {error}")
        if result is not None:
            sys.exit(result)
    if sys.platform == "win32":
        # Windows exec exits the launcher without waiting for the native tool.
        sys.exit(subprocess.run(arguments, check=False).returncode)  # ruff: ignore[subprocess-without-shell-equals-true]
    os.execv(executable, arguments)  # ruff: ignore[start-process-with-no-shell]


if __name__ == "__main__":
    main()
