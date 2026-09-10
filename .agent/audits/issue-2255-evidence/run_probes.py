#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Record raw diagnostic outcomes without interpreting them as passing tests."""

from __future__ import annotations

import json
import resource
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[2]
BUILD = ROOT / "build/release"
BINARY = BUILD / "issue-2255-probe"
BASELINE = json.loads(FOLDER.joinpath("results-d994fe683.json").read_text())


def normalized_text(value: str | bytes | None) -> str:
    """Normalize only local path prefixes, keeping diagnostics intact.

    Returns:
        Diagnostic text with portable path prefixes.
    """
    text = value.decode() if isinstance(value, bytes) else (value or "")
    return text.replace(str(FOLDER) + "/", "").replace(str(ROOT) + "/", "")


def run(case: dict[str, Any]) -> dict[str, Any]:
    """Capture one process outcome with a bounded execution time.

    Returns:
        Raw exit status, output streams, and timeout status.
    """
    _, mode, input_path = case["command"]
    command = [str(BINARY), mode, str(FOLDER / input_path)]
    try:
        # The executable and all input paths are local audit artifacts, never shell commands.
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)  # ruff: ignore[subprocess-without-shell-equals-true]
        return {
            "command": ["probe", mode, input_path],
            "exit_code": result.returncode,
            "stdout": normalized_text(result.stdout),
            "stderr": normalized_text(result.stderr),
            "timeout": False,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "command": ["probe", mode, input_path],
            "exit_code": None,
            "stdout": normalized_text(error.stdout),
            "stderr": normalized_text(error.stderr),
            "timeout": True,
        }


resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
with ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(run, BASELINE))
BUILD.joinpath("issue-2255-results.json").write_text(json.dumps(results, indent=2) + "\n")
