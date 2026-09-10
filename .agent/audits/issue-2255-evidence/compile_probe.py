#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Build the documented probe with the configured compiler-test flags."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[2]
BUILD = ROOT / "build/release"
SOURCE = BUILD / "issue-2255-probe.cpp"
BINARY = SOURCE.with_suffix("")
TARGET = "mqt-core-mlir-unittests-compiler"

SOURCE.write_text(FOLDER.joinpath("README.md").read_text().split("```cpp\n", 1)[1].split("```", 1)[0])
entries = json.loads(BUILD.joinpath("compile_commands.json").read_text())
entry = next(e for e in entries if e["file"].endswith("/test_compiler_pipeline.cpp"))
compile_args = shlex.split(entry["command"])
compile_args[compile_args.index("-o") + 1] = str(SOURCE.with_suffix(".o"))
compile_args[compile_args.index("-c") + 1] = str(SOURCE)
subprocess.run(compile_args, cwd=entry["directory"], check=True)  # ruff: ignore[subprocess-without-shell-equals-true] -- Trusted local CMake command, without a shell.

commands = subprocess.check_output(  # ruff: ignore[subprocess-without-shell-equals-true] -- Fixed Ninja query, without a shell.
    [shutil.which("ninja") or "ninja", "-t", "commands", TARGET], cwd=BUILD, text=True
).splitlines()
line = next(c for c in reversed(commands) if f" -o mlir/unittests/Compiler/{TARGET} " in c)
link_args = shlex.split(line)
while link_args and link_args[0] in {":", "&&"}:
    link_args.pop(0)
if "&&" in link_args:
    link_args = link_args[: link_args.index("&&")]
link_args = [a for a in link_args if not (f"/{TARGET}.dir/" in a and a.endswith(".o"))]
link_args.insert(1, str(SOURCE.with_suffix(".o")))
link_args[link_args.index("-o") + 1] = str(BINARY)
subprocess.run(link_args, cwd=BUILD, check=True)  # ruff: ignore[subprocess-without-shell-equals-true] -- Trusted local Ninja command, without a shell.
