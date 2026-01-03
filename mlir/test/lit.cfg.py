# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: INP001

"""LIT Configuration file for the MQT MLIR test suite.

This file configures the LLVM LIT testing infrastructure for MLIR dialect tests.
"""

from __future__ import annotations

from pathlib import Path

import lit.formats
from lit.llvm import llvm_config

# Use `lit_config` to access `config` from lit.site.cfg.py
config = globals().get("config")
if config is None:
    msg = "LIT config object is missing. Ensure lit.site.cfg.py is loaded first."
    raise RuntimeError(msg)

config.name = "MQT Core MLIR Lit Tests"
config.test_format = lit.formats.ShTest(execute_external=False)

# Define the file extensions to treat as test files.
config.suffixes = [".mlir"]

# Define the root path of where to look for tests.
config.test_source_root = Path(__file__).parent

# Define where to execute tests (and produce the output).
config.test_exec_root = Path(config.mqt_core_mlir_test_dir)

# For multi-config generators (MSVC), CMAKE_BUILD_TYPE may be empty or "$<CONFIG>"
# We need to detect the actual build directory at runtime
base_tool_dir_str = config.mqt_core_mlir_tools_dir

# Handle unexpanded generator expressions - remove the $<CONFIG> suffix if present
if base_tool_dir_str.endswith(("/$<CONFIG>", "\\$<CONFIG>")):
    base_tool_dir = Path(base_tool_dir_str.rsplit("$<CONFIG>", 1)[0].rstrip("/\\"))
else:
    base_tool_dir = Path(base_tool_dir_str)

# Try to find the quantum-opt tool
mqt_tool_dir = None

# First, check if tool exists in the base directory (single-config generators: Linux/macOS)
if (base_tool_dir / "quantum-opt").exists() or (base_tool_dir / "quantum-opt.exe").exists():
    mqt_tool_dir = str(base_tool_dir)
# If not found, search in multi-config subdirectories (Windows MSVC)
elif base_tool_dir.exists():
    for config_type in ["Release", "Debug", "RelWithDebInfo", "MinSizeRel"]:
        candidate = base_tool_dir / config_type
        if (candidate / "quantum-opt.exe").exists() or (candidate / "quantum-opt").exists():
            mqt_tool_dir = str(candidate)
            break

# Fall back to base directory if tool not found
if mqt_tool_dir is None:
    mqt_tool_dir = str(base_tool_dir)

tool_dirs = [config.llvm_tools_dir, mqt_tool_dir]
tools = ["not", "FileCheck", "quantum-opt"]
llvm_config.add_tool_substitutions(tools, tool_dirs)
