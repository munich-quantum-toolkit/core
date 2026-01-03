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
base_tool_dir = Path(config.mqt_core_mlir_tools_dir)

# Check for multi-config subdirectories (Release, Debug, etc.)
if config.cmake_build_type and config.cmake_build_type != "$<CONFIG>":
    multi_config_path = base_tool_dir / config.cmake_build_type
else:
    # Multi-config generator - search for the actual config directory
    multi_config_path = None
    for config_type in ["Release", "Debug", "RelWithDebInfo", "MinSizeRel"]:
        candidate = base_tool_dir / config_type
        if (candidate / "quantum-opt.exe").exists() or (candidate / "quantum-opt").exists():
            multi_config_path = candidate
            break

# Use the found path, or fall back to base directory
if multi_config_path and multi_config_path.exists():
    mqt_tool_dir = str(multi_config_path)
else:
    mqt_tool_dir = str(base_tool_dir)

tool_dirs = [config.llvm_tools_dir, mqt_tool_dir]
tools = ["not", "FileCheck", "quantum-opt"]
llvm_config.add_tool_substitutions(tools, tool_dirs)
