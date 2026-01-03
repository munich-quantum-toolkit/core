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

# Build tool search paths
# LLVM tools (FileCheck, not) come from LLVM installation
# quantum-opt comes from our build directory
base_tool_dir = Path(config.mqt_core_mlir_tools_dir)

# For multi-config generators, $<CONFIG> is expanded at build time by CMake
# For single-config generators, the path is the actual directory
# We need to check if the path contains a valid binary location
tool_dirs = [config.llvm_tools_dir]

# Check if base_tool_dir exists and contains quantum-opt
if base_tool_dir.exists():
    if (base_tool_dir / "quantum-opt").exists() or (base_tool_dir / "quantum-opt.exe").exists():
        tool_dirs.append(str(base_tool_dir))
    else:
        # For multi-config generators, check if parent directory exists
        # This handles cases where $<CONFIG> didn't expand properly
        parent_dir = base_tool_dir.parent
        if parent_dir.exists():
            # Search for quantum-opt in parent or common config directories
            found = False
            for candidate in [parent_dir] + [parent_dir / cfg for cfg in ["Release", "Debug", "RelWithDebInfo", "MinSizeRel"]]:
                if (candidate / "quantum-opt").exists() or (candidate / "quantum-opt.exe").exists():
                    tool_dirs.append(str(candidate))
                    found = True
                    break
            if not found:
                # Fallback: add parent directory anyway
                tool_dirs.append(str(parent_dir))
else:
    # Path doesn't exist - likely $<CONFIG> not expanded
    # Try parent directory
    parent_dir = base_tool_dir.parent
    if parent_dir.exists():
        tool_dirs.append(str(parent_dir))
        # Also try common config subdirectories
        for cfg in ["Release", "Debug", "RelWithDebInfo", "MinSizeRel"]:
            candidate = parent_dir / cfg
            if (candidate / "quantum-opt").exists() or (candidate / "quantum-opt.exe").exists():
                tool_dirs.append(str(candidate))
                break

tools = ["not", "FileCheck", "quantum-opt"]
llvm_config.add_tool_substitutions(tools, tool_dirs)
