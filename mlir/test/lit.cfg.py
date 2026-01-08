# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
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

import sys
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

# Add substitutions for the required tools.
found = llvm_config.add_tool_substitutions(tools=["not", "FileCheck", "split-file"])
if not found:
    msg = "Could not find one or more required LLVM tools: 'not', 'FileCheck', 'split-file'."
    raise RuntimeError(msg)

# Successively check directories whether they contain the quantum-opt tool
tool_name = "quantum-opt" if sys.platform != "win32" else "quantum-opt.exe"
base_tool_dir = Path(config.mqt_core_mlir_tools_dir)

candidate_dirs = [base_tool_dir]
if config.cmake_build_type:
    candidate_dirs.append(base_tool_dir / config.cmake_build_type)
for cfg in ["Debug", "Release", "RelWithDebInfo", "MinSizeRel"]:
    cfg_dir = base_tool_dir / cfg
    if cfg_dir not in candidate_dirs:
        candidate_dirs.append(cfg_dir)

found = False
for candidate_dir in candidate_dirs:
    if (candidate_dir / tool_name).exists():
        llvm_config.add_tool_substitutions(["quantum-opt"], [str(candidate_dir)])
        found = True
        break

if not found:
    msg = f"Could not find {tool_name} anywhere under {base_tool_dir}."
    raise RuntimeError(msg)
