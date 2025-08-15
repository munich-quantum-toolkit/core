# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: INP001

"""LIT Configuration file for the MQT MLIR Catalyst Plugin test suite.

This file configures the LLVM LIT testing infrastructure for MLIR dialect tests.
"""
from __future__ import annotations
from pathlib import Path
import lit.formats

config = globals().get("config")
if config is None:
    raise RuntimeError("LIT config object is missing. Ensure lit.site.cfg.py is loaded first.")

config.name = "MQT MLIR Catalyst Plugin test suite"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir"]
config.excludes = ["lit.cfg.py"]
config.test_source_root = Path(__file__).parent
config.test_exec_root = getattr(config, "plugin_test_dir", ".lit")

# Optional: ensure FileCheck is available if CMake provided a tools dir
try:
    from lit.llvm import llvm_config
    if getattr(config, "llvm_tools_dir", None):
        llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
except Exception:
    pass  # non-fatal; assume FileCheck is on PATH

# Dynamic plugin path substitution (prefers installed wheel; env override supported by the helper)
from mqt.core.plugins.catalyst import get_catalyst_plugin_abs_path
plugin_path = get_catalyst_plugin_abs_path()
config.substitutions.append(("%mqt_plugin_path%", str(plugin_path)))