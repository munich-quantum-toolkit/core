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

import contextlib
from pathlib import Path

import lit.formats
from lit.llvm import llvm_config

from mqt.core.plugins.catalyst import get_catalyst_plugin_abs_path

# Use `lit_config` to access `config` from lit.site.cfg.py
config = globals().get("config")
if config is None:
    msg = "LIT config object is missing. Ensure lit.site.cfg.py is loaded first."
    raise RuntimeError(msg)

config.name = "MQT MLIR Catalyst Plugin test suite"
config.test_format = lit.formats.ShTest(execute_external=True)

# Define the file extensions to treat as test files.
config.suffixes = [".mlir"]
config.excludes = ["lit.cfg.py"]

# Define the root path of where to look for tests.
config.test_source_root = Path(__file__).parent

# Define where to execute tests (and produce the output).
config.test_exec_root = getattr(config, "plugin_test_dir", ".lit")

# Define PATH to include the tools needed for our tests.
with contextlib.suppress(AttributeError):
    # From within a build target we have access to cmake variables configured in lit.site.cfg.py.in.
    llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)  # FileCheck

# Dynamic plugin path substitution (prefers installed wheel; env override supported by the helper)
plugin_path = get_catalyst_plugin_abs_path()
config.substitutions.append(("%mqt_plugin_path%", str(plugin_path)))
