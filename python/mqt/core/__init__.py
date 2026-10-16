# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Core - The Backbone of the Munich Quantum Toolkit."""

from __future__ import annotations

__lazy_modules__ = {"mqt.core._version", "pathlib", "sysconfig"}

import sys

# On Windows, add MQT Core's DLL directory to the search path.
if sys.platform == "win32":  # ruff:ignore[non-empty-init-module] This is actually required on Windows
    import os
    import sysconfig
    from pathlib import Path

    def _dll_patch() -> None:
        bin_dir = Path(sysconfig.get_paths()["purelib"]) / "mqt" / "core" / "bin"
        os.add_dll_directory(str(bin_dir))

    _dll_patch()
    del _dll_patch


from mqt.core._version import version as __version__
from mqt.core._version import version_tuple as version_info
from mqt.core.load import load

__all__ = ["__version__", "load", "version_info"]
