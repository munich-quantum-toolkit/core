# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Core - The Backbone of the Munich Quantum Toolkit."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Register bundled libraries before importing native extensions on Windows.
if sys.platform == "win32":  # ruff:ignore[non-empty-init-module] Native imports need the DLL search path.

    def _dll_patch() -> None:
        """Add bundled libraries to the Windows DLL search path."""
        import sysconfig  # ruff:ignore[import-outside-top-level] only used in Windows

        bin_dir = Path(sysconfig.get_paths()["purelib"]) / "mqt" / "core" / "bin"
        os.add_dll_directory(str(bin_dir))

    _dll_patch()
    del _dll_patch


from ._version import version as __version__
from ._version import version_tuple as version_info

__all__ = ["__version__", "version_info"]
