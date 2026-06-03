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

# under Windows, make sure to add the appropriate DLL directory to the PATH
if sys.platform == "win32":  # noqa: RUF067 This is actually required on Windows

    def _dll_patch() -> None:
        """Add the DLL directory to the PATH."""
        import sysconfig  # noqa: PLC0415 only used in Windows

        bin_dir = Path(sysconfig.get_paths()["purelib"]) / "mqt" / "core" / "bin"
        os.add_dll_directory(str(bin_dir))

    _dll_patch()
    del _dll_patch


from ._version import version as __version__
from ._version import version_tuple as version_info
from .load import load

__all__ = ["__version__", "load", "version_info"]
