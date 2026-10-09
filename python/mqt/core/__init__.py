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

# Register bundled libraries before importing native extensions on Windows.
if sys.platform == "win32":  # ruff:ignore[non-empty-init-module] Native imports need the DLL search path.
    from importlib.metadata import distribution

    os.add_dll_directory(str(distribution("mqt-core").locate_file("mqt/core/bin")))


from ._version import version as __version__
from ._version import version_tuple as version_info

__all__ = ["__version__", "version_info"]
