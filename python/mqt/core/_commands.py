# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Useful commands for obtaining information about mqt-core."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def include_dir() -> Path:
    """Return the path to the mqt-core include directory.

    Raises:
        FileNotFoundError: If the include directory is not found.
        ImportError: If mqt-core is not installed.
    """
    try:
        dist = distribution("mqt-core")
        located_include_dir = Path(str(dist.locate_file("mqt/core/include/mqt-core")))
        if located_include_dir.exists() and located_include_dir.is_dir():
            return located_include_dir
        msg = "mqt-core include files not found."
        raise FileNotFoundError(msg)
    except PackageNotFoundError:
        msg = "mqt-core not installed, installation required to access the include files."
        raise ImportError(msg) from None


def cmake_dir() -> Path:
    """Return the path to the mqt-core CMake module directory.

    Raises:
        FileNotFoundError: If the CMake module directory is not found.
        ImportError: If mqt-core is not installed.
    """
    try:
        dist = distribution("mqt-core")
        located_cmake_dir = Path(str(dist.locate_file("mqt/core/share/cmake")))
        if located_cmake_dir.exists() and located_cmake_dir.is_dir():
            return located_cmake_dir
        msg = "mqt-core CMake files not found."
        raise FileNotFoundError(msg)
    except PackageNotFoundError:
        msg = "mqt-core not installed, installation required to access the CMake files."
        raise ImportError(msg) from None
