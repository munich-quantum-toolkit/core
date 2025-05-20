# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Useful commands for obtaining information about mqt-catalyst-plugin."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def include_dir() -> Path:
    """Return the path to the mqt-catalyst-plugin include directory.

    Raises:
        FileNotFoundError: If the include directory is not found.
        ImportError: If mqt-catalyst-plugin is not installed.
    """
    try:
        dist = distribution("mqt-catalyst-plugin")
        located_include_dir = Path(dist.locate_file("mqt/catalyst/include/mqt-catalyst-plugin"))
        if located_include_dir.exists() and located_include_dir.is_dir():
            return located_include_dir
        msg = "mqt-catalyst-plugin include files not found."
        raise FileNotFoundError(msg)
    except PackageNotFoundError:
        msg = "mqt-catalyst-plugin not installed, installation required to access the include files."
        raise ImportError(msg) from None


def cmake_dir() -> Path:
    """Return the path to the mqt-catalyst-plugin CMake module directory.

    Raises:
        FileNotFoundError: If the CMake module directory is not found.
        ImportError: If mqt-catalyst-plugin is not installed.
    """
    try:
        dist = distribution("mqt-catalyst-plugin")
        located_cmake_dir = Path(dist.locate_file("mqt/catalyst/share/cmake"))
        if located_cmake_dir.exists() and located_cmake_dir.is_dir():
            return located_cmake_dir
        msg = "mqt-catalyst-plugin CMake files not found."
        raise FileNotFoundError(msg)
    except PackageNotFoundError:
        msg = "mqt-catalyst-plugin not installed, installation required to access the CMake files."
        raise ImportError(msg) from None
