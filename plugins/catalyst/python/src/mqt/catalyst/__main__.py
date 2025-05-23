# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Command line interface for mqt-catalyst-plugin."""

from __future__ import annotations

import argparse
import sys

from ._version import version as __version__


def main() -> None:
    """Entry point for the mqt-catalyst-plugin command line interface.

    This function is called when running the `mqt-catalyst-plugin-cli` script.

    .. code-block:: bash

        mqt-catalyst-plugin-cli [--version] [--include_dir] [--cmake_dir]

    It provides the following command line options:

    - :code:`--version`: Print the version and exit.
    - :code:`--include_dir`: Print the path to the mqt-catalyst-plugin C++ include directory.
    - :code:`--cmake_dir`: Print the path to the mqt-catalyst-plugin CMake module directory.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=__version__, help="Print version and exit.")

    parser.add_argument(
        "--include_dir", action="store_true", help="Print the path to the mqt-catalyst-plugin C++ include directory."
    )
    parser.add_argument(
        "--cmake_dir", action="store_true", help="Print the path to the mqt-catalyst-plugin CMake module directory."
    )
    args = parser.parse_args()
    if not sys.argv[1:]:
        parser.print_help()
    if args.include_dir:
        pass
    if args.cmake_dir:
        pass


if __name__ == "__main__":
    main()
