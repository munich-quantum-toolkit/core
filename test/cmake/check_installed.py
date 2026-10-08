# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Build and run a C++ consumer of the installed wheel."""

from __future__ import annotations

# This standalone check executes trusted build tools and the installed test binary.
# ruff: file-ignore[implicit-namespace-package, subprocess-without-shell-equals-true, start-process-with-partial-path]
import subprocess
import tempfile
from importlib.metadata import distribution
from pathlib import Path


def main() -> None:
    """Check installed libraries, package settings, and the device helper."""
    source = Path(__file__).parent / "installed_consumer"
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                directory,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_PREFIX_PATH=" + str(distribution("mqt-core").locate_file("mqt/core")),
            ],
            check=True,
        )
        subprocess.run(["cmake", "--build", directory, "--config", "Release", "--parallel", "2"], check=True)
        subprocess.run(["ctest", "--test-dir", directory, "-C", "Release", "--output-on-failure"], check=True)


if __name__ == "__main__":
    main()
