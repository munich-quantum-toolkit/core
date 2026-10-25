#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Build and run the documented CMake consumer against the installed wheel."""

# Commands use fixed arguments and the active Python/CMake toolchain.
# ruff: file-ignore[subprocess-without-shell-equals-true, start-process-with-partial-path]

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    """Check the driver, DDSIM bundle, and imported tools with the LLVM SDK."""
    source = (Path(__file__).resolve().parents[1] / "docs/cpp_api.md").read_text(encoding="utf-8")
    prefix = subprocess.check_output([sys.executable, "-m", "mqt.core", "--cmake_dir"], text=True).strip()
    # Reuse SDKs downloaded by the source build that executes these docs.
    project = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    sdks = project.glob("build/python/*/_deps/llvm-*")
    cmake_prefix_path = ";".join([prefix, *(str(sdk) for sdk in sdks)])
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for language, filename in (("cpp", "main.cpp"), ("cmake", "CMakeLists.txt")):
            code = source.split(f"```{language}\n", 1)[1].split("\n```", 1)[0]
            if language == "cmake":
                code += "\nadd_test(NAME compiler COMMAND MQT::mqt-cc --version)\n"
                code += "add_test(NAME benchmark COMMAND MQT::mqt-core-bench list)\n"
                code += "set_tests_properties(qdmi-runtime compiler benchmark PROPERTIES TIMEOUT 60)\n"
            (root / filename).write_text(code, encoding="utf-8")
        subprocess.run(
            [
                "cmake",
                "-S",
                directory,
                "-B",
                str(root / "build"),
                f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            check=True,
        )
        subprocess.run(["cmake", "--build", str(root / "build"), "--config", "Release"], check=True)
        # Use bundled devices without reading the host's QDMI configuration.
        config = root / "qdmi.json"
        config.write_text('{"schema-version": 1, "qdmi": {"devices": []}}', encoding="utf-8")
        subprocess.run(
            ["ctest", "--test-dir", str(root / "build"), "-C", "Release", "--output-on-failure", "--no-tests=error"],
            check=True,
            env={
                **os.environ,
                "MQT_CORE_QDMI_CONFIG_FILE": str(config),
                "MQT_CORE_QDMI_CONFIG_JSON": config.read_text(encoding="utf-8"),
            },
        )


if __name__ == "__main__":
    main()
