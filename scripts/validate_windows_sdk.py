#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Validate native Windows SDKs, repaired Core wheels, and installed CMake consumers."""

from __future__ import annotations

# Commands execute the selected local SDK and repository validation scripts.
# ruff: file-ignore[start-process-with-partial-path]
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from linux_optimization import run


def main() -> None:
    """Keep Windows release settings and record compatibility checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--full-python-tests", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Windows":
        parser.error("run this compatibility check in a Visual Studio developer shell on Windows")
    root, sdk = args.root.resolve(), args.sdk.resolve()
    project = Path(__file__).resolve().parents[1]
    root.mkdir(parents=True, exist_ok=True)
    env = {
        "DEPLOY": "ON",
        "CMAKE_BUILD_PARALLEL_LEVEL": "4",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MQT_CORE_QDMI_CONFIG_JSON": '{"schema-version":1,"qdmi":{"devices":[]}}',
        "PATH": str(sdk / "bin") + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": "",
    }

    def execute(label: str, command: list[str], extra: dict[str, str] | None = None) -> None:
        result = run(root / "measurements" / f"{label}.json", command, project, env | (extra or {}))
        if result:
            msg = f"{label} failed ({result}); see its measurement log"
            raise SystemExit(msg)

    build = root / "core-build"
    execute(
        "wheel-build",
        [
            "uv",
            "build",
            "--wheel",
            "--no-build-isolation",
            "--python",
            sys.executable,
            "--out-dir",
            str(root / "raw"),
            "-Cbuild-dir=" + str(build),
            "-Ccmake.define.ENABLE_IPO=OFF",
            "-Ccmake.define.BUILD_MQT_CORE_TESTS=OFF",
            "-Ccmake.define.LLVM_DIR=" + str(sdk / "lib/cmake/llvm"),
            "-Ccmake.define.MLIR_DIR=" + str(sdk / "lib/cmake/mlir"),
        ],
    )
    cpp_build = root / "cpp-build"
    execute(
        "cpp-configure",
        [
            "cmake",
            "--preset",
            "release",
            "-S",
            str(project),
            "-B",
            str(cpp_build),
            "-DENABLE_IPO=OFF",
            "-DLLVM_DIR=" + str(sdk / "lib/cmake/llvm"),
            "-DMLIR_DIR=" + str(sdk / "lib/cmake/mlir"),
        ],
    )
    execute("cpp-build", ["cmake", "--build", str(cpp_build), "--config", "Release", "-j", "4"])
    execute("cpp-tests", ["ctest", "--test-dir", str(cpp_build), "-C", "Release", "--output-on-failure", "-j", "4"])
    wheel = next((root / "raw").glob("*.whl"))
    execute(
        "repair",
        [
            sys.executable,
            "-m",
            "delvewheel",
            "repair",
            "-w",
            str(root / "artifacts"),
            str(wheel),
            "--namespace-pkg",
            "mqt",
            "--ignore-existing",
        ],
    )
    repaired = next((root / "artifacts").glob("*.whl"))
    venv = root / "installed"
    execute("venv", ["uv", "venv", "--python", sys.executable, str(venv)])
    python = venv / "Scripts/python.exe"
    execute("install", ["uv", "pip", "install", "--python", str(python), str(repaired)])
    requirements = root / "test-requirements.txt"
    with requirements.open("w") as stream:
        subprocess.run(
            [
                "uv",
                "export",
                "--frozen",
                "--no-default-groups",
                "--group",
                "test-base",
                "--no-emit-project",
                "--no-hashes",
            ],
            check=True,
            stdout=stream,
            cwd=project,
        )
    execute(
        "test-dependencies",
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            *(["-r", str(requirements)] if args.full_python_tests else ["--constraint", str(requirements), "numpy"]),
        ],
    )
    execute(
        "installed-checks",
        [
            str(python),
            str(project / "test/release/train_optimization.py"),
            "--expected-root",
            str(venv),
            *(["--tests"] if args.full_python_tests else []),
        ],
    )
    package = venv / "Lib/site-packages/mqt/core"
    dll_paths = sorted({str(path.parent) for path in (venv / "Lib/site-packages").rglob("*.dll")})
    consumer = root / "consumer"
    execute(
        "consumer-configure",
        [
            "cmake",
            "-S",
            str(project / "test/release/consumer"),
            "-B",
            str(consumer),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DENABLE_IPO=OFF",
            "-DCMAKE_PREFIX_PATH=" + str(package),
        ],
    )
    execute("consumer-build", ["cmake", "--build", str(consumer), "-j", "4"])
    execute("consumer-run", [str(consumer / "consumer.exe")], {"PATH": os.pathsep.join([*dll_paths, env["PATH"]])})
    with repaired.open("rb") as stream:
        wheel_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "core_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "visual_studio": os.environ.get("VCTOOLSVERSION"),
        "windows_sdk": os.environ.get("WINDOWSSDKVERSION"),
        "wheel_sha256": wheel_hash,
        "wheel_bytes": repaired.stat().st_size,
        "full_python_tests": args.full_python_tests,
        "python_test_limit": None
        if args.full_python_tests
        else "ARM64 retains the release dependency limit; numerical, compiler, QIR, and CLI checks run",
        "optimization_changes": False,
    }
    (root / "compatibility.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
