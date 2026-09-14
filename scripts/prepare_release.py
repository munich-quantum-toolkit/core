#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Train native SDK/Core PGO before cibuildwheel builds the release wheel."""

from __future__ import annotations

# /// script
# requires-python = ">=3.11"
# dependencies = ["cmake>=4.4.1", "ninja", "nanobind-backend>=1", "wheel"]
# ///
# Release preparation executes the build tools and artifacts in this checkout.
# ruff: file-ignore[subprocess-without-shell-equals-true, start-process-with-partial-path]
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path


def main() -> None:
    """Build, train, and prepare one Python ABI with fresh profiles.

    Raises:
        ValueError: The SDK retains ABI-breaking assertion checks.
    """
    project = Path(sys.argv[1]).resolve()
    base = Path(os.environ["MLIR_DIR"]).resolve().parents[2]
    if subprocess.check_output([str(base / "bin/llvm-config"), "--assertion-mode"], text=True).strip() != "OFF":
        msg = "Release PGO requires an assertion-free SDK"
        raise ValueError(msg)
    root = project / "build/release-pgo"
    output = project / "build/release-pgo.cmake"
    output.unlink(missing_ok=True)
    # The preceding ABI has already been repaired and tested by cibuildwheel.
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    sdk, build = root / "sdk", root / "core"
    shutil.copytree(base, sdk, symlinks=True)
    raw = root / "profiles"
    raw.mkdir()
    environment = os.environ | {"LLVM_PROFILE_FILE": str(root / "build-profiles/%m-%p.profraw")}
    run = partial(subprocess.run, cwd=project, env=environment, check=True)
    linux = platform.system() == "Linux"
    lto = "full" if linux else "thin"
    profdata = (
        "llvm-profdata" if linux else subprocess.check_output(["xcrun", "--find", "llvm-profdata"], text=True).strip()
    )
    linker = "-Wl,--lto-partitions=1,--no-relax,--build-id=sha1,--emit-relocs" if linux else "-Wl,-mllvm,-threads=1"
    definitions = {
        "MLIR_DIR": str(sdk / "lib/cmake/mlir"),
        "LLVM_DIR": str(sdk / "lib/cmake/llvm"),
        "ENABLE_IPO": "OFF",
        "CMAKE_JOB_POOLS": "release_links=1",
        "CMAKE_JOB_POOL_LINK": "release_links",
        **{f"CMAKE_{kind}_LINKER_FLAGS": linker for kind in ["EXE", "SHARED", "MODULE"]},
    }
    if linux:
        definitions["CMAKE_LINKER_TYPE"] = "LLD"
    instrumented = definitions | {
        "BUILD_MQT_CORE_TESTS": "ON",
        **{
            f"CMAKE_{language}_FLAGS": f"-flto={lto} -fprofile-generate -fprofile-update=atomic"
            for language in ["C", "CXX"]
        },
    }

    def build_wheel() -> Path:
        run([
            "uv",
            "build",
            str(project),
            "--wheel",
            "--python",
            sys.executable,
            "--out-dir",
            str(root / "wheel"),
            "-Cbuild-dir=" + str(build),
            "-Cinstall.strip=false",
            *[f"-Ccmake.define.{key}={value}" for key, value in instrumented.items()],
        ])
        return next((root / "wheel").glob("*.whl"))

    build_wheel()
    commands = subprocess.check_output(["ninja", "-C", str(build), "-t", "commands", "mqt-core-wheel"], text=True)
    targets = sorted(
        set(re.findall(r"lib((?:LLVM|MLIR)[A-Za-z0-9_]+)\.a", commands))
        & {path.stem.removeprefix("lib") for path in (base / "lib").glob("lib*.a")}
    )
    generators = {
        "LLVM_TABLEGEN": "llvm-tblgen",
        "MLIR_TABLEGEN": "mlir-tblgen",
        "MLIR_PDLL_TABLEGEN": "mlir-pdll",
        "MLIR_SRC_SHARDER_TABLEGEN": "mlir-src-sharder",
        "MLIR_LINALG_ODS_YAML_GEN": "mlir-linalg-ods-yaml-gen",
    }
    profile = root / "merged.profdata"
    for phase in ["generate", "use"]:
        run([
            "cmake",
            "-S",
            str(project / "build/release-tools/llvm-source/llvm"),
            "-B",
            str(root / "llvm"),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={sdk}",
            "-DLLVM_ENABLE_PROJECTS=mlir",
            "-DLLVM_TARGETS_TO_BUILD=host",
            "-DLLVM_ENABLE_ASSERTIONS=OFF",
            "-DLLVM_ENABLE_LTO=OFF",
            "-DLLVM_INSTALL_UTILS=ON",
            *[
                f"-DLLVM_{option}=OFF"
                for option in [
                    "INCLUDE_TESTS",
                    "INCLUDE_EXAMPLES",
                    "INCLUDE_BENCHMARKS",
                    "ENABLE_LIBXML2",
                    "ENABLE_LIBEDIT",
                    "ENABLE_LIBPFM",
                    "ENABLE_ZSTD",
                ]
            ],
            *[f"-D{key}={base / 'bin' / value}" for key, value in generators.items()],
            f"-DLLVM_BUILD_INSTRUMENTED={'IR' if phase == 'generate' else 'OFF'}",
            f"-DLLVM_PROFDATA_FILE={profile if phase == 'use' else ''}",
        ])
        run(["cmake", "--build", str(root / "llvm"), "--target", *targets])
        for component in ["llvm-headers", "mlir-headers", "cmake-exports", "mlir-cmake-exports"]:
            run(["cmake", "--install", str(root / "llvm"), "--component", component])
        for archive in (root / "llvm/lib").glob("*.a"):
            shutil.copyfile(archive, sdk / "lib" / archive.name)
        if phase == "generate":
            wheel = build_wheel()
            run(["cmake", "--build", str(build), "--target", "mlir/unittests/all"])
            training = os.environ | {"LLVM_PROFILE_FILE": str(raw / "%m-%p.profraw")}
            subprocess.run(
                ["ctest", "--test-dir", str(build / "mlir/unittests"), "--output-on-failure"], env=training, check=True
            )
            stage = root / "training"
            run([sys.executable, "-m", "wheel", "unpack", str(wheel), "-d", str(stage)])
            stage = next(stage.iterdir())
            (root / "qdmi.json").write_text('{"schema-version": 1, "qdmi": {"devices": []}}\n')
            training.update({"PYTHONPATH": str(stage), "MQT_CORE_QDMI_CONFIG_FILE": str(root / "qdmi.json")})
            training.pop("MQT_CORE_QDMI_CONFIG_JSON", None)
            subprocess.run([sys.executable, str(project / "test/release/train.py")], env=training, check=True)
            run([profdata, "merge", "-o", str(profile), *map(str, raw.glob("*.profraw"))])
            # Include profile contents in compiler-cache keys across ABI builds.
            with profile.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            profile = profile.rename(root / (digest + ".profdata"))

    # Keep profile-use flags out of CMake's unrelated compiler probes.
    flags = f"-flto={lto} [==[-fprofile-use={profile}]==]"
    output.write_text(
        "if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)\n"
        + "".join(
            f'  set({key} [==[{value}]==] CACHE STRING "Release optimization" FORCE)\n'
            for key, value in definitions.items()
        )
        + f"  add_compile_options({flags})\n  add_link_options({flags})\nendif()\n"
    )


if __name__ == "__main__":
    main()
