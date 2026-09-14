#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Train release PGO and configure cibuildwheel's final wheel build."""

from __future__ import annotations

# Release preparation executes trusted build tools and the wheel it just built.
# ruff: file-ignore[subprocess-without-shell-equals-true, start-process-with-partial-path]
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def digest(path: Path) -> str:
    """Return the content identity of a build input."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_profile(profdata: str, profile: Path) -> dict[str, str]:
    """Require executed Core and SDK counters.

    Returns:
        The executed counter report for each requested component.

    Raises:
        RuntimeError: Training did not execute a requested component.
    """
    result = {}
    for component, function in [("core", "_ZN4mlir3qco"), ("sdk", "_ZN4mlir11MLIRContextC")]:
        counts = subprocess.check_output(
            [profdata, "show", "--counts", "--function=" + function, str(profile)], text=True
        )
        values = re.findall(r"Function count: (\d+)", counts)
        blocks = re.findall(r"Block counts: \[([^]]*)\]", counts)
        if not any(int(value) > 0 for value in [*values, *re.findall(r"\d+", " ".join(blocks))]):
            msg = f"The {component} profile has no executed counters"
            raise RuntimeError(msg)
        result[component] = counts
    return result


def main() -> None:
    """Prepare one ABI using fresh build directories and profiles.

    Raises:
        RuntimeError: The SDK or training output violates the release contract.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    args = parser.parse_args()
    system = platform.system()
    if system not in {"Linux", "Darwin"}:
        parser.error("release PGO requires Linux or macOS")
    lto = "full" if system == "Linux" else "thin"
    project = args.project.resolve()
    compiler = shutil.which(os.environ.get("CC", "clang"))
    cxx = shutil.which(os.environ.get("CXX", "clang++"))
    profdata = shutil.which(os.environ.get("LLVM_PROFDATA", "llvm-profdata"))
    if not compiler or not cxx or not profdata:
        parser.error("CC, CXX, and LLVM_PROFDATA must select the qualified Clang toolchain")
    base = Path(os.environ["MLIR_DIR"]).resolve().parents[2]
    if subprocess.check_output([str(base / "bin/llvm-config"), "--assertion-mode"], text=True).strip() != "OFF":
        parser.error("release PGO requires an assertion-free SDK")
    output = project / "build/release-pgo.cmake"
    output.parent.mkdir(exist_ok=True)
    output.unlink(missing_ok=True)
    root = Path(tempfile.mkdtemp(prefix="release-pgo-", dir=output.parent))
    sdk = root / "sdk"
    shutil.copytree(base, sdk, symlinks=True)
    build = root / "core-build"
    environment = os.environ | {
        "LLVM_PROFILE_FILE": str(root / "build-profiles/%m-%p.profraw"),
        "MQT_CORE_QDMI_CONFIG_FILE": str(root / "qdmi.json"),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    environment.pop("MQT_CORE_QDMI_CONFIG_JSON", None)
    (root / "qdmi.json").write_text('{"schema-version": 1, "qdmi": {"devices": []}}\n')

    def execute(label: str, command: list[str], extra: dict[str, str] | None = None) -> None:
        sys.stdout.write(f"Release stage: {label}\n")
        sys.stdout.flush()
        subprocess.run(command, cwd=project, env=environment | (extra or {}), check=True)

    linker = (
        "-Wl,--lto-partitions=1,--no-relax,--build-id=sha1,--emit-relocs"
        if system == "Linux"
        else "-Wl,-mllvm,-threads=1"
    )
    definitions = {
        "MLIR_DIR": str(sdk / "lib/cmake/mlir"),
        "LLVM_DIR": str(sdk / "lib/cmake/llvm"),
        "CMAKE_C_COMPILER": compiler,
        "CMAKE_CXX_COMPILER": cxx,
        "ENABLE_IPO": "OFF",
        "ENABLE_CACHE": "OFF",
        "LLVM_ENABLE_LTO": "OFF",
        "CMAKE_JOB_POOLS": "release_links=1",
        "CMAKE_JOB_POOL_LINK": "release_links",
        **{f"CMAKE_{kind}_LINKER_FLAGS": linker for kind in ["EXE", "SHARED", "MODULE"]},
    }
    if system == "Linux":
        definitions["CMAKE_LINKER_TYPE"] = "LLD"
    definitions.update({"CMAKE_" + key: os.environ[key] for key in ["AR", "RANLIB"] if key in os.environ})
    cache = shutil.which("sccache")
    if cache:
        definitions.update({f"CMAKE_{language}_COMPILER_LAUNCHER": cache for language in ["C", "CXX"]})

    def flags(value: str) -> dict[str, str]:
        return {f"CMAKE_{language}_FLAGS": f"-flto={lto} {value}" for language in ["C", "CXX"]}

    def generate(label: str) -> Path:
        destination = root / label
        execute(
            label,
            [
                "uv",
                "build",
                str(project),
                "--wheel",
                "--python",
                sys.executable,
                "--out-dir",
                str(destination),
                "-Cbuild-dir=" + str(build),
                "-Cinstall.strip=false",
                *[
                    f"-Ccmake.define.{key}={value}"
                    for key, value in (definitions | flags("-fprofile-generate -fprofile-update=atomic")).items()
                ],
            ],
        )
        wheels = list(destination.glob("*.whl"))
        if len(wheels) != 1:
            msg = "Expected exactly one instrumented wheel"
            raise RuntimeError(msg)
        return wheels[0]

    def build_sdk(phase: str, profile: Path | None = None) -> None:
        command = [
            sys.executable,
            str(base / "share/mqt-mlir/rebuild-libraries.py"),
            "--source",
            str(args.tools.resolve() / "llvm-source"),
            "--base-sdk",
            str(base),
            "--build",
            str(root / "sdk-build"),
            "--install",
            str(sdk),
            "--targets",
            str(root / "pgo-targets.json"),
        ]
        if profile:
            command += ["--profile", str(profile)]
        execute("sdk-" + phase, command)

    generate("core-generate")
    commands = subprocess.check_output(["ninja", "-C", str(build), "-t", "commands", "mqt-core-wheel"], text=True)
    available = {path.stem.removeprefix("lib") for path in (sdk / "lib").glob("lib*.a")}
    targets = sorted(set(re.findall(r"lib((?:LLVM|MLIR)[A-Za-z0-9_]+)\.a", commands)) & available)
    (root / "pgo-targets.json").write_text(json.dumps(targets, indent=2) + "\n")
    build_sdk("generate")
    generated = generate("core-generate-sdk")
    stage = root / "training"
    execute("unpack", [sys.executable, "-m", "wheel", "unpack", str(generated), "-d", str(stage)])
    stage = next(stage.glob("mqt_core-*"))
    raw = root / "profiles"
    raw.mkdir()
    training = subprocess.run(
        [sys.executable, str(project / "test/release/train_pgo.py"), "--expected-root", str(stage)],
        cwd=project,
        env=environment | {"PYTHONPATH": str(stage), "LLVM_PROFILE_FILE": str(raw / "%m-%p.profraw")},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    sys.stdout.write(training.stdout)
    training.check_returncode()
    profiles = sorted(raw.glob("*.profraw"))
    if not profiles or "LLVM Profile Error" in training.stdout:
        msg = "Training produced missing or invalid profiles"
        raise RuntimeError(msg)
    profile = root / "merged.profdata"
    execute("merge", [profdata, "merge", "-o", str(profile), *map(str, profiles)])
    profile = profile.rename(root / (digest(profile) + ".profdata"))
    for component, counts in check_profile(profdata, profile).items():
        (root / (component + "-profile.txt")).write_text(counts)
    build_sdk("use", profile)
    optimization_flags = f"-flto={lto} [==[-fprofile-use={profile}]==]"
    # Compiler probes have unrelated function profiles; apply PGO only to targets.
    output.write_text(
        "if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)\n"
        + "".join(
            f'  set({key} [==[{value}]==] CACHE STRING "Release optimization" FORCE)\n'
            for key, value in definitions.items()
        )
        + f"  add_compile_options({optimization_flags})\n"
        + f"  add_link_options({optimization_flags})\n"
        + "endif()\n"
    )
    sys.stdout.write(f"Release PGO ready: {root}\n")


if __name__ == "__main__":
    main()
