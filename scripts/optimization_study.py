#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Build and validate one portable SDK/Core experiment without changing release defaults."""

from __future__ import annotations

# The study executes selected local compilers, artifacts, and adjacent validation scripts.
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
import zipfile
from pathlib import Path

from linux_optimization import run


def digest(path: Path) -> str:
    """Return an artifact's SHA-256 digest."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Build a recorded variant and preserve every failed stage.

    Raises:
        RuntimeError: A built artifact or training profile violates the study contract.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["sdk", "wheel"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--llvm-source", type=Path, required=True)
    parser.add_argument("--llvm-source-id", required=True)
    parser.add_argument("--toolchain-repo", type=Path, required=True)
    parser.add_argument("--sdk-lto", choices=["OFF", "Thin", "Full"], required=True)
    parser.add_argument("--core-lto", choices=["OFF", "Thin", "Full"], default="OFF")
    parser.add_argument("--pgo", choices=["none", "core", "both"], default="none")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--lto-workers", type=int, default=1)
    parser.add_argument("--define", action="append", default=[])
    args = parser.parse_args()
    if args.jobs < 1 or args.lto_workers < 1:
        parser.error("worker counts must be positive")
    root, base = args.root.resolve(), args.sdk.resolve()
    project = Path(__file__).resolve().parents[1]
    helper = args.toolchain_repo.resolve() / "scripts/toolchain/build-library-variant.py"
    if not helper.is_file():
        parser.error("the toolchain checkout must contain build-library-variant.py")
    root.mkdir(parents=True, exist_ok=True)
    records = root / "measurements"
    records.mkdir(exist_ok=True)
    system = platform.system()
    if system not in {"Linux", "Darwin"}:
        parser.error("optimization experiments currently support Linux and macOS")
    compiler = shutil.which(os.environ.get("CC", "clang"))
    cxx = shutil.which(os.environ.get("CXX", "clang++"))
    if not compiler or not cxx:
        parser.error("set CC and CXX to the selected compiler")
    version = subprocess.check_output([cxx, "--version"], text=True)
    env = {
        "DEPLOY": "ON",
        "CMAKE_BUILD_PARALLEL_LEVEL": str(args.jobs),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MQT_CORE_QDMI_CONFIG_JSON": '{"schema-version":1,"qdmi":{"devices":[]}}',
        "PATH": str(base / "bin") + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": "",
    }
    manifest = {
        "core_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip(),
        "core_source_trees": {
            path: subprocess.check_output(["git", "rev-parse", "HEAD:" + path], cwd=project, text=True).strip()
            for path in [
                "bindings",
                "include",
                "src",
                "python",
                "mlir",
                "cmake",
                "vendor",
                "CMakeLists.txt",
                "pyproject.toml",
                "uv.lock",
            ]
        },
        "llvm_source_id": args.llvm_source_id,
        "compiler": cxx,
        "compiler_version": version,
        "compiler_sha256": digest(Path(cxx)),
        "sdk_lto": args.sdk_lto,
        "core_lto": args.core_lto,
        "pgo": args.pgo,
        "system": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "jobs": args.jobs,
        "lto_workers": args.lto_workers,
        "benchmark_sha256": digest(project / "test/release/benchmark_optimization.py"),
    }
    (root / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def execute(label: str, command: list[str], extra: dict[str, str] | None = None) -> None:
        result = run(records / f"{label}.json", command, project, env | (extra or {}))
        if result:
            msg = f"{label} failed ({result}); see {records / (label + '.log')}"
            raise SystemExit(msg)

    sdk = root / "sdk"

    def build_sdk(phase: str, profile: Path | None = None, targets: Path | None = None) -> None:
        command = [
            sys.executable,
            str(helper),
            "--source",
            str(args.llvm_source.resolve()),
            "--source-id",
            args.llvm_source_id,
            "--base-sdk",
            str(base),
            "--build",
            str(root / "sdk-build"),
            "--install",
            str(sdk),
            "--lto",
            args.sdk_lto,
            "--phase",
            phase,
            "--jobs",
            str(args.jobs),
        ]
        if profile:
            command += ["--profile", str(profile)]
        if targets:
            command += ["--targets", str(targets)]
        for definition in args.define:
            command += ["--define", definition]
        for variable in ["AR", "RANLIB"]:
            if os.environ.get(variable):
                command += ["--define", "CMAKE_" + variable + "=" + os.environ[variable]]
        if system == "Linux":
            command += ["--define", "LLVM_USE_LINKER=lld"]
        execute("sdk-" + phase, command)

    if args.operation == "sdk":
        if args.pgo != "none":
            parser.error("SDK PGO is trained by the wheel operation")
        build_sdk("plain")
        consumer = root / "sdk-consumer"
        execute(
            "sdk-consumer-configure",
            [
                "cmake",
                "-S",
                str(args.toolchain_repo.resolve() / "tests/integration"),
                "-B",
                str(consumer),
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_PREFIX_PATH=" + str(sdk),
                "-DCMAKE_CXX_SCAN_FOR_MODULES=OFF",
                "-DEXPECTED_LLVM_ASSERTIONS=OFF",
                *(["-DLLVM_USE_LINKER=lld"] if system == "Linux" else []),
            ],
        )
        execute("sdk-consumer-build", ["cmake", "--build", str(consumer), "-j", str(args.jobs)])
        execute("sdk-consumer-run", [str(consumer / "hello_mlir")])
        execute(
            "sdk-pack",
            [
                sys.executable,
                "-c",
                "import sys,tarfile; t=tarfile.open(sys.argv[1],'w:zst'); t.add(sys.argv[2],arcname='.'); t.close()",
                str(root / "sdk.tar.zst"),
                str(sdk),
            ],
        )
        manifest["archive_sha256"] = digest(root / "sdk.tar.zst")
        (root / "sdk-artifact.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return
    sdk_manifest = json.loads((base / "library-variant.json").read_text())
    if sdk_manifest["lto"] != args.sdk_lto or sdk_manifest["compiler_version"] != version:
        parser.error("the SDK library variant must match the requested LTO mode and compiler")
    if sdk.exists():
        parser.error("wheel experiments require a fresh root to prevent stale profiles or archives")
    shutil.copytree(base, sdk, symlinks=True)
    build = root / "core-build"
    requirements = root / "requirements.txt"
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
            cwd=project,
            stdout=stream,
            check=True,
        )
    manifest["requirements_sha256"] = digest(requirements)
    (root / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lto = "" if args.core_lto == "OFF" else "-flto=" + args.core_lto.lower()
    linker = (
        f"-Wl,--thinlto-jobs={args.lto_workers},--lto-partitions={args.lto_workers} -Wl,--no-relax,--build-id=sha1"
        if system == "Linux"
        else f"-Wl,-mllvm,-threads={args.lto_workers}"
    )

    def core_wheel(phase: str, profile: Path | None = None) -> Path:
        flags = lto
        if phase == "generate":
            flags += " -fprofile-generate -fprofile-update=atomic"
        elif profile:
            flags += " -fprofile-use=" + str(profile)
        command = [
            "uv",
            "build",
            str(project),
            "--wheel",
            "--python",
            sys.executable,
            "--no-build-isolation",
            "--out-dir",
            str(root / "raw" / phase),
            "-Cbuild-dir=" + str(build),
            "-Cinstall.strip=false",
        ]
        definitions = {
            "MLIR_DIR": str(sdk / "lib/cmake/mlir"),
            "LLVM_DIR": str(sdk / "lib/cmake/llvm"),
            "CMAKE_C_COMPILER": compiler,
            "CMAKE_CXX_COMPILER": cxx,
            "ENABLE_IPO": "OFF",
            "LLVM_ENABLE_LTO": "OFF",
            "ENABLE_BOLT": "ON" if system == "Linux" else "OFF",
            "BUILD_MQT_CORE_TESTS": "ON",
            "CMAKE_C_FLAGS": flags,
            "CMAKE_CXX_FLAGS": flags,
            "CMAKE_EXE_LINKER_FLAGS": linker,
            "CMAKE_SHARED_LINKER_FLAGS": linker,
            "CMAKE_MODULE_LINKER_FLAGS": linker,
            "CMAKE_PROJECT_INCLUDE": str(project / "test/release/optimization.cmake"),
            "CMAKE_JOB_POOLS": "study_links=1",
            "CMAKE_JOB_POOL_LINK": "study_links",
        }
        if system == "Linux":
            definitions["CMAKE_LINKER_TYPE"] = "LLD"
        for key in ["AR", "RANLIB"]:
            if os.environ.get(key):
                definitions["CMAKE_" + key] = os.environ[key]
        definitions.update(item.split("=", 1) for item in args.define)
        command += [f"-Ccmake.define.{key}={value}" for key, value in definitions.items()]
        execute("core-" + phase, command, {"LLVM_PROFILE_FILE": str(root / "build-profiles/%m-%p.profraw")})
        wheels = list((root / "raw" / phase).glob("*.whl"))
        if len(wheels) != 1:
            msg = "expected exactly one built wheel"
            raise RuntimeError(msg)
        return wheels[0]

    def unpack(wheel: Path, destination: Path) -> Path:
        execute(
            "unpack-" + destination.name, [sys.executable, "-m", "wheel", "unpack", str(wheel), "-d", str(destination)]
        )
        return next(destination.glob("mqt_core-*"))

    phase = "plain"
    profile = None
    if args.pgo != "none":
        generated = core_wheel("generate")
        targets = root / "pgo-targets.json"
        if args.pgo == "both":
            commands = subprocess.check_output(
                ["ninja", "-C", str(build), "-t", "commands", "mqt-core-wheel"], text=True
            )
            available = {path.stem.removeprefix("lib") for path in (sdk / "lib").glob("lib*.a")}
            names = sorted(set(re.findall(r"lib((?:LLVM|MLIR)[A-Za-z0-9_]+)\.a", commands)) & available)
            targets.write_text(json.dumps(names, indent=2) + "\n")
            build_sdk("generate", targets=targets)
            generated = core_wheel("generate")
        stage = unpack(generated, root / "training")
        raw = root / "profiles"
        raw.mkdir()
        execute(
            "train",
            [sys.executable, str(project / "test/release/train_optimization.py"), "--expected-root", str(stage)],
            {"PYTHONPATH": str(stage), "LLVM_PROFILE_FILE": str(raw / "%m-%p.profraw")},
        )
        log = (records / "train.log").read_text()
        if "LLVM Profile Error" in log:
            msg = "training reported a profile error"
            raise RuntimeError(msg)
        files = sorted(raw.glob("*.profraw"))
        if not files:
            msg = "training produced no profiles"
            raise RuntimeError(msg)
        profdata = shutil.which(os.environ.get("LLVM_PROFDATA", "llvm-profdata"))
        if not profdata:
            msg = "the compiler's llvm-profdata is required"
            raise RuntimeError(msg)
        profile = root / "merged.profdata"
        execute("merge", [profdata, "merge", "-o", str(profile), *map(str, files)])
        profile = profile.rename(root / (digest(profile) + ".profdata"))
        for component, function in [
            ("core", "_ZN4mlir3qco"),
            *([("sdk", "_ZN4mlir11MLIRContextC")] if args.pgo == "both" else []),
        ]:
            counts = subprocess.check_output(
                [profdata, "show", "--counts", "--function=" + function, str(profile)], text=True
            )
            values = re.findall(r"Function count: (\d+)", counts)
            blocks = re.findall(r"Block counts: \[([^]]*)\]", counts)
            if not any(int(value) > 0 for value in [*values, *re.findall(r"\d+", " ".join(blocks))]):
                msg = f"the {component} profile has no executed counters"
                raise RuntimeError(msg)
            (records / (component + "-profile.txt")).write_text(counts)
        if args.pgo == "both":
            build_sdk("use", profile=profile, targets=targets)
        phase = "use"
    wheel = core_wheel(phase, profile)
    execute("cpp-build", ["cmake", "--build", str(build), "-j", str(args.jobs)])
    driver = build / "src/qdmi/driver"
    runtime = [directory for directory in [build / "lib", build / "lib64"] if directory.is_dir()]
    for directory in runtime:
        for path in directory.iterdir():
            if path.is_file() and (path.suffix == ".json" or ".so" in path.name or path.suffix == ".dylib"):
                shutil.copy2(path, driver / path.name)
    execute(
        "cpp-tests",
        [sys.executable, str(project / "test/release/train_cpp_optimization.py"), str(build)],
        {
            "LD_LIBRARY_PATH": os.pathsep.join(map(str, runtime)),
            "LLVM_PROFILE_FILE": str(root / "validation-profiles/%m-%p.profraw"),
        },
    )
    artifacts = []
    for bolt in [False, True] if system == "Linux" else [False]:
        name = "bolt" if bolt else "plain"
        stage = unpack(wheel, root / ("stage-" + name))
        training = [
            sys.executable,
            str(project / "test/release/train_optimization.py"),
            "--tests",
            "--expected-root",
            str(stage),
        ]
        staged_env = {"PYTHONPATH": str(stage)}
        if bolt:
            core = stage / "mqt/core"
            binaries = [
                next(core.glob("dd.*.so")),
                next(core.rglob("libmqt-core-dd.so")),
                next(core.glob("mlir.*.so")),
                next(core.rglob("libmqt-core-qdmi-ddsim-device.so")),
                core / "bin/mqt-core-bench",
            ]
            for index, binary in enumerate(binaries):
                execute(
                    f"bolt-{index}", [str(base / "bin/mqt-bolt-optimize"), str(binary), "--", *training], staged_env
                )
        for path in (stage / "mqt/core").rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            with path.open("rb") as stream:
                magic = stream.read(4)
            if magic in {b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf"}:
                subprocess.run(
                    [
                        str(base / "bin/llvm-strip"),
                        "--strip-unneeded" if system == "Linux" else "--strip-debug",
                        str(path),
                    ],
                    check=True,
                )
                if system == "Darwin":
                    subprocess.run(["codesign", "--force", "--sign", "-", str(path)], check=True)
        packed = root / ("packed-" + name)
        execute("pack-" + name, [sys.executable, "-m", "wheel", "pack", str(stage), "-d", str(packed)])
        destination = root / "artifacts" / name
        destination.mkdir(parents=True)
        raw_wheel = next(packed.glob("*.whl"))
        repair = (
            ["auditwheel", "repair", "-w", str(destination), str(raw_wheel)]
            if system == "Linux"
            else ["delocate-wheel", "--require-archs", "arm64", "-w", str(destination), str(raw_wheel)]
        )
        execute("repair-" + name, repair)
        repaired = next(destination.glob("*.whl"))
        venv = root / "venvs" / name
        execute("venv-" + name, ["uv", "venv", "--python", sys.executable, str(venv)])
        python = venv / "bin/python"
        execute("dependencies-" + name, ["uv", "pip", "sync", "--python", str(python), str(requirements)])
        execute("install-" + name, ["uv", "pip", "install", "--no-deps", "--python", str(python), str(repaired)])
        execute(
            "validate-" + name,
            [str(python), str(project / "test/release/train_optimization.py"), "--tests", "--expected-root", str(venv)],
        )
        package = next((venv / "lib").glob("python*/site-packages/mqt/core"))
        consumer = root / ("consumer-" + name)
        execute(
            "consumer-configure-" + name,
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
                "-DCMAKE_CXX_COMPILER=" + cxx,
            ],
        )
        execute("consumer-build-" + name, ["cmake", "--build", str(consumer), "-j", "2"])
        execute("consumer-run-" + name, [str(consumer / "consumer")])
        with zipfile.ZipFile(repaired) as archive:
            uncompressed = sum(item.file_size for item in archive.infolist())
        artifacts.append({
            "name": name,
            "wheel": str(repaired),
            "wheel_sha256": digest(repaired),
            "compressed_bytes": repaired.stat().st_size,
            "uncompressed_bytes": uncompressed,
        })
    (root / "artifacts.json").write_text(json.dumps(manifest | {"artifacts": artifacts}, indent=2) + "\n")


if __name__ == "__main__":
    main()
