#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Apply BOLT to Core's hot Linux binaries before wheel repair and RECORD generation."""

from __future__ import annotations

# Release checks execute trusted build artifacts and tools from the build environment.
# ruff: file-ignore[subprocess-without-shell-equals-true, start-process-with-partial-path]
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    """Run the release optimization checks."""
    wheel, destination, project = map(Path, sys.argv[1:])
    with tempfile.TemporaryDirectory(prefix="mqt-wheel-bolt-") as directory:
        work = Path(directory)
        subprocess.run([sys.executable, "-m", "wheel", "unpack", str(wheel), "-d", str(work)], check=True)
        unpacked = next(work.glob("mqt_core-*"))
        core = unpacked / "mqt" / "core"
        venv = work / "venv"
        subprocess.run(["uv", "venv", "--python", sys.executable, str(venv)], check=True)
        python = venv / "bin" / "python"
        subprocess.run(["uv", "pip", "install", "--python", str(python), "nanobind-backend>=1"], check=True)
        environment = os.environ | {"PYTHONPATH": str(unpacked), "MQT_CORE_QDMI_CONFIG_FILE": str(work / "qdmi.json")}
        environment.pop("MQT_CORE_QDMI_CONFIG_JSON", None)
        (work / "qdmi.json").write_text('{"schema-version": 1, "qdmi": {"devices": []}}\n')
        training = [str(python), str(project.resolve() / "test" / "release" / "train_bolt.py")]
        binaries = [
            next(core.glob("dd.*.so")),
            next(core.rglob("libmqt-core-dd.so")),
            next(core.glob("mlir.*.so")),
            next(core.rglob("libmqt-core-qdmi-ddsim-device.so")),
            core / "bin" / "mqt-core-bench",
        ]
        for binary in binaries:
            subprocess.run(["mqt-bolt-optimize", str(binary), "--", *training], env=environment, check=True)
        subprocess.run(training, env=environment, check=True)
        for path in core.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    is_elf = stream.read(4) == b"\x7fELF"
                if is_elf:
                    subprocess.run(["llvm-strip", "--strip-unneeded", str(path)], check=True)
        subprocess.run(training, env=environment, check=True)
        packed = work / "packed"
        packed.mkdir()
        subprocess.run([sys.executable, "-m", "wheel", "pack", str(unpacked), "-d", str(packed)], check=True)
        repaired = work / "repaired"
        subprocess.run(["auditwheel", "repair", "-w", str(repaired), str(next(packed.glob("*.whl")))], check=True)
        result = next(repaired.glob("*.whl"))
        validated = work / "validated"
        subprocess.run([sys.executable, "-m", "wheel", "unpack", str(result), "-d", str(validated)], check=True)
        subprocess.run(training, env=environment | {"PYTHONPATH": str(next(validated.iterdir()))}, check=True)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result, destination)


if __name__ == "__main__":
    main()
