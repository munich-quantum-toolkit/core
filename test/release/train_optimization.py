#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Train the selected installed Core binaries with the measured benchmark corpus."""

from __future__ import annotations

# Training executes trusted build artifacts and tools.
# ruff: file-ignore[subprocess-without-shell-equals-true]
import argparse
import os
import subprocess
import sys
from pathlib import Path

from mqt import core


def main() -> None:
    """Run every selected workload and propagate failures to the optimizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    environment = os.environ | {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PATH": os.pathsep.join([
            str(Path(sys.executable).parent),
            str(Path(core.__file__).parent / "bin"),
            os.environ.get("PATH", ""),
        ]),
    }
    subprocess.run([sys.executable, str(root / "test/release/train_bolt.py")], cwd=root, env=environment, check=True)
    subprocess.run(
        [
            sys.executable,
            str(root / "test/release/benchmark_optimization.py"),
            "--training",
            "--repetitions",
            "1",
            "--expected-root",
            str(args.expected_root),
        ],
        cwd=root,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
