# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Submit the three AFQMC stages without holding idle allocations between them."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    """Submit dependent jobs and print their IDs and artifact directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mqt.ddsim.default")
    parser.add_argument("--snapshots", type=int, default=4096)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--walkers", type=int, default=128)
    parser.add_argument("--steps", type=int, default=240)
    args = parser.parse_args()
    if min(args.snapshots, args.shots, args.steps) < 1 or args.walkers < 2:
        parser.error("Positive counts and at least two walkers are required")
    if not args.device or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in args.device
    ):
        parser.error("--device must be one QDMI catalogue ID")
    directory = Path(tempfile.mkdtemp(prefix="afqmc-", dir="/jobs"))
    common = [
        "python",
        str(Path(__file__).with_name("workflow.py")),
        "--directory",
        str(directory),
        "--device",
        args.device,
        "--snapshots",
        str(args.snapshots),
        "--shots",
        str(args.shots),
        "--walkers",
        str(args.walkers),
        "--steps",
        str(args.steps),
    ]
    jobs = []
    try:  # ruff: ignore[too-many-statements-in-try-clause] - cancel every accepted dependent job on failure
        for stage, allocation, launcher in (
            (
                "prepare",
                ["--partition=compute", "--nodes=1", "--ntasks=1", "--cpus-per-task=2", "--mem=2048M"],
                ["srun"],
            ),
            (
                "quantum",
                [
                    "--partition=quantum",
                    "--nodes=1",
                    "--ntasks=1",
                    "--cpus-per-task=4",
                    "--mem=32768M",
                    f"--licenses={args.device}:1",
                ],
                ["srun"],
            ),
            (
                "classical",
                [
                    "--partition=compute",
                    "--nodes=4",
                    "--ntasks=4",
                    "--ntasks-per-node=1",
                    "--cpus-per-task=1",
                    "--mem=2048M",
                ],
                ["srun", "--mpi=pmix", "--kill-on-bad-exit=1"],
            ),
        ):
            dependency = [f"--dependency=afterok:{jobs[-1]}", "--kill-on-invalid-dep=yes"] if jobs else []
            script = "#!/bin/sh\nset -eu\n" + shlex.join([*launcher, *common, stage]) + "\n"
            result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - fixed argv, no shell
                [
                    "/opt/slurm/bin/sbatch",
                    "--parsable",
                    f"--job-name=afqmc-{stage}",
                    "--time=01:00:00",
                    "--chdir=/jobs",
                    f"--output={directory}/slurm-%j.out",
                    *allocation,
                    *dependency,
                ],
                input=script,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            identifier = result.stdout.strip().split(";", maxsplit=1)[0]
            if not identifier.isdecimal():
                msg = f"Invalid Slurm job identifier: {identifier!r}"
                raise ValueError(msg)  # ruff: ignore[raise-within-try] - cleanup covers parse failures too
            jobs.append(identifier)
        manifest = {
            "directory": str(directory),
            "device": args.device,
            "jobs": dict(zip(("prepare", "quantum", "classical"), jobs, strict=True)),
        }
        (directory / "jobs.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        sys.stdout.write(json.dumps(manifest) + "\n")
    except BaseException:
        if jobs:
            subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - validated numeric job IDs
                ["/opt/slurm/bin/scancel", *jobs], check=False, timeout=30
            )
        raise


if __name__ == "__main__":
    main()
