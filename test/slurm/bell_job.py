# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Execute a Bell circuit on the DDSIM device named by the license environment."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from mqt.core.qdmi import ProgramFormat, slurm

BELL_PROGRAM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""
SHOTS = 256


def main() -> None:
    """Run the circuit, write its result, and optionally retain the allocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hold", action="store_true")
    args = parser.parse_args()

    job_id = os.environ["SLURM_JOB_ID"]
    device = slurm.open_device_from_license()
    job = device.submit_job(BELL_PROGRAM, ProgramFormat.OPENQASM2, SHOTS)
    if not job.wait(60):
        msg = "DDSIM did not complete the Bell circuit within 60 seconds"
        raise RuntimeError(msg)

    counts = job.get_counts()
    if sum(counts.values()) != SHOTS:
        msg = f"Expected {SHOTS} Bell samples, got {sum(counts.values())}"
        raise RuntimeError(msg)
    if set(counts) != {"00", "11"}:
        msg = f"Expected only Bell outcomes 00 and 11, got {sorted(counts)}"
        raise RuntimeError(msg)

    result = {
        "counts": counts,
        "device": device.name(),
        "job_id": job_id,
        "licenses": os.environ["SLURM_JOB_LICENSES"],
        "node": os.environ["SLURM_JOB_NODELIST"],
        "shots": SHOTS,
    }
    runtime = Path("/runtime")
    (runtime / f"ddsim-{job_id}.json").write_text(json.dumps(result, sort_keys=True), encoding="utf-8")

    if args.hold:
        release = runtime / f"release-{job_id}"
        deadline = time.monotonic() + 180
        while not release.exists():
            if time.monotonic() >= deadline:
                msg = f"Release marker for Slurm job {job_id} did not appear"
                raise TimeoutError(msg)
            time.sleep(0.2)


if __name__ == "__main__":
    main()
