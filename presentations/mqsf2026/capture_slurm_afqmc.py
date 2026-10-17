# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Capture H2 shadows and dependent distributed walkers in the upstream Slurm playground.

Start examples/slurm/compose.yml with the chemistry dependencies installed and
the pinned AWS example mounted at /workload. Results and scheduler evidence stay
in the playground's shared /jobs directory. No hardware device is submitted.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, perf_counter_ns, sleep

ROOT = Path(__file__).resolve().parents[2]
FIELDS = (
    "JobIDRaw",
    "JobName",
    "State",
    "ExitCode",
    "Partition",
    "NodeList",
    "AllocCPUS",
    "AllocTRES",
    "Submit",
    "Start",
    "End",
)


def main() -> None:
    """Submit actual dependent jobs and retain only complete, successful allocations.

    Raises:
        ValueError: If Slurm identifiers or captured allocations disagree.
        RuntimeError: If a scheduled job fails.
        TimeoutError: If the workflow exceeds its bounded capture time.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--project", default="mqsf-afqmc")
    parser.add_argument("--compose-file", type=Path, help="Additional playground image/environment overlay")
    parser.add_argument("--snapshots", type=int, default=4096)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--walkers", type=int, default=128)
    parser.add_argument("--steps", type=int, default=240)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    environment = {**os.environ, "MQT_CORE_SLURM_RUNTIME": str(runtime), "MQT_CORE_SLURM_CORE": str(ROOT)}
    compose = ["docker", "compose", "--project-name", args.project, "--file", str(ROOT / "examples/slurm/compose.yml")]
    if args.compose_file:
        compose += ["--file", str(args.compose_file.resolve())]

    def command(*arguments: str) -> str:
        return subprocess.check_output(  # ruff: ignore[subprocess-without-shell-equals-true] - explicit argv, no shell
            [*compose, "exec", "-T", "--user", "10000:10000", "login", *arguments],
            env=environment,
            text=True,
        ).strip()

    started = monotonic()
    events = []
    revision = subprocess.check_output(  # ruff: ignore[subprocess-without-shell-equals-true] - read-only local Git query
        [shutil.which("git") or "/usr/bin/git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    capture = [
        "env",
        f"MQT_MQSF_CORE_REVISION={revision}",
        "MQT_CORE_DD_WORKER_BUDGET=4",
        "python3",
        "/workspace/presentations/mqsf2026/capture_afqmc.py",
        "--source",
        "/workload",
        "--snapshots",
        str(args.snapshots),
        "--shots",
        str(args.shots),
        "--walkers",
        str(args.walkers),
        "--steps",
        str(args.steps),
    ]

    def submit(name: str, allocation: list[str], payload: list[str]) -> str:
        job_id = command(
            "sbatch",
            "--parsable",
            f"--job-name={name}",
            "--time=00:30:00",
            "--chdir=/jobs",
            "--output=/jobs/slurm-%j.out",
            *allocation,
            "--wrap",
            shlex.join(payload),
        ).split(";", maxsplit=1)[0]
        if not job_id.isdecimal():
            msg = f"Invalid Slurm job identifier: {job_id!r}"
            raise ValueError(msg)
        events.append({
            "time_ms": (monotonic() - started) * 1000,
            "timestamp": datetime.now(UTC).isoformat(),
            "job_id": job_id,
            "operation": "sbatch",
            "state": "submitted",
            "name": name,
        })
        return job_id

    quantum = submit(
        "afqmc-shadows",
        [
            "--partition=quantum",
            "--licenses=mqt.ddsim.default:1",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=4",
            "--mem=8192M",
        ],
        [*capture, "--quantum-only", "--output", "/jobs/quantum.json"],
    )
    classical = submit(
        "afqmc-walkers",
        [
            "--partition=compute",
            "--nodes=4",
            "--ntasks=4",
            "--ntasks-per-node=1",
            "--cpus-per-task=1",
            "--mem=2048M",
            f"--dependency=afterok:{quantum}",
        ],
        [*capture, "--classical-input", "/jobs/quantum.json", "--output", "/jobs/afqmc.json"],
    )
    previous = {}
    deadline = monotonic() + 1800
    records = {}
    while monotonic() < deadline:
        output = command(
            "sacct",
            "-X",
            "--noheader",
            "--parsable2",
            f"--jobs={quantum},{classical}",
            "--format=" + ",".join(f"{field}%200" for field in FIELDS),
        )
        for line in output.splitlines():
            record = dict(zip(FIELDS, line.split("|"), strict=True))
            job_id, state = record["JobIDRaw"], record["State"]
            records[job_id] = record
            if previous.get(job_id) != state:
                events.append({
                    "time_ms": (monotonic() - started) * 1000,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "job_id": job_id,
                    "operation": "sacct",
                    "state": state,
                    "node": record["NodeList"],
                })
                previous[job_id] = state
            if state not in {"PENDING", "RUNNING", "COMPLETING", "COMPLETED"} or (
                state == "COMPLETED" and record["ExitCode"] != "0:0"
            ):
                command("scancel", classical)
                msg = f"Slurm job {job_id} failed: {record}; see {runtime / 'jobs' / f'slurm-{job_id}.out'}"
                raise RuntimeError(msg)
        if all(records.get(job_id, {}).get("State") == "COMPLETED" for job_id in (quantum, classical)):
            break
        sleep(0.25)
    else:
        command("scancel", quantum, classical)
        msg = "The AFQMC Slurm allocations did not complete within 30 minutes"
        raise TimeoutError(msg)
    path = runtime / "jobs/afqmc.json"
    data = json.loads(path.read_text())
    data["timeline"] = {
        "propagation_started_ms": (data["propagation"]["started_ns"] - data["execution"]["started_ns"]) / 1e6,
        "completed_ms": (perf_counter_ns() - data["execution"]["started_ns"]) / 1e6,
        "clock": "Shared Docker-host perf_counter_ns; elapsed from application preparation, including Slurm handoff",
    }
    data["slurm"] = {
        "version": command("scontrol", "--version"),
        "jobs": records,
        "events": events,
        "dependencies": [{"job_id": classical, "afterok": quantum}],
        "duration_ms": (monotonic() - started) * 1000,
        "clock": "Host monotonic time for observations; Slurm UTC accounting times retained separately",
        "scope": "Upstream Docker Slurm playground; nodes share one physical host",
    }
    if (
        data["execution"]["slurm"]["job_id"] != quantum
        or data["propagation"]["slurm"]["job_id"] != classical
        or len({worker["node"] for worker in data["propagation"]["workers"]}) != 4
        or len(data["propagation"]["workers"]) != 4
    ):
        msg = "The capture does not match the two requested Slurm allocations and four distributed ranks"
        raise ValueError(msg)
    path = path.with_name("afqmc-slurm.json")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(path)


if __name__ == "__main__":
    main()
