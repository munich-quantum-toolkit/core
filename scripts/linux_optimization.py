#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Record commands, with Linux and macOS resource accounting."""

from __future__ import annotations

# This local runner executes the command explicitly provided by its caller.
# ruff: file-ignore[subprocess-without-shell-equals-true, start-process-with-partial-path]
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path


def tree_rss(root_pid: int) -> int:
    """Sum descendant resident bytes, including Ninja's separate process groups.

    Returns:
        Aggregate resident bytes; shared pages may be counted repeatedly.
    """
    processes = {}
    if platform.system() == "Darwin":
        listing = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True)
        for line in listing.splitlines():
            pid, parent, rss = map(int, line.split())
            processes[pid] = (parent, rss * 1024)
    else:
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
                rss = int((proc / "statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
                processes[int(proc.name)] = (int(stat[1]), rss)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    descendants = {root_pid}
    while True:
        found = {pid for pid, (parent, _) in processes.items() if parent in descendants}
        if found <= descendants:
            break
        descendants |= found
    return sum(rss for pid, (_, rss) in processes.items() if pid in descendants)


def run(
    output: Path,
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    container: str | None = None,
    systemd_scope: str | None = None,
) -> int:
    """Run one command and retain its inputs, logs, timing, and RSS.

    Returns:
        The command exit status.

    Raises:
        ValueError: Linux-only resource accounting was requested on macOS.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    timing = output.with_suffix(".time.json")
    log = output.with_suffix(".log")
    record: dict[str, object] = {
        "command": command,
        "cwd": str(cwd.resolve()),
        "environment": environment,
        "started": time.time(),
        "container": container,
        "systemd_scope": systemd_scope,
        "platform": platform.platform(),
    }
    darwin = platform.system() == "Darwin"
    windows = platform.system() == "Windows"
    if (darwin or windows) and (container or systemd_scope):
        msg = "Container and systemd cgroup accounting require Linux"
        raise ValueError(msg)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    measured = [
        "/usr/bin/time",
        "-q",
        "-f",
        '{"user_seconds":%U,"system_seconds":%S,"max_process_rss_kib":%M,"command_elapsed_seconds":%e}',
        "-o",
        str(timing.resolve()),
        *command,
    ]
    if darwin:
        measured = ["/usr/bin/time", "-l", *command]
    elif windows:
        measured = command
    peak = 0
    cgroup_peak = 0
    swap_peak = 0
    cgroup = None
    free_start = shutil.disk_usage(cwd).free
    free_min = free_start
    started = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(
            measured,
            cwd=cwd,
            env=os.environ | environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            if not windows:
                peak = max(peak, tree_rss(process.pid))
            free_min = min(free_min, shutil.disk_usage(cwd).free)
            if container and cgroup is None:
                inspect = subprocess.run(
                    [
                        "docker",
                        "inspect",
                        "--format",
                        (
                            '{"pid":{{.State.Pid}},"image":"{{.Image}}","nano_cpus":{{.HostConfig.NanoCpus}},'
                            '"memory":{{.HostConfig.Memory}},"memory_swap":{{.HostConfig.MemorySwap}}}'
                        ),
                        container,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                parameters = json.loads(inspect.stdout) if inspect.returncode == 0 else {}
                if parameters.get("pid"):
                    record["container_parameters"] = parameters
                    try:
                        relative = (
                            Path("/proc", str(parameters["pid"]), "cgroup")
                            .read_text(encoding="utf-8")
                            .strip()
                            .split("::", 1)[1]
                        )
                        cgroup = Path("/sys/fs/cgroup") / relative.lstrip("/")
                    except (FileNotFoundError, IndexError):
                        pass
            if systemd_scope and cgroup is None:
                inspect = subprocess.run(
                    ["systemctl", "--user", "show", systemd_scope, "--property=ControlGroup", "--value"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if inspect.returncode == 0 and inspect.stdout.strip():
                    cgroup = Path("/sys/fs/cgroup") / inspect.stdout.strip().lstrip("/")
            if cgroup:
                try:
                    cgroup_peak = max(cgroup_peak, int((cgroup / "memory.peak").read_text()))
                    swap_peak = max(swap_peak, int((cgroup / "memory.swap.current").read_text()))
                except FileNotFoundError:
                    pass
            time.sleep(0.5)
    record |= {
        "elapsed_seconds": time.monotonic() - started,
        "returncode": process.returncode,
        "sampled_tree_rss_bytes": None if windows else peak,
        "cgroup_peak_bytes": cgroup_peak or None,
        "sampled_swap_bytes": swap_peak if cgroup else None,
        "filesystem_free_start_bytes": free_start,
        "filesystem_free_min_bytes": free_min,
    }
    if darwin:
        match = re.search(r"(\d+)\s+maximum resident set size", log.read_text())
        record["max_process_rss_kib"] = int(match[1]) // 1024 if match else None
    elif timing.exists():
        record |= json.loads(timing.read_text())
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return process.returncode


def main() -> None:
    """Measure one command or replay a previously recorded command.

    Raises:
        SystemExit: Propagate the measured command exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--container", help="Container name for cgroup memory accounting")
    parser.add_argument("--systemd-scope", help="Named systemd user scope for native cgroup memory accounting")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    environment = dict(value.split("=", 1) for value in args.env)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.replay:
        saved = json.loads(args.replay.read_text())
        command = saved["command"]
        environment = saved["environment"] | environment
        args.cwd = Path(saved["cwd"])
        args.container = args.container or saved.get("container")
        args.systemd_scope = args.systemd_scope or saved.get("systemd_scope")
    if not command:
        parser.error("provide a command after -- or a --replay record")
    raise SystemExit(run(args.output, command, args.cwd, environment, args.container, args.systemd_scope))


if __name__ == "__main__":
    main()
