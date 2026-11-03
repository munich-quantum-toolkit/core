#!/usr/bin/env -S uv run --no-project --python 3.15 --offline --no-python-downloads
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Block a local Slurm device license until its QDMI availability probe succeeds."""

from __future__ import annotations

__lazy_modules__ = ["argparse", "json", "logging", "mqt.core.qdmi", "pathlib", "re", "subprocess"]

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mqt.core.qdmi import Device
from mqt.core.qdmi import builtin_driver as qdmi_driver

if TYPE_CHECKING:
    from collections.abc import Sequence


def probe(device_id: str) -> int:
    """Check one device in the isolated worker.

    Returns:
        Zero for an operational device, one for any failure.
    """
    try:
        device = qdmi_driver.open_device(device_id)
        return int(device.status() not in {Device.Status.IDLE, Device.Status.BUSY})
    except Exception:  # ruff: ignore[blind-except]
        # Device diagnostics may contain credentials.
        return 1


def scontrol(*arguments: str) -> str:
    """Run a bounded controller operation.

    Returns:
        The command's standard output.
    """
    # Slurm clients come from the administrator's PATH; no shell is involved.
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        ("scontrol", *arguments),  # ruff: ignore[start-process-with-partial-path]
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        timeout=10,
    ).stdout


def main(arguments: Sequence[str] | None = None) -> int:
    """Close admission before probing.

    Returns:
        Zero for an available device or block-only success, one for a failed probe, or two
        for a controller error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--license", required=True, metavar="ID:COUNT", help="local device ID and full configured capacity"
    )
    parser.add_argument("--timeout", type=int, default=10, help="probe deadline in seconds (default: 10)")
    parser.add_argument("--block-only", action="store_true", help="close admission without probing")
    parser.add_argument("--watch", action="store_true", help="wait 30 seconds between successful probes")
    options = parser.parse_args(arguments)
    if re.fullmatch(r"[A-Za-z0-9_.-]+:[1-9][0-9]*", options.license) is None:
        parser.error("--license must be a local ID:COUNT with a positive count")
    if not 1 <= options.timeout <= 3600:
        parser.error("--timeout must be between 1 and 3600 seconds")

    while True:
        result = update_availability(options.license, options.timeout, block_only=options.block_only)
        if result or not options.watch or options.block_only:
            return result
        time.sleep(30)


def update_availability(license_spec: str, timeout: int, *, block_only: bool) -> int:
    """Reserve the license, then release it only after a successful probe.

    Returns:
        Zero for success, one for a failed probe, or two for a controller error.
    """
    logger = logging.getLogger("qdmi-slurm-availability")
    device = license_spec.rsplit(":", maxsplit=1)[0]
    name = f"qdmi-unavailable-{device}"
    try:
        reservations = json.loads(scontrol("--json", "show", "reservations"))["reservations"]
        exists = any(reservation["name"] == name for reservation in reservations)
        # Active reservations cannot change their start time. Renew the end time
        # because Slurm implements Duration=infinite as one year.
        scontrol(
            "update" if exists else "create",
            f"ReservationName={name}",
            *(("StartTime=now",) if not exists else ()),
            "EndTime=now+365days",
            "Users=root",
            "Flags=LICENSE_ONLY,IGNORE_JOBS",
            f"Licenses={license_spec}",
        )
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        logger.exception("Could not block %s", device)
        return 2

    if block_only:
        return 0
    try:
        # Suppress diagnostics before importing QDMI; bound initialization,
        # status queries, and teardown. systemd cleans descendants on failure.
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            (sys.executable, str(Path(__file__).resolve()), "--probe", device),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        logger.exception("Availability probe failed; %s remains blocked", device)
        return 1
    if result.returncode != 0:
        logger.error("Availability probe failed; %s remains blocked", device)
        return 1

    try:
        scontrol("delete", f"ReservationName={name}")
    except (OSError, subprocess.SubprocessError):
        logger.exception("Could not reopen %s", device)
        return 2
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--probe":
        sys.exit(probe(sys.argv[2]))
    sys.exit(main())
