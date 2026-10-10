#!/usr/bin/env -S uv run --no-project --offline --no-python-downloads
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Block a local Slurm device license until a site readiness probe succeeds."""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = logging.getLogger("qdmi-slurm-availability")


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
        Zero for readiness or block-only success, one for a failed probe, or two
        for a controller error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--license", required=True, metavar="ID:COUNT", help="local device ID and full configured capacity"
    )
    parser.add_argument("--checker", default="mqt-core-qdmi-check", help="site checker executable")
    parser.add_argument("--timeout", type=int, default=10, help="checker deadline in seconds (default: 10)")
    parser.add_argument("--block-only", action="store_true", help="close admission without running the checker")
    options = parser.parse_args(arguments)
    if re.fullmatch(r"[A-Za-z0-9_.-]+:[1-9][0-9]*", options.license) is None:
        parser.error("--license must be a local ID:COUNT with a positive count")
    if not 1 <= options.timeout <= 3600:
        parser.error("--timeout must be between 1 and 3600 seconds")

    device = options.license.rsplit(":", maxsplit=1)[0]
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
            f"Licenses={options.license}",
        )
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        LOGGER.exception("Could not block %s", device)
        return 2

    if options.block_only:
        return 0
    try:
        # The administrator selects the site executable; arguments use no shell.
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            (options.checker, "--device", device),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=options.timeout,
        )
    except (OSError, subprocess.SubprocessError):
        LOGGER.exception("Readiness probe failed; %s remains blocked", device)
        return 1
    if result.returncode != 0:
        LOGGER.error("Readiness probe failed; %s remains blocked", device)
        return 1

    try:
        scontrol("delete", f"ReservationName={name}")
    except (OSError, subprocess.SubprocessError):
        LOGGER.exception("Could not reopen %s", device)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
