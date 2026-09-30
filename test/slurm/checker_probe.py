#!/usr/bin/python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Observe checker execution inside real Slurm tasks using non-secret test data."""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time

mode = os.environ.get("MQT_SLURM_CHECKER_MODE", "check")
child = None
if mode == "hang":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])

record = {
    "uid": os.getuid(),
    "euid": os.geteuid(),
    "gid": os.getgid(),
    "groups": os.getgroups(),
    "job": os.environ["SLURM_JOB_ID"],
    "step": os.environ.get("SLURM_STEP_ID", "batch"),
    "node": os.environ["SLURMD_NODENAME"],
    "pid": os.getpid(),
    "child": child.pid if child else None,
    "reference": os.environ.get("MQT_SLURM_TEST_REFERENCE"),
    "catalogue": os.environ.get("MQT_CORE_QDMI_CONFIG_FILE"),
    "arguments": sys.argv[1:],
}
with pathlib.Path(os.environ["MQT_SLURM_CHECKER_LOG"]).open("a", encoding="utf-8") as output:
    output.write(json.dumps(record) + "\n")

if mode == "hang":
    time.sleep(300)
elif mode == "fail":
    sys.stderr.write("test provider credential must stay private\n")
    sys.exit(1)
else:
    executable = "/usr/local/bin/mqt-core-qdmi-check"
    os.execv(executable, [executable, *sys.argv[1:]])  # ruff: ignore[start-process-with-no-shell]
