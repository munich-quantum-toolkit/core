#!/opt/venv/bin/python
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Register a playground QAN with each configured QDMI HRES layer."""

from __future__ import annotations

import json
import os
import subprocess
import sys


def main(configuration: str) -> None:
    """Read the controller's layers and start slurmd with their membership."""
    result = subprocess.run(
        ("/opt/slurm/bin/scontrol", "--json", "show", "licenses"),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    layers = [
        {"hres_name": entry["LicenseName"], "layer_name": entry["LayerName"]}
        for entry in json.loads(result.stdout)["licenses"]
        if entry.get("LayerName") == "quantum"
    ]
    if not layers:
        msg = "No quantum HRES layers are configured"
        raise RuntimeError(msg)
    membership = json.dumps(layers, separators=(",", ":"))
    # The executable and flags are fixed; JSON comes from the trusted Slurm controller.
    os.execv(  # ruff: ignore[start-process-with-no-shell]
        "/opt/slurm/sbin/slurmd", ("slurmd", "--systemd", "-Z", "--conf", f"{configuration} HRES={membership}")
    )


if __name__ == "__main__":
    main(sys.argv[1])
