# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Open the SC device named by the license environment on a free processor."""

from __future__ import annotations

import json
import os
from pathlib import Path

from mqt.core.qdmi import slurm


def main() -> None:
    """Query the license-selected SC device and record the compute node."""
    job_id = os.environ["SLURM_JOB_ID"]
    device = slurm.open_device_from_license()
    result = {
        "device": device.name(),
        "job_id": job_id,
        "licenses": os.environ["SLURM_JOB_LICENSES"],
        "node": os.environ["SLURM_JOB_NODELIST"],
        "qubits": device.qubits_num(),
    }
    Path(f"/runtime/sc-{job_id}.json").write_text(json.dumps(result, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
