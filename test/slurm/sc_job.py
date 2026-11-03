# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Query the SC device with one allocated device license."""

from __future__ import annotations

import json
import os
from pathlib import Path

from mqt.core.qdmi import builtin_driver as qdmi_driver


def main() -> None:
    """Query the selected SC device and record the compute node."""
    job_id = os.environ["SLURM_JOB_ID"]
    device_id = os.environ["SLURM_JOB_LICENSES"].removesuffix(":1")
    device = qdmi_driver.open_device(device_id)
    result = {
        "device": device.name(),
        "job_id": job_id,
        "licenses": os.environ["SLURM_JOB_LICENSES"],
        "node": os.environ["SLURM_JOB_NODELIST"],
        "qubits": device.qubits_num(),
    }
    result_path = Path(f"/jobs/sc-{job_id}.json")
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    temporary.replace(result_path)


if __name__ == "__main__":
    main()
