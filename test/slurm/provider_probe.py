# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Common assertions for provider workloads in the shared Slurm fixture."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from mqt.core.qdmi import slurm

if TYPE_CHECKING:
    from mqt.core.qdmi import Device


def open_device_from_license() -> Device:
    """Open the licensed device and require the catalogue's native library."""
    device = slurm.open_device_from_license()
    catalogue = Path(os.environ["MQT_CORE_QDMI_CONFIG_FILE"])
    definitions = json.loads(catalogue.read_text(encoding="utf-8"))["qdmi"]["devices"]
    device_id = os.environ["SLURM_JOB_LICENSES"].split(":", maxsplit=1)[0]
    definition = next(entry for entry in definitions if entry["id"] == device_id)
    library = (catalogue.parent / definition["library"]).resolve()
    native_prefix = Path("/opt/provider-native")
    if not library.is_relative_to(native_prefix):
        assert not native_prefix.exists()
    mappings = (line.split(maxsplit=5) for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines())
    loaded = {Path(fields[5]).resolve() for fields in mappings if len(fields) == 6 and fields[5].startswith("/")}
    assert {path for path in loaded if path.name == library.name} == {library}
    return device
