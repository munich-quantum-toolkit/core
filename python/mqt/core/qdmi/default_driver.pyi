# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Configure MQT Core's packaged QDMI Client driver."""

import os

import mqt.core.qdmi

def add_manifest(manifest_path: str | os.PathLike) -> None:
    """Stage one installed package manifest before the default driver freezes."""

def open_device(
    device_id: str,
    *,
    driver_path: str | os.PathLike | None = None,
    base_url: str | None = None,
    token: str | None = None,
    auth_file: str | os.PathLike | None = None,
    auth_url: str | None = None,
    username: str | None = None,
    password: str | None = None,
    device_config: str | None = None,
    device_config_file: str | os.PathLike | None = None,
    custom1: str | None = None,
    custom2: str | None = None,
    custom3: str | None = None,
    custom4: str | None = None,
    custom5: str | None = None,
) -> mqt.core.qdmi.Device:
    """Open one device through MQT Core's strict private driver extension."""
