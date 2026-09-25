# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Configure the MQT Core QDMI driver."""

import os

import mqt.core.qdmi

def add_manifest(manifest_path: str | os.PathLike) -> None:
    """Register an installed device manifest before opening devices."""

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
    """Open an independent device session with the MQT Core QDMI driver."""
