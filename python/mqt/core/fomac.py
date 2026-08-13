# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compatibility imports for the former Python FoMaC namespace."""

from __future__ import annotations

import warnings

from . import qdmi as _qdmi
from .qdmi import driver as _driver

warnings.warn(
    "mqt.core.fomac is deprecated and will be removed in MQT Core 4.0; import mqt.core.qdmi and "
    "mqt.core.qdmi.driver instead",
    DeprecationWarning,
    stacklevel=2,
)

CustomProperty = _qdmi.CustomProperty
Device = _qdmi.Device
DeviceDefinition = _driver.DeviceDefinition
Job = _qdmi.Job
ProgramFormat = _qdmi.ProgramFormat
Session = _driver.Session
open_device = _driver.open_device
register_device = _driver.register_device
register_device_if_absent = _driver.register_device_if_absent
registered_device_ids = _driver.registered_device_ids

__all__ = [
    "CustomProperty",
    "Device",
    "DeviceDefinition",
    "Job",
    "ProgramFormat",
    "Session",
    "open_device",
    "register_device",
    "register_device_if_absent",
    "registered_device_ids",
]
