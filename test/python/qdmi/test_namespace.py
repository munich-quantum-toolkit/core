# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the QDMI Python namespace and the v3 compatibility module."""

from __future__ import annotations

import importlib
import sys

import pytest

from mqt.core import qdmi
from mqt.core.na import fomac as legacy_na
from mqt.core.na import qdmi as na_qdmi
from mqt.core.qdmi import driver


def test_qdmi_entities_live_in_qdmi_module() -> None:
    """Separate QDMI entities from the device-registry API."""
    assert qdmi.Device.Site.__module__ == "mqt.core.qdmi"
    assert qdmi.Device.Operation.__module__ == "mqt.core.qdmi"
    assert not hasattr(driver, "Device")


def test_fomac_module_warns_and_preserves_object_identity() -> None:
    """Keep v3 imports compatible without creating duplicate wrapper types."""
    sys.modules.pop("mqt.core.fomac", None)
    with pytest.warns(DeprecationWarning, match="mqt.core.fomac is deprecated"):
        legacy = importlib.import_module("mqt.core.fomac")

    qdmi_names = ("CustomProperty", "Device", "Job", "ProgramFormat")
    driver_names = (
        "DeviceDefinition",
        "Session",
        "open_device",
        "register_device",
        "register_device_if_absent",
        "registered_device_ids",
    )
    assert all(getattr(legacy, name) is getattr(qdmi, name) for name in qdmi_names)
    assert all(getattr(legacy, name) is getattr(driver, name) for name in driver_names)


def test_na_qdmi_namespace_preserves_v3_fomac_aliases() -> None:
    """Expose the neutral-atom device through QDMI while retaining v3 aliases."""
    assert na_qdmi.Device.__module__ == "mqt.core.na.qdmi"
    assert legacy_na.Device is na_qdmi.Device
    assert legacy_na.devices is na_qdmi.devices


def test_legacy_session_warns_on_construction() -> None:
    """Warn before MQT Core 4.0 removes the legacy session API."""
    with pytest.warns(DeprecationWarning, match="driver.Session is deprecated") as warnings:
        driver.Session()
    assert warnings[0].filename == __file__
