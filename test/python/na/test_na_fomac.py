# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test NADevice class creation."""

from __future__ import annotations

import pathlib
from json import load
from typing import TYPE_CHECKING, Any

import pytest

from mqt.core.na.fomac import devices

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mqt.core.na.fomac import Device


@pytest.fixture
def device_tuple() -> tuple[Device, Mapping[str, Any]]:
    """Return a neutral atom FoMaC device instance."""
    with pathlib.Path("json/na/device.json").open(encoding="utf-8") as f:
        device_dict = load(f)
    return devices()[0], device_dict


def test_name(device_tuple: tuple[Device, Mapping[str, Any]]) -> None:
    """Test retrieving the name of the device."""
    device, device_dict = device_tuple
    assert device.name() == device_dict["name"]
