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


def test_num_qubits(device_tuple: tuple[Device, Mapping[str, Any]]) -> None:
    """Test retrieving the number of qubits of the device."""
    device, device_dict = device_tuple
    assert device.qubits_num() == device_dict["numQubits"]


def test_length_unit(device_tuple: tuple[Device, Mapping[str, Any]]) -> None:
    """Test retrieving the length unit of the device."""
    device, device_dict = device_tuple
    assert device.length_unit() == device_dict["lengthUnit"]["unit"]


def test_length_scale_factor(device_tuple: tuple[Device, Mapping[str, Any]]) -> None:
    """Test retrieving the length scale factor of the device."""
    device, device_dict = device_tuple
    assert device.length_scale_factor() == device_dict["lengthUnit"]["scaleFactor"]


def test_duration_unit(device_tuple: tuple[Device, Mapping[str, Any]]) -> None:
    """Test retrieving the duration unit of the device."""
    device, device_dict = device_tuple
    assert device.duration_unit() == device_dict["durationUnit"]["unit"]


def test_duration_scale_factor(device_tuple: tuple[Device, Mapping[str, Any]]) -> None:
    """Test retrieving the duration scale factor of the device."""
    device, device_dict = device_tuple
    assert device.duration_scale_factor() == device_dict["durationUnit"]["scaleFactor"]


def test_min_atom_distance(device_tuple: tuple[Device, Mapping[str, Any]]) -> None:
    """Test retrieving the minimum atom distance of the device."""
    device, device_dict = device_tuple
    assert device.min_atom_distance() == device_dict["minAtomDistance"]


def test_t1(device_tuple: tuple[Device, Mapping[str, Any]]) -> None:
    """Test retrieving the T1 time of the device."""
    device, device_dict = device_tuple
    assert device.t1 == device_dict["decoherenceTimes"]["t1"]


def test_t2(device_tuple: tuple[Device, Mapping[str, Any]]) -> None:
    """Test retrieving the T2 time of the device."""
    device, device_dict = device_tuple
    assert device.t2 == device_dict["decoherenceTimes"]["t2"]


def test_traps(device_tuple: tuple[Device, Mapping[str, Any]]) -> None:
    """Test retrieving the traps of the device."""
    device, device_dict = device_tuple
    for trap, trap_dict in zip(device.traps, device_dict["traps"], strict=False):
        assert trap.lattice_origin.x == trap_dict["latticeOrigin"]["x"]
        assert trap.lattice_origin.y == trap_dict["latticeOrigin"]["y"]
        assert {
            (trap.lattice_vector_1.x, trap.lattice_vector_1.y),
            (trap.lattice_vector_2.x, trap.lattice_vector_2.y),
        } == {
            (trap_dict["latticeVector1"]["x"], trap_dict["latticeVector1"]["y"]),
            (trap_dict["latticeVector2"]["x"], trap_dict["latticeVector2"]["y"]),
        }
        assert trap.extent.origin.x == trap_dict["extent"]["origin"]["x"]
        assert trap.extent.origin.y == trap_dict["extent"]["origin"]["y"]
        assert trap.extent.size.width == trap_dict["extent"]["size"]["width"]
        assert trap.extent.size.height == trap_dict["extent"]["size"]["height"]
        for offset, offset_dict in zip(trap.sublattice_offsets, trap_dict["sublatticeOffsets"], strict=False):
            assert offset.x == offset_dict["x"]
            assert offset.y == offset_dict["y"]
