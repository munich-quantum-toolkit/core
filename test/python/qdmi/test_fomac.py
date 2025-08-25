# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the quantum computation IR."""

from __future__ import annotations

import pytest

from mqt.core.fomac import Device, DeviceStatus, Operation, Site, devices


@pytest.mark.parametrize("device", devices())
def test_device_name(device: Device) -> None:
    """Test that the device name is a non-empty string."""
    name = device.name()
    assert isinstance(name, str)
    assert len(name) > 0


@pytest.mark.parametrize("device", devices())
def test_device_version(device: Device) -> None:
    """Test that the device version is a non-empty string."""
    version = device.version()
    assert isinstance(version, str)
    assert len(version) > 0


@pytest.mark.parametrize("device", devices())
def test_device_status(device: Device) -> None:
    """Test that the device status is a valid DeviceStatus enum member."""
    status = device.status()
    assert isinstance(status, DeviceStatus)


@pytest.mark.parametrize("device", devices())
def test_device_library_version(device: Device) -> None:
    """Test that the device library version is a non-empty string."""
    lib_version = device.library_version()
    assert isinstance(lib_version, str)
    assert len(lib_version) > 0


@pytest.mark.parametrize("device", devices())
def test_device_qubits_num(device: Device) -> None:
    """Test that the device qubits number is a positive integer."""
    qubits_num = device.qubits_num()
    assert isinstance(qubits_num, int)
    assert qubits_num > 0


@pytest.mark.parametrize("device", devices())
def test_device_sites(device: Device) -> None:
    """Test that the device sites is a non-empty list of Site objects."""
    sites = device.sites()
    assert isinstance(sites, list)
    assert len(sites) > 0
    assert all(isinstance(site, Site) for site in sites)


@pytest.mark.parametrize("device", devices())
def test_device_operations(device: Device) -> None:
    """Test that the device operations is a non-empty list of Operation objects."""
    operations = device.operations()
    assert isinstance(operations, list)
    assert len(operations) > 0
    assert all(isinstance(op, Operation) for op in operations)


@pytest.mark.parametrize("device", devices())
def test_device_coupling_map(device: Device) -> None:
    """Test that the device coupling map is a list of tuples of Site objects."""
    with pytest.raises(RuntimeError, match=r".*Not supported.*"):
        cm = device.coupling_map()
    assert isinstance(cm, list)
    assert all(len(pair) == 2 for pair in cm)
    assert all(isinstance(site, Site) for pair in cm for site in pair)


@pytest.mark.parametrize("device", devices())
def test_device_needs_calibration(device: Device) -> None:
    """Test that the device needs calibration is an integer."""
    with pytest.raises(RuntimeError, match=r".*Not supported.*"):
        needs_cal = device.needs_calibration()
    assert isinstance(needs_cal, int)


@pytest.mark.parametrize("device", devices())
def test_device_length_unit(device: Device) -> None:
    """Test that the device length unit is a non-empty string."""
    with pytest.raises(RuntimeError, match=r".*Not supported.*"):
        lu = device.length_unit()
    assert isinstance(lu, str)
    assert len(lu) > 0


@pytest.mark.parametrize("device", devices())
def test_device_length_scale_factor(device: Device) -> None:
    """Test that the device length scale factor is a positive float."""
    with pytest.raises(RuntimeError, match=r".*Not supported.*"):
        lsf = device.length_scale_factor()
    assert isinstance(lsf, float)
    assert lsf > 0.0


@pytest.mark.parametrize("device", devices())
def test_device_duration_unit(device: Device) -> None:
    """Test that the device duration unit is a non-empty string."""
    with pytest.raises(RuntimeError, match=r".*Not supported.*"):
        du = device.duration_unit()
    assert isinstance(du, str)
    assert len(du) > 0


@pytest.mark.parametrize("device", devices())
def test_device_duration_scale_factor(device: Device) -> None:
    """Test that the device duration scale factor is a positive float."""
    with pytest.raises(RuntimeError, match=r".*Not supported.*"):
        dsf = device.duration_scale_factor()
    assert isinstance(dsf, float)
    assert dsf > 0.0


@pytest.mark.parametrize("device", devices())
def test_device_min_atom_distance(device: Device) -> None:
    """Test that the device minimum atom distance is a positive float."""
    with pytest.raises(RuntimeError, match=r".*Not supported.*"):
        mad = device.min_atom_distance()
    assert isinstance(mad, float)
    assert mad > 0.0
