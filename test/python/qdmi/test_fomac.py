# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the quantum computation IR."""

from __future__ import annotations

from re import match
from typing import cast

import pytest

from mqt.core.qdmi.fomac import Device, DeviceStatus, Operation, Site, devices


@pytest.fixture(params=devices())
def device(request: pytest.FixtureRequest) -> Device:
    """Fixture to provide a device for testing.

    Returns:
        Device: A quantum device instance.
    """
    return cast("Device", request.param)


@pytest.fixture(params=devices())
def device_and_site(request: pytest.FixtureRequest) -> tuple[Device, Site]:
    """Fixture to provide a device for testing.

    Returns:
        tuple[Device, Site]: A tuple containing a quantum device instance and one of its sites.
    """
    device = request.param
    site = device.sites()[0]
    return device, site


@pytest.fixture(params=devices())
def device_and_operation(request: pytest.FixtureRequest) -> tuple[Device, Operation]:
    """Fixture to provide a device for testing.

    Returns:
        tuple[Device, Operation]: A tuple containing a quantum device instance and one of its operations.
    """
    device = request.param
    operation = device.operations()[0]
    return device, operation


def test_device_name(device: Device) -> None:
    """Test that the device name is a non-empty string."""
    name = device.name()
    assert isinstance(name, str)
    assert len(name) > 0


def test_device_version(device: Device) -> None:
    """Test that the device version is a non-empty string."""
    version = device.version()
    assert isinstance(version, str)
    assert len(version) > 0


def test_device_status(device: Device) -> None:
    """Test that the device status is a valid DeviceStatus enum member."""
    status = device.status()
    assert isinstance(status, DeviceStatus)


def test_device_library_version(device: Device) -> None:
    """Test that the device library version is a non-empty string."""
    lib_version = device.library_version()
    assert isinstance(lib_version, str)
    assert len(lib_version) > 0


def test_device_qubits_num(device: Device) -> None:
    """Test that the device qubits number is a positive integer."""
    qubits_num = device.qubits_num()
    assert isinstance(qubits_num, int)
    assert qubits_num > 0


def test_device_sites(device: Device) -> None:
    """Test that the device sites is a non-empty list of Site objects."""
    sites = device.sites()
    assert isinstance(sites, list)
    assert len(sites) > 0
    assert all(isinstance(site, Site) for site in sites)


def test_device_operations(device: Device) -> None:
    """Test that the device operations is a non-empty list of Operation objects."""
    operations = device.operations()
    assert isinstance(operations, list)
    assert len(operations) > 0
    assert all(isinstance(op, Operation) for op in operations)


def test_device_coupling_map(device: Device) -> None:
    """Test that the device coupling map is a list of tuples of Site objects."""
    try:
        cm = device.coupling_map()
        assert isinstance(cm, list)
        assert all(len(pair) == 2 for pair in cm)
        assert all(isinstance(site, Site) for pair in cm for site in pair)
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_device_needs_calibration(device: Device) -> None:
    """Test that the device needs calibration is an integer."""
    try:
        needs_cal = device.needs_calibration()
        assert isinstance(needs_cal, int)
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_device_length_unit(device: Device) -> None:
    """Test that the device length unit is a non-empty string."""
    try:
        lu = device.length_unit()
        assert isinstance(lu, str)
        assert len(lu) > 0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_device_length_scale_factor(device: Device) -> None:
    """Test that the device length scale factor is a positive float."""
    try:
        lsf = device.length_scale_factor()
        assert isinstance(lsf, float)
        assert lsf > 0.0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_device_duration_unit(device: Device) -> None:
    """Test that the device duration unit is a non-empty string."""
    try:
        du = device.duration_unit()
        assert isinstance(du, str)
        assert len(du) > 0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_device_duration_scale_factor(device: Device) -> None:
    """Test that the device duration scale factor is a positive float."""
    try:
        dsf = device.duration_scale_factor()
        assert isinstance(dsf, float)
        assert dsf > 0.0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_device_min_atom_distance(device: Device) -> None:
    """Test that the device minimum atom distance is a positive float."""
    try:
        mad = device.min_atom_distance()
        assert isinstance(mad, int)
        assert mad > 0.0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_site_index(device_and_site: tuple[Device, Site]) -> None:
    """Test that the site index is a non-negative integer."""
    _device, site = device_and_site
    index = site.index()
    assert isinstance(index, int)
    assert index >= 0


def test_site_t1(device_and_site: tuple[Device, Site]) -> None:
    """Test that the site T1 coherence time is a positive integer."""
    _device, site = device_and_site
    try:
        t1 = site.t1()
        assert isinstance(t1, int)
        assert t1 > 0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_site_t2(device_and_site: tuple[Device, Site]) -> None:
    """Test that the site T2 coherence time is a positive integer."""
    _device, site = device_and_site
    try:
        t2 = site.t2()
        assert isinstance(t2, int)
        assert t2 > 0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_site_name(device_and_site: tuple[Device, Site]) -> None:
    """Test that the site name is a non-empty string."""
    _device, site = device_and_site
    try:
        name = site.name()
        assert isinstance(name, str)
        assert len(name) > 0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_site_x_coordinate(device_and_site: tuple[Device, Site]) -> None:
    """Test that the site x coordinate is an integer."""
    _device, site = device_and_site
    try:
        x = site.x_coordinate()
        assert isinstance(x, int)
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_site_y_coordinate(device_and_site: tuple[Device, Site]) -> None:
    """Test that the site y coordinate is an integer."""
    _device, site = device_and_site
    try:
        y = site.y_coordinate()
        assert isinstance(y, int)
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_site_z_coordinate(device_and_site: tuple[Device, Site]) -> None:
    """Test that the site z coordinate is an integer."""
    _device, site = device_and_site
    try:
        z = site.z_coordinate()
        assert isinstance(z, int)
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_site_is_zone(device_and_site: tuple[Device, Site]) -> None:
    """Test that the site is_zone is a boolean."""
    _device, site = device_and_site
    try:
        iz = site.is_zone()
        assert isinstance(iz, bool)
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_site_x_extent(device_and_site: tuple[Device, Site]) -> None:
    """Test that the site x extent is a positive integer."""
    _device, site = device_and_site
    try:
        xe = site.x_extent()
        assert isinstance(xe, int)
        assert xe > 0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_site_y_extent(device_and_site: tuple[Device, Site]) -> None:
    """Test that the site y extent is a positive integer."""
    _device, site = device_and_site
    try:
        ye = site.y_extent()
        assert isinstance(ye, int)
        assert ye > 0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_site_z_extent(device_and_site: tuple[Device, Site]) -> None:
    """Test that the site z extent is a positive integer."""
    _device, site = device_and_site
    try:
        ze = site.z_extent()
        assert isinstance(ze, int)
        assert ze > 0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_site_module_index(device_and_site: tuple[Device, Site]) -> None:
    """Test that the site module index is a non-negative integer."""
    _device, site = device_and_site
    try:
        mi = site.module_index()
        assert isinstance(mi, int)
        assert mi >= 0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_site_submodule_index(device_and_site: tuple[Device, Site]) -> None:
    """Test that the site submodule index is a non-negative integer."""
    _device, site = device_and_site
    try:
        smi = site.submodule_index()
        assert isinstance(smi, int)
        assert smi >= 0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_operation_name(device_and_operation: tuple[Device, Operation]) -> None:
    """Test that the operation name is a non-empty string."""
    _device, operation = device_and_operation
    name = operation.name()
    assert isinstance(name, str)
    assert len(name) > 0


def test_operation_qubits_num(device_and_operation: tuple[Device, Operation]) -> None:
    """Test that the operation qubits number is a positive integer."""
    _device, operation = device_and_operation
    try:
        qn = operation.qubits_num()
        assert isinstance(qn, int)
        assert qn > 0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_operation_parameters_num(device_and_operation: tuple[Device, Operation]) -> None:
    """Test that the operation parameters number is a non-negative integer."""
    _device, operation = device_and_operation
    try:
        on = operation.parameters_num()
        assert isinstance(on, int)
        assert on >= 0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_operation_duration(device_and_operation: tuple[Device, Operation]) -> None:
    """Test that the operation duration is a positive integer."""
    _device, operation = device_and_operation
    try:
        dur = operation.duration()
        assert isinstance(dur, int)
        assert dur > 0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_operation_fidelity(device_and_operation: tuple[Device, Operation]) -> None:
    """Test that the operation fidelity is a float between 0 and 1."""
    _device, operation = device_and_operation
    try:
        fid = operation.fidelity()
        assert isinstance(fid, float)
        assert 0.0 <= fid <= 1.0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_operation_interaction_radius(device_and_operation: tuple[Device, Operation]) -> None:
    """Test that the operation interaction radius is a non-negative integer."""
    _device, operation = device_and_operation
    try:
        ir = operation.interaction_radius()
        assert isinstance(ir, int)
        assert ir >= 0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_operation_blocking_radius(device_and_operation: tuple[Device, Operation]) -> None:
    """Test that the operation blocking radius is a non-negative integer."""
    _device, operation = device_and_operation
    try:
        br = operation.blocking_radius()
        assert isinstance(br, int)
        assert br >= 0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_operation_idling_fidelity(device_and_operation: tuple[Device, Operation]) -> None:
    """Test that the operation idling fidelity is a float between 0 and 1."""
    _device, operation = device_and_operation
    try:
        idf = operation.idling_fidelity()
        assert isinstance(idf, float)
        assert 0.0 <= idf <= 1.0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_operation_is_zoned(device_and_operation: tuple[Device, Operation]) -> None:
    """Test that the operation is_zoned is a boolean."""
    _device, operation = device_and_operation
    try:
        iz = operation.is_zoned()
        assert isinstance(iz, bool)
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_operation_sites(device_and_operation: tuple[Device, Operation]) -> None:
    """Test that the operation sites is a non-empty list of Site objects."""
    device, operation = device_and_operation
    try:
        sites = operation.sites()
        assert isinstance(sites, list)
        assert len(sites) > 0
        assert all(isinstance(site, Site) for site in sites)
        device_sites = device.sites()
        assert all(site in device_sites for site in sites)
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))


def test_operation_mean_shuttling_speed(device_and_operation: tuple[Device, Operation]) -> None:
    """Test that the operation mean shuttling speed is a positive integer."""
    _device, operation = device_and_operation
    try:
        mss = operation.mean_shuttling_speed()
        assert isinstance(mss, int)
        assert mss > 0
    except RuntimeError as e:
        assert match(r".*Not supported.*", str(e))
