# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the quantum computation IR."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from mqt.core.fomac import Device, devices

if TYPE_CHECKING:
    from mqt.core.fomac import Job


@pytest.fixture(params=devices())
def device(request: pytest.FixtureRequest) -> Device:
    """Fixture to provide a device for testing.

    Returns:
        Device: A quantum device instance.
    """
    return cast("Device", request.param)


@pytest.fixture(params=devices())
def device_and_site(request: pytest.FixtureRequest) -> tuple[Device, Device.Site]:
    """Fixture to provide a device for testing.

    Returns:
        tuple[Device, Device.Site]: A tuple containing a quantum device instance and one of its sites.
    """
    device = request.param
    site = device.sites()[0]
    return device, site


@pytest.fixture(params=devices())
def device_and_operation(request: pytest.FixtureRequest) -> tuple[Device, Device.Operation]:
    """Fixture to provide a device for testing.

    Returns:
        tuple[Device, Device.Operation]: A tuple containing a quantum device instance and one of its operations.
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
    """Test that the device status is a valid Device.Status enum member."""
    status = device.status()
    assert isinstance(status, Device.Status)


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
    """Test that the device sites is a non-empty list of Device.Site objects."""
    sites = device.sites()
    assert isinstance(sites, list)
    assert len(sites) > 0
    assert all(isinstance(site, Device.Site) for site in sites)


def test_device_operations(device: Device) -> None:
    """Test that the device operations is a non-empty list of Device.Operation objects."""
    operations = device.operations()
    assert isinstance(operations, list)
    assert len(operations) > 0
    assert all(isinstance(op, Device.Operation) for op in operations)


def test_device_coupling_map(device: Device) -> None:
    """Test that the device coupling map is a list of tuples of Device.Site objects."""
    cm = device.coupling_map()
    if cm is not None:
        assert isinstance(cm, list)
        assert all(len(pair) == 2 for pair in cm)
        assert all(isinstance(site, Device.Site) for pair in cm for site in pair)


def test_device_needs_calibration(device: Device) -> None:
    """Test that the device needs calibration is an integer."""
    needs_cal = device.needs_calibration()
    if needs_cal is not None:
        assert isinstance(needs_cal, int)


def test_device_length_unit(device: Device) -> None:
    """Test that the device length unit is a non-empty string."""
    lu = device.length_unit()
    if lu is not None:
        assert isinstance(lu, str)
        assert len(lu) > 0


def test_device_length_scale_factor(device: Device) -> None:
    """Test that the device length scale factor is a positive float."""
    lsf = device.length_scale_factor()
    if lsf is not None:
        assert isinstance(lsf, float)
        assert lsf > 0.0


def test_device_duration_unit(device: Device) -> None:
    """Test that the device duration unit is a non-empty string."""
    du = device.duration_unit()
    if du is not None:
        assert isinstance(du, str)
        assert len(du) > 0


def test_device_duration_scale_factor(device: Device) -> None:
    """Test that the device duration scale factor is a positive float."""
    dsf = device.duration_scale_factor()
    if dsf is not None:
        assert isinstance(dsf, float)
        assert dsf > 0.0


def test_device_min_atom_distance(device: Device) -> None:
    """Test that the device minimum atom distance is a positive float."""
    mad = device.min_atom_distance()
    if mad is not None:
        assert isinstance(mad, int)
        assert mad > 0.0


def test_site_index(device_and_site: tuple[Device, Device.Site]) -> None:
    """Test that the site index is a non-negative integer."""
    _device, site = device_and_site
    index = site.index()
    assert isinstance(index, int)
    assert index >= 0


def test_site_t1(device_and_site: tuple[Device, Device.Site]) -> None:
    """Test that the site T1 coherence time is a positive integer."""
    _device, site = device_and_site
    t1 = site.t1()
    if t1 is not None:
        assert isinstance(t1, int)
        assert t1 > 0


def test_site_t2(device_and_site: tuple[Device, Device.Site]) -> None:
    """Test that the site T2 coherence time is a positive integer."""
    _device, site = device_and_site
    t2 = site.t2()
    if t2 is not None:
        assert isinstance(t2, int)
        assert t2 > 0


def test_site_name(device_and_site: tuple[Device, Device.Site]) -> None:
    """Test that the site name is a non-empty string."""
    _device, site = device_and_site
    name = site.name()
    if name is not None:
        assert isinstance(name, str)
        assert len(name) > 0


def test_site_x_coordinate(device_and_site: tuple[Device, Device.Site]) -> None:
    """Test that the site x coordinate is an integer."""
    _device, site = device_and_site
    x = site.x_coordinate()
    if x is not None:
        assert isinstance(x, int)


def test_site_y_coordinate(device_and_site: tuple[Device, Device.Site]) -> None:
    """Test that the site y coordinate is an integer."""
    _device, site = device_and_site
    y = site.y_coordinate()
    if y is not None:
        assert isinstance(y, int)


def test_site_z_coordinate(device_and_site: tuple[Device, Device.Site]) -> None:
    """Test that the site z coordinate is an integer."""
    _device, site = device_and_site
    z = site.z_coordinate()
    if z is not None:
        assert isinstance(z, int)


def test_site_is_zone(device_and_site: tuple[Device, Device.Site]) -> None:
    """Test that the site is_zone is a boolean."""
    _device, site = device_and_site
    iz = site.is_zone()
    if iz is not None:
        assert isinstance(iz, bool)


def test_site_x_extent(device_and_site: tuple[Device, Device.Site]) -> None:
    """Test that the site x extent is a positive integer."""
    _device, site = device_and_site
    xe = site.x_extent()
    if xe is not None:
        assert isinstance(xe, int)
        assert xe > 0


def test_site_y_extent(device_and_site: tuple[Device, Device.Site]) -> None:
    """Test that the site y extent is a positive integer."""
    _device, site = device_and_site
    ye = site.y_extent()
    if ye is not None:
        assert isinstance(ye, int)
        assert ye > 0


def test_site_z_extent(device_and_site: tuple[Device, Device.Site]) -> None:
    """Test that the site z extent is a positive integer."""
    _device, site = device_and_site
    ze = site.z_extent()
    if ze is not None:
        assert isinstance(ze, int)
        assert ze > 0


def test_site_module_index(device_and_site: tuple[Device, Device.Site]) -> None:
    """Test that the site module index is a non-negative integer."""
    _device, site = device_and_site
    mi = site.module_index()
    if mi is not None:
        assert isinstance(mi, int)
        assert mi >= 0


def test_site_submodule_index(device_and_site: tuple[Device, Device.Site]) -> None:
    """Test that the site submodule index is a non-negative integer."""
    _device, site = device_and_site
    smi = site.submodule_index()
    if smi is not None:
        assert isinstance(smi, int)
        assert smi >= 0


def test_operation_name(device_and_operation: tuple[Device, Device.Operation]) -> None:
    """Test that the operation name is a non-empty string."""
    _device, operation = device_and_operation
    name = operation.name()
    assert isinstance(name, str)
    assert len(name) > 0


def test_operation_qubits_num(device_and_operation: tuple[Device, Device.Operation]) -> None:
    """Test that the operation qubits number is a positive integer."""
    _device, operation = device_and_operation
    qn = operation.qubits_num()
    if qn is not None:
        assert isinstance(qn, int)
        assert qn > 0


def test_operation_parameters_num(device_and_operation: tuple[Device, Device.Operation]) -> None:
    """Test that the operation parameters number is a non-negative integer."""
    _device, operation = device_and_operation
    on = operation.parameters_num()
    if on is not None:
        assert isinstance(on, int)
        assert on >= 0


def test_operation_duration(device_and_operation: tuple[Device, Device.Operation]) -> None:
    """Test that the operation duration is a positive integer."""
    _device, operation = device_and_operation
    dur = operation.duration()
    if dur is not None:
        assert isinstance(dur, int)
        assert dur > 0


def test_operation_fidelity(device_and_operation: tuple[Device, Device.Operation]) -> None:
    """Test that the operation fidelity is a float between 0 and 1."""
    _device, operation = device_and_operation
    fid = operation.fidelity()
    if fid is not None:
        assert isinstance(fid, float)
        assert 0.0 <= fid <= 1.0


def test_operation_interaction_radius(device_and_operation: tuple[Device, Device.Operation]) -> None:
    """Test that the operation interaction radius is a non-negative integer."""
    _device, operation = device_and_operation
    ir = operation.interaction_radius()
    if ir is not None:
        assert isinstance(ir, int)
        assert ir >= 0


def test_operation_blocking_radius(device_and_operation: tuple[Device, Device.Operation]) -> None:
    """Test that the operation blocking radius is a non-negative integer."""
    _device, operation = device_and_operation
    br = operation.blocking_radius()
    if br is not None:
        assert isinstance(br, int)
        assert br >= 0


def test_operation_idling_fidelity(device_and_operation: tuple[Device, Device.Operation]) -> None:
    """Test that the operation idling fidelity is a float between 0 and 1."""
    _device, operation = device_and_operation
    idf = operation.idling_fidelity()
    if idf is not None:
        assert isinstance(idf, float)
        assert 0.0 <= idf <= 1.0


def test_operation_is_zoned(device_and_operation: tuple[Device, Device.Operation]) -> None:
    """Test that the operation is_zoned is a boolean."""
    _device, operation = device_and_operation
    iz = operation.is_zoned()
    if iz is not None:
        assert isinstance(iz, bool)


def test_operation_sites(device_and_operation: tuple[Device, Device.Operation]) -> None:
    """Test that the operation sites is a non-empty list of Device.Site objects."""
    device, operation = device_and_operation
    sites = operation.sites()
    if sites is not None:
        assert isinstance(sites, list)
        assert len(sites) > 0
        assert all(isinstance(site, Device.Site) for site in sites)
        device_sites = device.sites()
        assert all(site in device_sites for site in sites)


def test_operation_mean_shuttling_speed(device_and_operation: tuple[Device, Device.Operation]) -> None:
    """Test that the operation mean shuttling speed is a positive integer."""
    _device, operation = device_and_operation
    mss = operation.mean_shuttling_speed()
    if mss is not None:
        assert isinstance(mss, int)
        assert mss > 0


def test_device_submit_job_returns_valid_job(device: Device) -> None:
    """Test that submit_job creates a Job object with valid properties."""
    from mqt.core.fomac import Job, ProgramFormat

    qasm3_program = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""

    job = device.submit_job(qasm3_program, ProgramFormat.QASM3, num_shots=100)
    assert isinstance(job, Job)

    # Job should have a non-empty ID
    assert len(job.id) > 0

    # Num shots should match request
    assert job.num_shots == 100


def test_device_submit_job_preserves_num_shots(device: Device) -> None:
    """Test that different shot counts are correctly preserved."""
    from mqt.core.fomac import ProgramFormat

    qasm3_program = """
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
c[0] = measure q[0];
"""

    # Submit jobs with different shot counts
    job1 = device.submit_job(qasm3_program, ProgramFormat.QASM3, num_shots=10)
    job2 = device.submit_job(qasm3_program, ProgramFormat.QASM3, num_shots=100)
    job3 = device.submit_job(qasm3_program, ProgramFormat.QASM3, num_shots=1000)

    assert job1.num_shots == 10
    assert job2.num_shots == 100
    assert job3.num_shots == 1000


@pytest.fixture
def submitted_job(device: Device) -> Job:
    """Fixture that provides a submitted job for testing.

    Returns:
        A submitted job with 10 shots.
    """
    from mqt.core.fomac import ProgramFormat

    qasm3_program = """
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
c[0] = measure q[0];
"""
    return device.submit_job(qasm3_program, ProgramFormat.QASM3, num_shots=10)


def test_job_ids_are_unique(device: Device) -> None:
    """Test that different jobs have unique IDs."""
    from mqt.core.fomac import ProgramFormat

    qasm3_program = """
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
c[0] = measure q[0];
"""

    job1 = device.submit_job(qasm3_program, ProgramFormat.QASM3, num_shots=10)
    job2 = device.submit_job(qasm3_program, ProgramFormat.QASM3, num_shots=10)

    assert job1.id != job2.id


def test_job_status_progresses(submitted_job: Job) -> None:
    """Test that job status progresses to completion."""
    from mqt.core.fomac import JobStatus

    initial_status = submitted_job.check()
    assert isinstance(initial_status, JobStatus)

    # Wait for completion
    submitted_job.wait()

    # After waiting, status should be DONE or FAILED
    final_status = submitted_job.check()
    assert final_status in {JobStatus.DONE, JobStatus.FAILED}


def test_job_get_counts_returns_valid_histogram(submitted_job: Job) -> None:
    """Test that job get_counts() returns valid measurement results."""
    # Wait for job to complete
    submitted_job.wait()

    # Get counts
    counts = submitted_job.get_counts()
    assert isinstance(counts, dict)
    assert len(counts) > 0

    # For a single qubit, all keys should be "0" or "1"
    for key in counts:
        assert isinstance(key, str)
        assert len(key) == 1
        assert key in {"0", "1"}

    # All values should be positive integers
    for value in counts.values():
        assert isinstance(value, int)
        assert value > 0

    # Verify total counts match num_shots
    total_counts = sum(counts.values())
    assert total_counts == submitted_job.num_shots


def test_job_get_counts_is_consistent(submitted_job: Job) -> None:
    """Test that multiple get_counts() calls return consistent results."""
    # Wait for job to complete
    submitted_job.wait()

    # Get counts multiple times
    counts1 = submitted_job.get_counts()
    counts2 = submitted_job.get_counts()

    # Results should be identical
    assert counts1 == counts2
