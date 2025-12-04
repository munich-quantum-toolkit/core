# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the quantum computation IR."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

import pytest

from mqt.core.fomac import Device, Job, ProgramFormat, Session


def _get_devices() -> list[Device]:
    """Get all available devices from a Session.

    Returns:
        List of all available QDMI devices.
    """
    session = Session()
    return session.get_devices()


@pytest.fixture(params=_get_devices())
def device(request: pytest.FixtureRequest) -> Device:
    """Fixture to provide a device for testing.

    Returns:
       A quantum device instance.
    """
    return cast("Device", request.param)


@pytest.fixture(params=_get_devices())
def device_and_site(request: pytest.FixtureRequest) -> tuple[Device, Device.Site]:
    """Fixture to provide a device for testing.

    Returns:
       A tuple containing a quantum device instance and one of its sites.
    """
    dev = request.param
    site = dev.sites()[0]
    return dev, site


@pytest.fixture(params=_get_devices())
def device_and_operation(request: pytest.FixtureRequest) -> tuple[Device, Device.Operation]:
    """Fixture to provide a device for testing.

    Returns:
       A tuple containing a quantum device instance and one of its operations.
    """
    dev = request.param
    operation = dev.operations()[0]
    return dev, operation


@pytest.fixture
def ddsim_device() -> Device:
    """Fixture to provide the DDSIM device for job submission testing.

    Returns:
        The MQT Core DDSIM QDMI Device if it can be found.
    """
    for dev in _get_devices():
        if dev.name() == "MQT Core DDSIM QDMI Device":
            return dev
    pytest.skip("DDSIM device not found - job submission tests require DDSIM device")


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
    """Test that the operation qubits number is a non-negative integer."""
    _device, operation = device_and_operation
    qn = operation.qubits_num()
    if qn is not None:
        assert isinstance(qn, int)
        assert qn >= 0


def test_operation_parameters_num(device_and_operation: tuple[Device, Device.Operation]) -> None:
    """Test that the operation parameters number is a non-negative integer."""
    _device, operation = device_and_operation
    on = operation.parameters_num()
    if on is not None:
        assert isinstance(on, int)
        assert on >= 0


def test_operation_duration(device_and_operation: tuple[Device, Device.Operation]) -> None:
    """Test that the operation duration is a non-negative integer."""
    _device, operation = device_and_operation
    dur = operation.duration()
    if dur is not None:
        assert isinstance(dur, int)
        assert dur >= 0


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


def test_device_submit_job_returns_valid_job(ddsim_device: Device) -> None:
    """Test that submit_job creates a Job object with valid properties."""
    qasm3_program = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""

    job = ddsim_device.submit_job(qasm3_program, ProgramFormat.QASM3, num_shots=100)

    # Job should have a non-empty ID
    assert len(job.id) > 0
    # The program format should be preserved
    assert job.program_format == ProgramFormat.QASM3
    # The program should be preserved
    assert job.program == qasm3_program
    # Num shots should match request
    assert job.num_shots == 100


def test_device_submit_job_preserves_num_shots(ddsim_device: Device) -> None:
    """Test that different shot counts are correctly preserved."""
    qasm3_program = """
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
c[0] = measure q[0];
"""

    # Submit jobs with different shot counts
    job1 = ddsim_device.submit_job(qasm3_program, ProgramFormat.QASM3, num_shots=10)
    job2 = ddsim_device.submit_job(qasm3_program, ProgramFormat.QASM3, num_shots=100)
    job3 = ddsim_device.submit_job(qasm3_program, ProgramFormat.QASM3, num_shots=1000)

    assert job1.num_shots == 10
    assert job2.num_shots == 100
    assert job3.num_shots == 1000


@pytest.fixture
def submitted_job(ddsim_device: Device) -> Job:
    """Fixture that provides a submitted job for testing.

    Returns:
        A submitted job with 10 shots.
    """
    qasm3_program = """
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
c[0] = measure q[0];
"""
    return ddsim_device.submit_job(qasm3_program, ProgramFormat.QASM3, num_shots=10)


def test_job_ids_are_unique(ddsim_device: Device) -> None:
    """Test that different jobs have unique IDs."""
    qasm3_program = """
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
c[0] = measure q[0];
"""

    job1 = ddsim_device.submit_job(qasm3_program, ProgramFormat.QASM3, num_shots=10)
    job2 = ddsim_device.submit_job(qasm3_program, ProgramFormat.QASM3, num_shots=10)

    assert job1.id != job2.id


def test_job_status_progresses(submitted_job: Job) -> None:
    """Test that job status progresses to completion."""
    initial_status = submitted_job.check()
    assert isinstance(initial_status, Job.Status)

    # Wait for completion
    submitted_job.wait()

    # After waiting, status should be DONE or FAILED
    final_status = submitted_job.check()
    assert final_status in {Job.Status.DONE, Job.Status.FAILED}


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


@pytest.fixture
def simulator_job(ddsim_device: Device) -> Job:
    """Fixture that provides a simulator job for testing.

    Returns:
        A submitted job with 0 shots.
    """
    qasm3_program = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cx q[0], q[1];
"""
    return ddsim_device.submit_job(qasm3_program, ProgramFormat.QASM3, num_shots=0)


def test_simulator_job_get_dense_state_vector_returns_valid_state(simulator_job: Job) -> None:
    """Test that get_dense_statevector() returns the correct Bell state."""
    simulator_job.wait()

    state_vector = simulator_job.get_dense_statevector()
    assert len(state_vector) == 4  # 2 qubits -> 4 amplitudes

    # The expected state is (|00> + |11>)/sqrt(2)
    inv_sqrt2 = 1.0 / (2**0.5)
    assert abs(state_vector[0]) == pytest.approx(inv_sqrt2)  # |00>
    assert abs(state_vector[1]) == pytest.approx(0.0)  # |01>
    assert abs(state_vector[2]) == pytest.approx(0.0)  # |10>
    assert abs(state_vector[3]) == pytest.approx(inv_sqrt2)  # |11>


def test_simulator_job_get_dense_probabilities_returns_valid_probabilities(simulator_job: Job) -> None:
    """Test that get_dense_probabilities() returns the correct probabilities."""
    simulator_job.wait()

    probabilities = simulator_job.get_dense_probabilities()
    assert len(probabilities) == 4  # 2 qubits -> 4 probabilities

    # The expected probabilities are 0.5 for |00> and |11>, and 0 for |01> and |10>
    assert probabilities[0] == pytest.approx(0.5)  # |00>
    assert probabilities[1] == pytest.approx(0.0)  # |01>
    assert probabilities[2] == pytest.approx(0.0)  # |10>
    assert probabilities[3] == pytest.approx(0.5)  # |11>


def test_simulator_job_get_sparse_state_vector_returns_valid_state(simulator_job: Job) -> None:
    """Test that get_sparse_statevector() returns the correct Bell state."""
    simulator_job.wait()

    sparse_state_vector = simulator_job.get_sparse_statevector()
    assert len(sparse_state_vector) == 2  # Only |00> and |11> should be present

    inv_sqrt2 = 1.0 / (2**0.5)
    assert "00" in sparse_state_vector
    assert abs(sparse_state_vector["00"]) == pytest.approx(inv_sqrt2)

    assert "11" in sparse_state_vector
    assert abs(sparse_state_vector["11"]) == pytest.approx(inv_sqrt2)


def test_simulator_job_get_sparse_probabilities_returns_valid_probabilities(simulator_job: Job) -> None:
    """Test that get_sparse_probabilities() returns the correct probabilities."""
    simulator_job.wait()

    sparse_probabilities = simulator_job.get_sparse_probabilities()
    assert len(sparse_probabilities) == 2  # Only |00> and |11> should be present

    assert "00" in sparse_probabilities
    assert sparse_probabilities["00"] == pytest.approx(0.5)

    assert "11" in sparse_probabilities
    assert sparse_probabilities["11"] == pytest.approx(0.5)


def test_session_construction_with_token() -> None:
    """Test Session construction with a token parameter.

    Note: The underlying QDMI library may not support authentication parameters yet,
    so this test verifies that the parameter can be passed without causing errors
    during construction, even if it's not actually used.
    """
    # Empty token should be accepted
    try:
        session = Session(token="")
        assert session is not None
    except RuntimeError:
        # If not supported, that's okay for now
        pass

    # Non-empty token should be accepted
    try:
        session = Session(token="test_token_123")  # noqa: S106
        assert session is not None
    except RuntimeError:
        # If not supported, that's okay for now
        pass

    # Token with special characters should be accepted
    try:
        session = Session(token="very_long_token_with_special_characters_!@#$%^&*()")  # noqa: S106
        assert session is not None
    except RuntimeError:
        # If not supported, that's okay for now
        pass


def test_session_construction_with_auth_url() -> None:
    """Test Session construction with auth URL parameter.

    Note: The currently available QDMI devices don't support authentication.
    Valid URLs should either be accepted or rejected with "Not supported" error.
    Invalid URLs should be rejected with validation errors.
    """
    # Valid HTTPS URL
    try:
        session = Session(auth_url="https://example.com")
        assert session is not None
    except RuntimeError:
        # Either not supported or validation failed - both acceptable
        pass
    # Valid HTTP URL with port and path
    try:
        session = Session(auth_url="http://auth.server.com:8080/api")
        assert session is not None
    except RuntimeError:
        # Either not supported or validation failed - both acceptable
        pass
    # Valid HTTPS URL with query parameters
    try:
        session = Session(auth_url="https://auth.example.com/token?param=value")
        assert session is not None
    except RuntimeError:
        # Either not supported or validation failed - both acceptable
        pass
    # Invalid URL - not a URL at all
    with pytest.raises(RuntimeError):
        Session(auth_url="not-a-url")
    # Invalid URL - unsupported protocol
    with pytest.raises(RuntimeError):
        Session(auth_url="ftp://invalid.com")

    # Invalid URL - missing protocol
    with pytest.raises(RuntimeError):
        Session(auth_url="example.com")


def test_session_construction_with_auth_file() -> None:
    """Test Session construction with auth file parameter.

    Note: The currently available QDMI devices don't support authentication.
    Existing files should either be accepted or rejected with "Not supported" error.
    Non-existent files should be rejected with validation errors.
    """
    # Test with non-existent file - should raise error
    with pytest.raises(RuntimeError):
        Session(auth_file="/nonexistent/path/to/file.txt")

    # Test with another non-existent file
    with pytest.raises(RuntimeError):
        Session(auth_file="/tmp/this_file_does_not_exist_12345.txt")  # noqa: S108

    # Test with existing file
    with tempfile.NamedTemporaryFile(encoding="utf-8", mode="w", delete=False, suffix=".txt") as tmp_file:
        tmp_file.write("test_token_content")
        tmp_path = tmp_file.name

    try:
        # Existing file should be accepted or rejected with "Not supported"
        try:
            session = Session(auth_file=tmp_path)
            assert session is not None
        except RuntimeError:
            # If not supported, that's okay for now
            pass
    finally:
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)


def test_session_construction_with_username_password() -> None:
    """Test Session construction with username and password parameters.

    Note: The currently available QDMI devices don't support authentication.
    """
    # Username only
    try:
        session = Session(username="user123")
        assert session is not None
    except RuntimeError:
        # If not supported, that's okay for now
        pass

    # Password only
    try:
        session = Session(password="secure_password")  # noqa: S106
        assert session is not None
    except RuntimeError:
        # If not supported, that's okay for now
        pass

    # Both username and password
    try:
        session = Session(username="user123", password="secure_password")  # noqa: S106
        assert session is not None
    except RuntimeError:
        # If not supported, that's okay for now
        pass


def test_session_construction_with_project_id() -> None:
    """Test Session construction with project ID parameter.

    Note: The currently available QDMI devices don't support authentication.
    """
    try:
        session = Session(project_id="project-123-abc")
        assert session is not None
    except RuntimeError:
        # If not supported, that's okay for now
        pass


def test_session_construction_with_multiple_parameters() -> None:
    """Test Session construction with multiple authentication parameters.

    Note: The currently available QDMI devices don't support authentication.
    """
    try:
        session = Session(
            token="test_token",  # noqa: S106
            username="test_user",
            password="test_pass",  # noqa: S106
            project_id="test_project",
        )
        assert session is not None
    except RuntimeError:
        # If not supported, that's okay for now
        pass


def test_session_get_devices_returns_list() -> None:
    """Test that get_devices() returns a list of Device objects."""
    session = Session()
    devices = session.get_devices()

    assert isinstance(devices, list)
    assert len(devices) > 0

    # All elements should be Device instances
    for device in devices:
        assert isinstance(device, Device)
        # Device should have a name
        assert len(device.name()) > 0


def test_session_multiple_instances() -> None:
    """Test that multiple Session instances can be created independently."""
    session1 = Session()
    session2 = Session()

    devices1 = session1.get_devices()
    devices2 = session2.get_devices()

    # Both should return devices
    assert len(devices1) > 0
    assert len(devices2) > 0

    # Should return the same number of devices
    assert len(devices1) == len(devices2)
