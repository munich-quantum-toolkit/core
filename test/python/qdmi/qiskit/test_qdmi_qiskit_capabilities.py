# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for capability extraction & caching.

These tests exercise the pure-Python capabilities module without requiring
Qiskit. They rely on the built-in NA device exposed via FoMaC bindings.
"""

from __future__ import annotations

import pytest

from mqt.core import fomac
from mqt.core.qdmi.qiskit.capabilities import (
    DeviceCapabilities,
    DeviceOperationInfo,
    DeviceSiteInfo,
    extract_capabilities,
)


@pytest.fixture
def device() -> fomac.Device:
    """Fixture providing a single FoMaC device for testing.

    Returns:
        First available FoMaC device.

    Note:
        Skips the test if no devices are available in the test environment.
    """
    devices = list(fomac.devices())
    if not devices:
        pytest.skip("No FoMaC devices available in test environment")
    return devices[0]


@pytest.fixture
def capabilities(device: fomac.Device) -> DeviceCapabilities:
    """Fixture providing extracted device capabilities.

    Args:
        device: FoMaC device fixture.

    Returns:
        Extracted device capabilities.
    """
    return extract_capabilities(device)


def test_extract_capabilities_basic(capabilities: DeviceCapabilities) -> None:
    """Extract snapshot and verify core structural and hash invariants."""
    assert isinstance(capabilities, DeviceCapabilities)
    assert capabilities.device_name
    assert capabilities.num_qubits >= 0
    assert capabilities.capabilities_hash is not None
    assert len(capabilities.capabilities_hash) == 64
    if capabilities.sites:
        assert isinstance(capabilities.sites[0], DeviceSiteInfo)
    if capabilities.operations:
        first_key = next(iter(capabilities.operations))
        assert isinstance(capabilities.operations[first_key], DeviceOperationInfo)


def test_device_capabilities_has_device_name(capabilities: DeviceCapabilities) -> None:
    """DeviceCapabilities should contain device name."""
    assert capabilities.device_name
    assert isinstance(capabilities.device_name, str)


def test_device_capabilities_has_version(capabilities: DeviceCapabilities) -> None:
    """DeviceCapabilities should contain device version."""
    assert capabilities.device_version
    assert isinstance(capabilities.device_version, str)


def test_device_capabilities_has_library_version(capabilities: DeviceCapabilities) -> None:
    """DeviceCapabilities should contain library version."""
    assert capabilities.library_version
    assert isinstance(capabilities.library_version, str)


def test_device_capabilities_operations_non_empty(capabilities: DeviceCapabilities) -> None:
    """DeviceCapabilities should contain at least one operation."""
    assert len(capabilities.operations) > 0


def test_device_operation_info_has_name(capabilities: DeviceCapabilities) -> None:
    """DeviceOperationInfo should have a name."""
    for op_name, op_info in capabilities.operations.items():
        assert op_info.name == op_name
        assert isinstance(op_info.name, str)


def test_device_capabilities_signature_is_stable(device: fomac.Device) -> None:
    """Device signature should be stable across multiple extractions."""
    caps1 = extract_capabilities(device)
    caps2 = extract_capabilities(device)
    assert caps1.signature == caps2.signature


def test_device_capabilities_to_canonical_json(capabilities: DeviceCapabilities) -> None:
    """DeviceCapabilities should produce valid canonical JSON."""
    json_str = capabilities.to_canonical_json()

    assert isinstance(json_str, str)
    assert len(json_str) > 0
    # Should be valid JSON
    import json

    parsed = json.loads(json_str)
    assert "device_name" in parsed
    assert "num_qubits" in parsed
    assert "operations" in parsed


def test_device_capabilities_canonical_json_excludes_hash(capabilities: DeviceCapabilities) -> None:
    """Canonical JSON should not include the capabilities_hash itself."""
    json_str = capabilities.to_canonical_json()

    # Parse and check
    import json

    parsed = json.loads(json_str)
    assert "capabilities_hash" not in parsed


def test_device_capabilities_coupling_map(capabilities: DeviceCapabilities) -> None:
    """DeviceCapabilities should include coupling map if available."""
    # Coupling map may be None or a list
    if capabilities.coupling_map is not None:
        assert isinstance(capabilities.coupling_map, list)
        if capabilities.coupling_map:
            # Should contain tuples of ints
            assert isinstance(capabilities.coupling_map[0], tuple)
            assert len(capabilities.coupling_map[0]) == 2


def test_device_site_info_has_index(capabilities: DeviceCapabilities) -> None:
    """DeviceSiteInfo should have an index."""
    if capabilities.sites:
        for site in capabilities.sites:
            assert isinstance(site.index, int)
            assert site.index >= 0


def test_device_operation_info_parameters_num(capabilities: DeviceCapabilities) -> None:
    """DeviceOperationInfo should have parameters_num field."""
    for op_info in capabilities.operations.values():
        assert isinstance(op_info.parameters_num, int)
        assert op_info.parameters_num >= 0


def test_device_capabilities_sites_sorted_by_index(capabilities: DeviceCapabilities) -> None:
    """Device sites should be sorted by index."""
    if len(capabilities.sites) > 1:
        indices = [site.index for site in capabilities.sites]
        assert indices == sorted(indices)


def test_device_capabilities_status(capabilities: DeviceCapabilities) -> None:
    """DeviceCapabilities should include status."""
    assert capabilities.status is not None
    assert isinstance(capabilities.status, str)


def test_extract_capabilities_is_fresh(device: fomac.Device) -> None:
    """extract_capabilities should always create a new object.

    Note:
        While device objects returned by different calls to fomac.devices()
        compare positively with == (same device, different object instances),
        extracted capabilities should always create fresh DeviceCapabilities objects.
    """
    caps1 = extract_capabilities(device)
    caps2 = extract_capabilities(device)
    assert caps1 is not caps2  # Different objects
    assert caps1.signature == caps2.signature  # Same content


def test_device_capabilities_hash_is_sha256(capabilities: DeviceCapabilities) -> None:
    """Capabilities hash should be a valid SHA256 hex string."""
    assert capabilities.capabilities_hash is not None
    assert len(capabilities.capabilities_hash) == 64  # SHA256 hex length
    # Should be valid hex
    int(capabilities.capabilities_hash, 16)  # Raises if not valid hex


def test_device_capabilities_hash_deterministic(device: fomac.Device) -> None:
    """Capabilities hash should be deterministic for same device state."""
    caps1 = extract_capabilities(device)
    caps2 = extract_capabilities(device)
    assert caps1.capabilities_hash == caps2.capabilities_hash


def test_device_operation_info_sites_is_tuple_or_none(capabilities: DeviceCapabilities) -> None:
    """DeviceOperationInfo.sites should be a tuple or None."""
    for op_info in capabilities.operations.values():
        if op_info.sites is not None:
            assert isinstance(op_info.sites, tuple)


def test_device_capabilities_duration_units(capabilities: DeviceCapabilities) -> None:
    """DeviceCapabilities should have duration unit and scale factor."""
    # These may be None, but should exist
    assert hasattr(capabilities, "duration_unit")
    assert hasattr(capabilities, "duration_scale_factor")


def test_device_capabilities_length_units(capabilities: DeviceCapabilities) -> None:
    """DeviceCapabilities should have length unit and scale factor."""
    # These may be None, but should exist
    assert hasattr(capabilities, "length_unit")
    assert hasattr(capabilities, "length_scale_factor")


def test_device_capabilities_min_atom_distance(capabilities: DeviceCapabilities) -> None:
    """DeviceCapabilities should have min_atom_distance field."""
    assert hasattr(capabilities, "min_atom_distance")
