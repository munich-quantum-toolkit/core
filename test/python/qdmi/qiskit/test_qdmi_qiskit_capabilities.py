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

from mqt.core import fomac
from mqt.core.qdmi.qiskit.capabilities import (
    DeviceCapabilities,
    DeviceOperationInfo,
    DeviceSiteInfo,
    extract_capabilities,
    get_capabilities,
)


def _get_single_device() -> fomac.Device:
    devices = list(fomac.devices())
    # At least one device (NA) should be present in test environment
    assert devices, "Expected at least one FoMaC device"
    return devices[0]


def test_extract_capabilities_basic() -> None:
    """Extract snapshot and verify core structural and hash invariants."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    assert isinstance(caps, DeviceCapabilities)
    assert caps.device_name
    assert caps.num_qubits >= 0
    assert caps.capabilities_hash is not None
    assert len(caps.capabilities_hash) == 64
    if caps.sites:
        assert isinstance(caps.sites[0], DeviceSiteInfo)
    if caps.operations:
        first_key = next(iter(caps.operations))
        assert isinstance(caps.operations[first_key], DeviceOperationInfo)


def test_get_capabilities_cache_hit() -> None:
    """Ensure repeated cached lookups return identical object instance."""
    dev = _get_single_device()
    c1 = get_capabilities(dev, use_cache=True)
    c2 = get_capabilities(dev, use_cache=True)
    assert c1 is c2


def test_get_capabilities_force_refresh() -> None:
    """A forced refresh returns a new object but with identical signature/hash."""
    dev = _get_single_device()
    cached = get_capabilities(dev, use_cache=True)
    fresh = get_capabilities(dev, use_cache=False)
    assert fresh is not cached
    assert fresh.signature == cached.signature
    assert fresh.capabilities_hash == cached.capabilities_hash


def test_device_capabilities_has_device_name() -> None:
    """DeviceCapabilities should contain device name."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    assert caps.device_name
    assert isinstance(caps.device_name, str)


def test_device_capabilities_has_version() -> None:
    """DeviceCapabilities should contain device version."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    assert caps.device_version
    assert isinstance(caps.device_version, str)


def test_device_capabilities_has_library_version() -> None:
    """DeviceCapabilities should contain library version."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    assert caps.library_version
    assert isinstance(caps.library_version, str)


def test_device_capabilities_operations_non_empty() -> None:
    """DeviceCapabilities should contain at least one operation."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    assert len(caps.operations) > 0


def test_device_operation_info_has_name() -> None:
    """DeviceOperationInfo should have a name."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    for op_name, op_info in caps.operations.items():
        assert op_info.name == op_name
        assert isinstance(op_info.name, str)


def test_device_capabilities_signature_is_stable() -> None:
    """Device signature should be stable across multiple extractions."""
    dev = _get_single_device()
    caps1 = extract_capabilities(dev)
    caps2 = extract_capabilities(dev)
    assert caps1.signature == caps2.signature


def test_device_capabilities_to_canonical_json() -> None:
    """DeviceCapabilities should produce valid canonical JSON."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    json_str = caps.to_canonical_json()

    assert isinstance(json_str, str)
    assert len(json_str) > 0
    # Should be valid JSON
    import json

    parsed = json.loads(json_str)
    assert "device_name" in parsed
    assert "num_qubits" in parsed
    assert "operations" in parsed


def test_device_capabilities_canonical_json_excludes_hash() -> None:
    """Canonical JSON should not include the capabilities_hash itself."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    json_str = caps.to_canonical_json()

    # Parse and check
    import json

    parsed = json.loads(json_str)
    assert "capabilities_hash" not in parsed


def test_device_capabilities_coupling_map() -> None:
    """DeviceCapabilities should include coupling map if available."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    # Coupling map may be None or a list
    if caps.coupling_map is not None:
        assert isinstance(caps.coupling_map, list)
        if caps.coupling_map:
            # Should contain tuples of ints
            assert isinstance(caps.coupling_map[0], tuple)
            assert len(caps.coupling_map[0]) == 2


def test_device_site_info_has_index() -> None:
    """DeviceSiteInfo should have an index."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    if caps.sites:
        for site in caps.sites:
            assert isinstance(site.index, int)
            assert site.index >= 0


def test_device_operation_info_parameters_num() -> None:
    """DeviceOperationInfo should have parameters_num field."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    for op_info in caps.operations.values():
        assert isinstance(op_info.parameters_num, int)
        assert op_info.parameters_num >= 0


def test_device_capabilities_sites_sorted_by_index() -> None:
    """Device sites should be sorted by index."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    if len(caps.sites) > 1:
        indices = [site.index for site in caps.sites]
        assert indices == sorted(indices)


def test_device_capabilities_status() -> None:
    """DeviceCapabilities should include status."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    assert caps.status is not None
    assert isinstance(caps.status, str)


def test_extract_capabilities_is_fresh() -> None:
    """extract_capabilities should always create a new object."""
    dev = _get_single_device()
    caps1 = extract_capabilities(dev)
    caps2 = extract_capabilities(dev)
    assert caps1 is not caps2  # Different objects
    assert caps1.signature == caps2.signature  # Same content


def test_get_capabilities_default_uses_cache() -> None:
    """get_capabilities should use cache by default."""
    dev = _get_single_device()
    caps1 = get_capabilities(dev)  # Default is use_cache=True
    caps2 = get_capabilities(dev)
    assert caps1 is caps2


def test_device_capabilities_hash_is_sha256() -> None:
    """Capabilities hash should be a valid SHA256 hex string."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    assert caps.capabilities_hash is not None
    assert len(caps.capabilities_hash) == 64  # SHA256 hex length
    # Should be valid hex
    int(caps.capabilities_hash, 16)  # Raises if not valid hex


def test_device_capabilities_hash_deterministic() -> None:
    """Capabilities hash should be deterministic for same device state."""
    dev = _get_single_device()
    caps1 = extract_capabilities(dev)
    caps2 = extract_capabilities(dev)
    assert caps1.capabilities_hash == caps2.capabilities_hash


def test_device_operation_info_sites_is_tuple_or_none() -> None:
    """DeviceOperationInfo.sites should be a tuple or None."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    for op_info in caps.operations.values():
        if op_info.sites is not None:
            assert isinstance(op_info.sites, tuple)


def test_device_capabilities_duration_units() -> None:
    """DeviceCapabilities should have duration unit and scale factor."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    # These may be None, but should exist
    assert hasattr(caps, "duration_unit")
    assert hasattr(caps, "duration_scale_factor")


def test_device_capabilities_length_units() -> None:
    """DeviceCapabilities should have length unit and scale factor."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    # These may be None, but should exist
    assert hasattr(caps, "length_unit")
    assert hasattr(caps, "length_scale_factor")


def test_device_capabilities_min_atom_distance() -> None:
    """DeviceCapabilities should have min_atom_distance field."""
    dev = _get_single_device()
    caps = extract_capabilities(dev)
    assert hasattr(caps, "min_atom_distance")
