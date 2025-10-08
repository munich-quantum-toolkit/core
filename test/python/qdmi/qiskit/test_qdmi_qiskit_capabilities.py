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
