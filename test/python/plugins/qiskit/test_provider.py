# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QDMIProvider."""

from __future__ import annotations

import pytest

from mqt.core.plugins.qiskit import QDMIProvider


def test_provider_backends_filter_by_name() -> None:
    """Provider can filter backends by name substring."""
    provider = QDMIProvider()

    # Get all backends first
    all_backends = provider.backends()
    assert len(all_backends) > 0

    # Filter by full name
    backend_name = all_backends[0].name
    filtered = provider.backends(name=backend_name)

    assert len(filtered) >= 1
    assert any(b.name == backend_name for b in filtered)


def test_provider_backends_filter_by_substring() -> None:
    """Provider can filter backends by name substring."""
    provider = QDMIProvider()

    # Filter by "QDMI" substring (should match "MQT Core DDSIM QDMI Device")
    filtered = provider.backends(name="QDMI")
    assert len(filtered) > 0
    for backend in filtered:
        assert backend.name is not None
        assert "QDMI" in backend.name

    # Filter by "DDSIM" substring
    filtered_ddsim = provider.backends(name="DDSIM")
    assert len(filtered_ddsim) > 0
    for backend in filtered_ddsim:
        assert backend.name is not None
        assert "DDSIM" in backend.name


def test_provider_backends_filter_nonexistent_name() -> None:
    """Provider returns empty list for non-existent name substring."""
    provider = QDMIProvider()
    backends = provider.backends(name="NonExistentDevice")
    assert backends == []


def test_provider_get_backend_by_name() -> None:
    """Provider can get backend by name."""
    provider = QDMIProvider()
    backend = provider.get_backend("MQT Core DDSIM QDMI Device")
    assert backend.name == "MQT Core DDSIM QDMI Device"
    assert backend.provider is provider


def test_provider_get_backend_nonexistent() -> None:
    """Provider raises ValueError for non-existent backend."""
    provider = QDMIProvider()
    with pytest.raises(ValueError, match="No backend found with name"):
        provider.get_backend("NonExistentDevice")


def test_provider_get_backend_no_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider raises ValueError when no devices available."""
    monkeypatch.setattr("mqt.core.plugins.qiskit.provider.registered_device_ids", list)

    provider = QDMIProvider()
    with pytest.raises(ValueError, match="No backend found with name"):
        provider.get_backend("MQT Core DDSIM QDMI Device")


def test_provider_repr() -> None:
    """Provider has a useful repr."""
    provider = QDMIProvider()
    repr_str = repr(provider)
    assert "QDMIProvider" in repr_str
    assert "backends=" in repr_str


def test_backend_has_provider_reference() -> None:
    """Backend created by provider has reference back to provider."""
    provider = QDMIProvider()
    backend = provider.get_backend("MQT Core DDSIM QDMI Device")

    assert backend.provider is provider


def test_provider_default_constructor() -> None:
    """Provider discovers registered devices without generic session parameters."""
    provider = QDMIProvider()
    backends = provider.backends()
    assert len(backends) > 0
    backend = provider.get_backend("MQT Core DDSIM QDMI Device")
    assert backend.name == "MQT Core DDSIM QDMI Device"
