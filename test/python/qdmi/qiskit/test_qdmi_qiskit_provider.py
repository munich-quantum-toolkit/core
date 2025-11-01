# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QDMIProvider."""

from __future__ import annotations

import pytest

from mqt.core.qdmi.qiskit import QDMIProvider

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Device operation.*cannot be mapped to a Qiskit gate.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:Device does not define a measurement operation.*:UserWarning"),
]


def test_provider_backends_filter_by_name() -> None:
    """Provider can filter backends by name."""
    provider = QDMIProvider()

    # Get all backends first
    all_backends = provider.backends()
    assert len(all_backends) > 0

    # Get specific backend by name
    backend_name = all_backends[0].name
    filtered = provider.backends(name=backend_name)

    assert len(filtered) == 1
    assert filtered[0].name == backend_name


def test_provider_backends_filter_nonexistent_name() -> None:
    """Provider returns empty list for non-existent name."""
    provider = QDMIProvider()
    backends = provider.backends(name="NonExistentDevice")
    assert backends == []


def test_provider_get_backend_by_name() -> None:
    """Provider can get backend by name."""
    provider = QDMIProvider()
    backend = provider.get_backend("MQT NA Default QDMI Device")
    assert backend.name == "MQT NA Default QDMI Device"
    assert backend.provider is provider


def test_provider_get_backend_nonexistent() -> None:
    """Provider raises ValueError for non-existent backend."""
    provider = QDMIProvider()
    with pytest.raises(ValueError, match="No backend found with name"):
        provider.get_backend("NonExistentDevice")


def test_provider_get_backend_no_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider raises ValueError when no devices available."""
    from mqt.core import fomac

    # Monkeypatch to return empty device list
    def mock_devices() -> list[object]:
        return []

    monkeypatch.setattr(fomac, "devices", mock_devices)

    provider = QDMIProvider()
    with pytest.raises(ValueError, match="No backend found with name"):
        provider.get_backend("MQT NA Default QDMI Device")


def test_provider_repr() -> None:
    """Provider has a useful repr."""
    provider = QDMIProvider()
    repr_str = repr(provider)
    assert "QDMIProvider" in repr_str
    assert "backends=" in repr_str


def test_provider_backends_return_different_instances() -> None:
    """Provider creates new backend instances each time."""
    provider = QDMIProvider()

    backends1 = provider.backends()
    backends2 = provider.backends()

    # Should be different instances
    assert backends1[0] is not backends2[0]


def test_backend_has_provider_reference() -> None:
    """Backend created by provider has reference back to provider."""
    provider = QDMIProvider()
    backend = provider.get_backend("MQT NA Default QDMI Device")

    assert backend.provider is provider
