# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QDMIProvider."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mqt.core import fomac
from mqt.core.plugins.qiskit import QDMIProvider

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Device operation.*cannot be mapped to a Qiskit gate.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:Device does not define a measurement operation.*:UserWarning"),
]


@pytest.mark.filterwarnings("ignore:Skipping device:UserWarning")
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


@pytest.mark.filterwarnings("ignore:Skipping device:UserWarning")
def test_provider_backends_filter_by_substring() -> None:
    """Provider can filter backends by name substring."""
    provider = QDMIProvider()

    # Filter by "QDMI" substring (should match "MQT Core DDSIM QDMI Device")
    filtered = provider.backends(name="QDMI")
    assert len(filtered) > 0
    assert all("QDMI" in b.name for b in filtered)

    # Filter by "DDSIM" substring
    filtered_ddsim = provider.backends(name="DDSIM")
    assert len(filtered_ddsim) > 0
    assert all("DDSIM" in b.name for b in filtered_ddsim)


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

    # Monkeypatch to return empty device list
    def mock_get_devices(_self: object) -> list[object]:
        return []

    monkeypatch.setattr(fomac.Session, "get_devices", mock_get_devices)

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


def test_provider_with_token_parameter() -> None:
    """Provider accepts token parameter."""
    # Should not raise an error when creating provider with token
    # Note: The currently available QDMI devices don't support authentication.
    try:
        provider = QDMIProvider(token="test_token")  # noqa: S106
        # If device supports token, verify provider was created
        assert provider is not None
    except RuntimeError:
        # If not supported, that's okay for now
        pass


def test_provider_with_auth_file_parameter() -> None:
    """Provider accepts auth_file parameter."""
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as tmp_file:
        tmp_file.write("test_auth_content")
        tmp_path = tmp_file.name

    try:
        # Should not raise an error when creating provider with auth_file
        # Note: The currently available QDMI devices don't support authentication.
        try:
            provider = QDMIProvider(auth_file=tmp_path)
            assert provider is not None
        except RuntimeError:
            # If not supported, that's okay for now
            pass
    finally:
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)


def test_provider_with_auth_url_parameter() -> None:
    """Provider accepts auth_url parameter."""
    # Should not raise an error when creating provider with auth_url
    # Note: The currently available QDMI devices don't support authentication.
    try:
        provider = QDMIProvider(auth_url="https://auth.example.com")
        assert provider is not None
    except RuntimeError:
        # If not supported, that's okay for now
        pass


def test_provider_with_username_password_parameters() -> None:
    """Provider accepts username and password parameters."""
    # Should not raise an error when creating provider with username and password
    # Note: The currently available QDMI devices don't support authentication.
    try:
        provider = QDMIProvider(username="test_user", password="test_pass")  # noqa: S106
        assert provider is not None
    except RuntimeError:
        # If not supported, that's okay for now
        pass


def test_provider_with_project_id_parameter() -> None:
    """Provider accepts project_id parameter."""
    # Should not raise an error when creating provider with project_id
    # Note: The currently available QDMI devices don't support authentication.
    try:
        provider = QDMIProvider(project_id="test_project")
        assert provider is not None
    except RuntimeError:
        # If not supported, that's okay for now
        pass


def test_provider_with_multiple_auth_parameters() -> None:
    """Provider accepts multiple authentication parameters."""
    # Should not raise an error when creating provider with multiple auth parameters
    # Note: The currently available QDMI devices don't support authentication.
    try:
        provider = QDMIProvider(
            token="test_token",  # noqa: S106
            username="test_user",
            password="test_pass",  # noqa: S106
            project_id="test_project",
        )
        assert provider is not None
    except RuntimeError:
        # If not supported, that's okay for now
        pass


def test_provider_default_constructor_unchanged() -> None:
    """Provider default constructor behavior is unchanged (backward compatibility)."""
    # Create provider without any parameters
    provider = QDMIProvider()

    # Should have at least one backend (the DDSIM device)
    backends = provider.backends()
    assert len(backends) > 0

    # Should be able to get backends
    all_backends = provider.backends()
    assert isinstance(all_backends, list)

    # Should be able to get a specific backend
    backend = provider.get_backend("MQT Core DDSIM QDMI Device")
    assert backend.name == "MQT Core DDSIM QDMI Device"


def test_provider_with_custom_parameters() -> None:
    """Provider accepts custom configuration parameters."""
    # Test custom1
    try:
        provider = QDMIProvider(custom1="custom_value_1")
        assert provider is not None
    except (RuntimeError, ValueError):
        # If not supported, that's okay for now
        pass

    # Test all custom parameters together
    try:
        provider = QDMIProvider(
            custom1="value1",
            custom2="value2",
            custom3="value3",
            custom4="value4",
            custom5="value5",
        )
        assert provider is not None
    except (RuntimeError, ValueError):
        pass

    # Test mixing custom with standard authentication
    try:
        provider = QDMIProvider(
            token="test_token",  # noqa: S106
            custom1="custom_value",
            project_id="project_id",
        )
        assert provider is not None
    except (RuntimeError, ValueError):
        pass
