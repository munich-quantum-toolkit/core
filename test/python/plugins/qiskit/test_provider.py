# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QDMIProvider."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from mqt.core.plugins.qiskit import QDMIBackend, QDMIProvider
from mqt.core.plugins.qiskit.exceptions import UnsupportedDeviceError
from mqt.core.qdmi import open_device


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


def test_provider_get_backend_stops_after_exact_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact-name lookup does not open devices after the matching backend."""
    expected = QDMIBackend(open_device("mqt.ddsim.default"), device_id="matching.device")
    opened_ids: list[str] = []
    monkeypatch.setattr(
        QDMIProvider,
        "device_ids",
        staticmethod(lambda: ["unavailable.device", "matching.device", "later.device"]),
    )

    def lookup(device_id: str, **_session_parameters: object) -> QDMIBackend:
        opened_ids.append(device_id)
        if device_id == "unavailable.device":
            msg = "credential=do-not-disclose"
            raise RuntimeError(msg)
        if device_id == "later.device":
            pytest.fail("exact-name lookup continued after finding its backend")
        return expected

    provider = QDMIProvider()
    monkeypatch.setattr(provider, "get_backend_by_device_id", lookup)

    expected_name = expected.name
    assert expected_name is not None
    with pytest.warns(RuntimeWarning, match="unavailable.device"):
        assert provider.get_backend(expected_name) is expected
    assert opened_ids == ["unavailable.device", "matching.device"]


def test_provider_get_backend_nonexistent() -> None:
    """Provider raises ValueError for non-existent backend."""
    provider = QDMIProvider()
    with pytest.raises(ValueError, match="No backend found with name"):
        provider.get_backend("NonExistentDevice")


def test_provider_get_backend_no_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider raises ValueError when no devices available."""
    monkeypatch.setattr(QDMIProvider, "device_ids", staticmethod(list))

    provider = QDMIProvider()
    with pytest.raises(ValueError, match="No backend found with name"):
        provider.get_backend("MQT Core DDSIM QDMI Device")


def test_provider_repr() -> None:
    """Provider has a useful repr."""
    provider = QDMIProvider()
    repr_str = repr(provider)
    assert "QDMIProvider" in repr_str
    assert "devices=" in repr_str


def test_backend_has_provider_reference() -> None:
    """Backend created by provider has reference back to provider."""
    provider = QDMIProvider()
    backend = provider.get_backend("MQT Core DDSIM QDMI Device")

    assert backend.provider is provider


def test_provider_default_constructor() -> None:
    """Provider discovers Client-visible devices without session parameters."""
    provider = QDMIProvider()
    backends = provider.backends()
    assert len(backends) > 0
    backend = provider.get_backend("MQT Core DDSIM QDMI Device")
    assert backend.name == "MQT Core DDSIM QDMI Device"


def test_provider_construction_opens_no_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing a provider does not initialize any device."""
    monkeypatch.setattr(
        "mqt.core.plugins.qiskit.provider.QDMIBackend.from_device_id",
        lambda *_args, **_kwargs: pytest.fail("provider construction opened a device"),
    )
    QDMIProvider()


def test_provider_reads_client_session_on_each_discovery_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider starts a fresh Client session for each discovery call."""
    device_ids = ["first.device"]
    monkeypatch.setattr(
        "mqt.core.plugins.qiskit.provider.ClientSession",
        lambda: SimpleNamespace(devices=[SimpleNamespace(id=device_id) for device_id in device_ids]),
    )
    provider = QDMIProvider()

    assert provider.device_ids() == ["first.device"]
    device_ids.append("second.device")
    assert provider.device_ids() == ["first.device", "second.device"]


def test_provider_exact_id_lookup_forwards_session_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact-ID lookup opens only that device with typed overrides."""
    observed: tuple[str, QDMIProvider, dict[str, object]] | None = None
    expected = QDMIBackend(open_device("mqt.ddsim.default"), device_id="test.device")

    def fake_from_device_id(
        device_id: str,
        *,
        provider: QDMIProvider,
        **session_parameters: object,
    ) -> QDMIBackend:
        nonlocal observed
        observed = device_id, provider, session_parameters
        return expected

    monkeypatch.setattr("mqt.core.plugins.qiskit.provider.QDMIBackend.from_device_id", fake_from_device_id)
    provider = QDMIProvider()
    token = str(123)

    backend = provider.get_backend_by_device_id("test.device", token=token, custom1="queue")

    assert backend is expected
    assert observed == ("test.device", provider, {"token": token, "custom1": "queue"})


def test_provider_warns_with_only_id_and_skips_unavailable_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enumeration reports an unavailable ID without leaking failure details."""
    available = QDMIBackend(open_device("mqt.ddsim.default"), device_id="available.device")
    monkeypatch.setattr(
        QDMIProvider,
        "device_ids",
        staticmethod(lambda: ["available.device", "unavailable.device"]),
    )

    def lookup(device_id: str, **_session_parameters: object) -> QDMIBackend:
        if device_id == "unavailable.device":
            msg = "credential=do-not-disclose"
            raise RuntimeError(msg)
        return available

    provider = QDMIProvider()
    monkeypatch.setattr(provider, "get_backend_by_device_id", lookup)

    with pytest.warns(RuntimeWarning) as warnings:
        assert provider.backends() == [available]

    message = str(warnings[0].message)
    assert warnings[0].filename == __file__
    assert "unavailable.device" in message
    assert "credential" not in message
    assert "do-not-disclose" not in message


def test_provider_silently_skips_incompatible_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enumeration silently omits devices that Qiskit cannot represent."""
    available = QDMIBackend(open_device("mqt.ddsim.default"), device_id="available.device")
    monkeypatch.setattr(
        QDMIProvider,
        "device_ids",
        staticmethod(lambda: ["incompatible.device", "available.device"]),
    )

    def lookup(device_id: str, **_session_parameters: object) -> QDMIBackend:
        if device_id == "incompatible.device":
            msg = "device cannot be represented by a Qiskit target"
            raise UnsupportedDeviceError(msg)
        return available

    provider = QDMIProvider()
    monkeypatch.setattr(provider, "get_backend_by_device_id", lookup)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert provider.backends() == [available]
