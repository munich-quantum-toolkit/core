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
from typing import TYPE_CHECKING

import pytest

from mqt.core.plugins.qiskit import QDMIBackend, QDMIProvider
from mqt.core.plugins.qiskit.exceptions import UnsupportedDeviceError

if TYPE_CHECKING:
    from mqt.core.qdmi import Device


def test_provider_discovers_and_filters_backends() -> None:
    """List installed devices, filter by substring, and retrieve the DDSIM backend by name."""
    provider = QDMIProvider()
    names = {backend.name for backend in provider.backends()}
    ddsim = "MQT Core DDSIM QDMI Device"
    assert ddsim in names
    assert {backend.name for backend in provider.backends(name="DDSIM")} == {
        name for name in names if name is not None and "DDSIM" in name
    }

    backend = provider.get_backend(ddsim)
    assert backend.name == ddsim
    assert backend.provider is provider


def test_provider_get_backend_stops_after_exact_match(
    monkeypatch: pytest.MonkeyPatch, ddsim_backend: QDMIBackend
) -> None:
    """Exact-name lookup does not open devices after the matching backend."""
    opened_ids: list[str] = []
    monkeypatch.setattr(
        QDMIProvider,
        "device_ids",
        staticmethod(lambda: ["unavailable.device", "matching.device", "later.device"]),
    )

    def lookup(device_id: str, **_session_parameters: object) -> Device:
        opened_ids.append(device_id)
        if device_id == "unavailable.device":
            msg = "credential=do-not-disclose"
            raise RuntimeError(msg)
        if device_id == "later.device":
            pytest.fail("exact-name lookup continued after finding its backend")
        return ddsim_backend.device

    provider = QDMIProvider()
    monkeypatch.setattr("mqt.core.plugins.qiskit.provider.open_device", lookup)

    expected_name = ddsim_backend.name
    assert expected_name is not None
    with pytest.warns(RuntimeWarning, match="unavailable.device"):
        assert provider.get_backend(expected_name).device_id == "matching.device"
    assert opened_ids == ["unavailable.device", "matching.device"]


def test_provider_get_backend_requires_exact_name() -> None:
    """A substring can filter the catalogue but cannot select a backend by name."""
    provider = QDMIProvider()
    with pytest.raises(ValueError, match="No backend found with name"):
        provider.get_backend("DDSIM")


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


def test_provider_construction_opens_no_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing a provider does not initialize any device."""
    monkeypatch.setattr(
        "mqt.core.plugins.qiskit.provider.QDMIBackend.from_device_id",
        lambda *_args, **_kwargs: pytest.fail("provider construction opened a device"),
    )
    QDMIProvider()


def test_provider_reads_catalogue_on_each_discovery_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discovery uses the driver's metadata without caching another copy."""
    device_ids = ["first.device"]
    monkeypatch.setattr(
        "mqt.core.plugins.qiskit.provider.registered_device_ids",
        lambda: list(device_ids),
    )
    provider = QDMIProvider()

    assert provider.device_ids() == ["first.device"]
    device_ids.append("second.device")
    assert provider.device_ids() == ["first.device", "second.device"]


def test_provider_exact_id_lookup_forwards_session_parameters(
    monkeypatch: pytest.MonkeyPatch, ddsim_backend: QDMIBackend
) -> None:
    """Exact-ID lookup opens only that device with typed overrides."""
    observed: tuple[str, QDMIProvider, dict[str, object]] | None = None

    def fake_from_device_id(
        device_id: str,
        *,
        provider: QDMIProvider,
        **session_parameters: object,
    ) -> QDMIBackend:
        nonlocal observed
        observed = device_id, provider, session_parameters
        return ddsim_backend

    monkeypatch.setattr("mqt.core.plugins.qiskit.provider.QDMIBackend.from_device_id", fake_from_device_id)
    provider = QDMIProvider()
    token = str(123)

    backend = provider.get_backend_by_device_id("test.device", token=token, custom1="queue")

    assert backend is ddsim_backend
    assert observed == ("test.device", provider, {"token": token, "custom1": "queue"})


def test_provider_skips_unavailable_and_incompatible_devices(
    monkeypatch: pytest.MonkeyPatch, ddsim_backend: QDMIBackend
) -> None:
    """Return available backends and warn only for a device that failed to open."""
    monkeypatch.setattr(
        QDMIProvider,
        "device_ids",
        staticmethod(lambda: ["unavailable.device", "incompatible.device", "available.device"]),
    )

    def lookup(device_id: str, **_session_parameters: object) -> Device:
        if device_id == "unavailable.device":
            msg = "credential=do-not-disclose"
            raise RuntimeError(msg)
        if device_id == "incompatible.device":
            msg = "device cannot be represented by a Qiskit target"
            raise UnsupportedDeviceError(msg)
        return ddsim_backend.device

    monkeypatch.setattr("mqt.core.plugins.qiskit.provider.open_device", lookup)
    with pytest.warns(RuntimeWarning, match="unavailable.device") as caught:
        (backend,) = QDMIProvider().backends()

    assert backend.device_id == "available.device"
    (warning,) = caught
    assert warning.filename == __file__
    message = str(warning.message)
    assert "credential" not in message
    assert "do-not-disclose" not in message


def test_provider_skips_snapshot_for_nonmatching_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Filtered discovery never snapshots an unrelated device, even if it would fail."""
    monkeypatch.setattr(QDMIProvider, "device_ids", staticmethod(lambda: ["mqt.ddsim.default"]))

    def fail_snapshot(*_args: object, **_kwargs: object) -> None:
        pytest.fail("nonmatching device was snapshotted")

    monkeypatch.setattr(QDMIBackend, "_build_target", fail_snapshot)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert QDMIProvider().backends("unrelated") == []
