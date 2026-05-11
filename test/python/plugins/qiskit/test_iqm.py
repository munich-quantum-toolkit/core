# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the IQM-specific QDMI integration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from iqm.qdmi import IQM_QDMI_LIBRARY_PATH

from mqt.core.plugins.qiskit import iqm as mqt_iqm
from mqt.core.plugins.qiskit.estimator import QDMIEstimator
from mqt.core.plugins.qiskit.sampler import QDMISampler

if TYPE_CHECKING:
    from collections.abc import Callable


def _patch_backend_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the heavy QDMI backend initialization with a no-op for wrapper tests."""

    def no_op_backend_init(self: object, device: object, provider: object | None = None) -> None:
        _ = (self, device, provider)

    monkeypatch.setattr(mqt_iqm.QDMIBackend, "__init__", no_op_backend_init)


def _capture_loader_call(recorded_call: dict[str, object], *, device: object) -> Callable[..., object]:
    """Create a loader stub that records its arguments and returns ``device``.

    Returns:
        A stub for ``add_dynamic_device_library`` that records the received arguments.
    """

    def fake_add_dynamic_device_library(library_path: str, prefix: str, **kwargs: object) -> object:
        recorded_call.update({"library_path": library_path, "prefix": prefix, **kwargs})
        return device

    return fake_add_dynamic_device_library


def _build_backend(monkeypatch: pytest.MonkeyPatch) -> mqt_iqm.IQMBackend:
    """Create an IQM backend instance without invoking the native device loader.

    Returns:
        An ``IQMBackend`` instance whose native loader and heavy backend initialization are stubbed.
    """
    _patch_backend_init(monkeypatch)

    def fake_add_dynamic_device_library(library_path: str, prefix: str, **kwargs: object) -> object:
        _ = (library_path, prefix, kwargs)
        return object()

    monkeypatch.setattr(mqt_iqm.fomac, "add_dynamic_device_library", fake_add_dynamic_device_library)
    return mqt_iqm.IQMBackend(base_url="https://resonance.iqm.tech")


def test_iqm_backend_uses_installed_iqm_qdmi_library() -> None:
    """The MQT Core IQM wrapper should reuse the installed ``iqm-qdmi`` library path."""
    iqm_qdmi_module = mqt_iqm.HAS_IQM_QDMI.require_module("run IQM QDMI wrapper tests")
    library_path = Path(iqm_qdmi_module.IQM_QDMI_LIBRARY_PATH)

    assert library_path == IQM_QDMI_LIBRARY_PATH
    assert library_path.is_absolute()
    assert library_path.is_file()


def test_iqm_backend_passes_explicit_configuration_with_installed_library(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The MQT Core wrapper should forward explicit IQM configuration with the installed library path."""
    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{}", encoding="utf-8")

    _patch_backend_init(monkeypatch)

    recorded_loader_call: dict[str, object] = {}
    monkeypatch.setattr(
        mqt_iqm.fomac,
        "add_dynamic_device_library",
        _capture_loader_call(recorded_loader_call, device=object()),
    )

    backend_kwargs = {
        "base_url": "https://resonance.iqm.tech",
        "token": "explicit-token",
        "tokens_file": str(auth_file),
        "qc_id": "garnet",
        "qc_alias": "garnet-alias",
    }

    mqt_iqm.IQMBackend(**backend_kwargs)

    assert recorded_loader_call == {
        "library_path": str(IQM_QDMI_LIBRARY_PATH),
        "prefix": "IQM",
        "base_url": "https://resonance.iqm.tech",
        "token": "explicit-token",
        "auth_file": str(auth_file),
        "custom1": "garnet",
        "custom2": "garnet-alias",
    }


def test_iqm_backend_uses_documented_environment_fallbacks_with_installed_library(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The MQT Core wrapper should use the documented IQM environment-variable precedence."""
    auth_file = tmp_path / "env-auth.json"
    auth_file.write_text("{}", encoding="utf-8")
    env_token = auth_file.name

    _patch_backend_init(monkeypatch)

    monkeypatch.setenv("IQM_BASE_URL", "https://env.resonance.iqm.tech")
    monkeypatch.setenv("IQM_TOKEN", env_token)
    monkeypatch.setenv("RESONANCE_API_KEY", "fallback-token")
    monkeypatch.setenv("IQM_TOKENS_FILE", str(auth_file))
    monkeypatch.setenv("IQM_QC_ID", "env-qc-id")
    monkeypatch.setenv("IQM_QC_ALIAS", "env-qc-alias")

    recorded_loader_call: dict[str, object] = {}
    monkeypatch.setattr(
        mqt_iqm.fomac,
        "add_dynamic_device_library",
        _capture_loader_call(recorded_loader_call, device=object()),
    )

    mqt_iqm.IQMBackend()

    assert recorded_loader_call == {
        "library_path": str(IQM_QDMI_LIBRARY_PATH),
        "prefix": "IQM",
        "base_url": "https://env.resonance.iqm.tech",
        "token": env_token,
        "auth_file": str(auth_file),
        "custom1": "env-qc-id",
        "custom2": "env-qc-alias",
    }


def test_iqm_backend_sampler_uses_real_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sampler helper should return the real sampler bound to the IQM backend instance."""
    backend = _build_backend(monkeypatch)

    sampler = backend.sampler(default_shots=2048, options={"resilience_level": 2})

    assert isinstance(sampler, QDMISampler)
    assert sampler.backend is backend
    assert sampler._default_shots == 2048  # noqa: SLF001
    assert sampler._options == {"resilience_level": 2}  # noqa: SLF001


def test_iqm_backend_estimator_uses_real_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    """The estimator helper should return the real estimator bound to the IQM backend instance."""
    backend = _build_backend(monkeypatch)

    estimator = backend.estimator(default_precision=0.25, options={"default_shots": 512})

    assert isinstance(estimator, QDMIEstimator)
    assert estimator.backend is backend
    assert estimator._default_precision == pytest.approx(0.25)  # noqa: SLF001
    assert estimator._options == {"default_shots": 512}  # noqa: SLF001
