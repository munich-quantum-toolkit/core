# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Offline checks for the Catalyst device ABI canary."""

from __future__ import annotations

import importlib.util
import json
from email.message import Message
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError

import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.catalyst_abi


@pytest.fixture
def canary() -> ModuleType:
    """Load the standalone ABI checker.

    Returns:
        The standalone script module.
    """
    root = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location("catalyst_abi_check", root / "scripts/catalyst_abi_check.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("changed", [None, "include/QuantumDevice.hpp"])
@pytest.mark.parametrize("missing", [False, True])
def test_header_drift(
    canary: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    changed: str | None,
    *,
    missing: bool,
) -> None:
    """The canary accepts identical headers and names an altered ABI header."""
    root = Path(__file__).parents[2]
    snapshot = root / "vendor/catalyst/0.15.0"

    def download(url: str, *, timeout: int) -> BytesIO:
        assert timeout == 30
        if url == "https://pypi.org/pypi/pennylane-catalyst/json":
            return BytesIO(json.dumps({"info": {"version": "0.16.0"}}).encode())
        assert "/v0.16.0/" in url
        name = f"include/{url.rsplit('/', 1)[1]}"
        if name == changed and missing:
            with HTTPError(url, 404, "Not Found", Message(), BytesIO()) as error:
                raise error
        # GitHub serves LF bytes even when Git checks out headers with CRLF.
        content = (snapshot / name).read_text(encoding="utf-8").encode("utf-8")
        return BytesIO(b"changed header" if name == changed else content)

    monkeypatch.setattr(canary.urllib.request, "urlopen", download)
    assert canary.main() == int(changed is not None)
    if changed:
        assert changed in caplog.text


@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("error", [URLError("unavailable"), TimeoutError("timed out")])
def test_download_failure(
    canary: ModuleType, monkeypatch: pytest.MonkeyPatch, error: Exception, *, metadata: bool
) -> None:
    """Infrastructure failures have their own exit code instead of reporting drift."""

    def download(url: str, *, timeout: int) -> BytesIO:
        assert timeout == 30
        if not metadata and url == "https://pypi.org/pypi/pennylane-catalyst/json":
            return BytesIO(json.dumps({"info": {"version": "0.16.0"}}).encode())
        raise error

    monkeypatch.setattr(canary.urllib.request, "urlopen", download)
    assert canary.main() == 2


def test_connection_reset_during_read(canary: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """A disconnected response is a download failure, not ABI drift."""

    class DisconnectedResponse(BytesIO):
        @staticmethod
        def read(_size: int | None = -1, /) -> bytes:
            msg = "connection reset during read"
            raise ConnectionResetError(msg)

    def download(url: str, *, timeout: int) -> BytesIO:
        assert timeout == 30
        if url == "https://pypi.org/pypi/pennylane-catalyst/json":
            return BytesIO(json.dumps({"info": {"version": "0.16.0"}}).encode())
        return DisconnectedResponse()

    monkeypatch.setattr(canary.urllib.request, "urlopen", download)
    assert canary.main() == 2


@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("status", [404, 503])
def test_http_failure(canary: ModuleType, monkeypatch: pytest.MonkeyPatch, status: int, *, metadata: bool) -> None:
    """Only a missing ABI header is drift; metadata and server errors are failures."""

    def download(url: str, *, timeout: int) -> BytesIO:
        assert timeout == 30
        if not metadata and url == "https://pypi.org/pypi/pennylane-catalyst/json":
            return BytesIO(json.dumps({"info": {"version": "0.16.0"}}).encode())
        with HTTPError(url, status, "Unavailable", Message(), BytesIO()) as error:
            raise error

    monkeypatch.setattr(canary.urllib.request, "urlopen", download)
    assert canary.main() == (1 if not metadata and status == 404 else 2)
