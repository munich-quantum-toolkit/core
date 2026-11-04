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
from io import BytesIO
from pathlib import Path

import pytest

pytestmark = pytest.mark.catalyst_abi


@pytest.mark.parametrize("changed", [None, "include/QuantumDevice.hpp"])
def test_header_drift(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, changed: str | None) -> None:
    """The canary accepts identical headers and names an altered ABI header."""
    root = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location("catalyst_abi_check", root / "scripts/catalyst_abi_check.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshot = root / "vendor/catalyst/0.15.0"

    def download(url: str, *, timeout: int) -> BytesIO:
        assert timeout == 30
        if url == "https://pypi.org/pypi/pennylane-catalyst/json":
            return BytesIO(json.dumps({"info": {"version": "0.16.0"}}).encode())
        assert "/v0.16.0/" in url
        name = f"include/{url.rsplit('/', 1)[1]}"
        return BytesIO(b"changed header" if name == changed else (snapshot / name).read_bytes())

    monkeypatch.setattr(module.urllib.request, "urlopen", download)
    assert module.main() == int(changed is not None)
    if changed:
        assert changed in caplog.text
