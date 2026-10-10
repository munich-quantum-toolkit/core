# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test metadata-only QDMI manifest discovery."""

from __future__ import annotations

import json
import sys
from importlib.metadata import distribution
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from mqt.core import _qdmi_discovery  # ruff: ignore[import-private-name]

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_console_scripts import ScriptRunner


class _Distribution:
    def __init__(self, root: Path, files: list[str] | None) -> None:
        self.root = root
        self.files = None if files is None else [PurePosixPath(file) for file in files]

    def locate_file(self, file: PurePosixPath) -> Path:
        return self.root / file


def _entry(
    root: Path,
    files: list[str] | None,
    *,
    name: str = "vendor",
    value: str = "vendor.device",
) -> object:
    return SimpleNamespace(name=name, value=value, dist=_Distribution(root, files))


def test_discovers_record_manifest_without_importing_owner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Resolve a manifest through wheel metadata without importing its package."""
    manifest = "vendor/device/data/lib/device.qdmi.json"
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "__init__.py").write_text("raise RuntimeError\n")
    monkeypatch.syspath_prepend(tmp_path)
    second = "vendor/device/data/lib/second.qdmi.json"
    monkeypatch.setattr(_qdmi_discovery, "entry_points", lambda **_: [_entry(tmp_path, [manifest, second])])

    discovered: list[Path] = []
    _qdmi_discovery.discover_qdmi_manifests(discovered.append)

    assert discovered == [tmp_path / manifest, tmp_path / second]
    assert "vendor" not in sys.modules


def test_skips_failed_entry_point_enumeration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warn once when the metadata backend cannot enumerate entry points."""

    def fail_enumeration(**_: object) -> Iterator[object]:
        yield from ()
        msg = "broken metadata"
        raise RuntimeError(msg)

    monkeypatch.setattr(_qdmi_discovery, "entry_points", fail_enumeration)
    with pytest.warns(RuntimeWarning, match="Skipping QDMI manifest discovery") as warnings:
        _qdmi_discovery.discover_qdmi_manifests(lambda _: pytest.fail("must skip"))
    assert len(warnings) == 1


@pytest.mark.parametrize(
    ("files", "name", "value"),
    [
        (None, "device.qdmi.json", "vendor.device"),
        (["../device.qdmi.json"], "device.qdmi.json", "vendor.device"),
        (["other/device.qdmi.json"], "device.qdmi.json", "vendor.device"),
        (["vendor/device/data/device.qdmi.json"], "device.qdmi.json", "vendor/device"),
    ],
)
def test_skips_invalid_manifest_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    files: list[str] | None,
    name: str,
    value: str,
) -> None:
    """Warn once and skip malformed manifest metadata."""
    monkeypatch.setattr(
        _qdmi_discovery,
        "entry_points",
        lambda **_: [_entry(tmp_path, files, name=name, value=value)],
    )
    with pytest.warns(RuntimeWarning, match="Skipping QDMI manifest") as warnings:
        _qdmi_discovery.discover_qdmi_manifests(lambda _: pytest.fail("must skip"))
    assert len(warnings) == 1


@pytest.mark.script_launch_mode("subprocess")
def test_availability_discovers_installed_provider(
    script_runner: ScriptRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Discover installed manifests in the native checker without importing providers."""
    core = distribution("mqt-core")
    packaged = next(file for file in core.files or () if file.name == "mqt-core-qdmi-sc-device.qdmi.json")
    packaged_path = Path(str(core.locate_file(packaged)))
    definition = json.loads(packaged_path.read_text(encoding="utf-8"))["qdmi"]["devices"][0]
    definition.update(id="test.installed", library=str(packaged_path.parent / definition["library"]))

    provider = tmp_path / "test_qdmi_provider"
    provider.mkdir()
    (provider / "__init__.py").write_text("raise RuntimeError('provider must not be imported')\n")
    (provider / "device.qdmi.json").write_text(json.dumps({"schema-version": 1, "qdmi": {"devices": [definition]}}))
    (provider / "invalid.qdmi.json").write_text("{")
    metadata = tmp_path / "test_qdmi_provider-0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: test-qdmi-provider\nVersion: 0.0\n")
    (metadata / "entry_points.txt").write_text("[mqt.core.qdmi.manifests]\nprobe = test_qdmi_provider\n")
    (metadata / "RECORD").write_text("test_qdmi_provider/device.qdmi.json,,\ntest_qdmi_provider/invalid.qdmi.json,,\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    result = script_runner.run(["mqt-core-qdmi-check", "--device", "test.installed"])
    assert result.success
    assert not result.stdout

    monkeypatch.setenv(
        "MQT_CORE_QDMI_CONFIG_JSON",
        json.dumps({"schema-version": 1, "qdmi": {"devices": [{"id": "test.installed", "enabled": False}]}}),
    )
    assert not script_runner.run(["mqt-core-qdmi-check", "--device", "test.installed"]).success
