# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test metadata-only QDMI manifest discovery."""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from mqt.core import _qdmi_discovery  # ruff: ignore[import-private-name]

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Distribution:
    def __init__(self, root: Path, files: list[str] | None, *, record: str | None = "") -> None:
        self.root = root
        self.files = None if files is None else [PurePosixPath(file) for file in files]
        self.record = record

    def locate_file(self, file: PurePosixPath) -> Path:
        return self.root / file

    def read_text(self, filename: str) -> str | None:
        assert filename == "RECORD"
        return self.record


def _entry(
    root: Path,
    files: list[str] | None,
    *,
    name: str = "device.qdmi.json",
    value: str = "vendor.device",
    record: str | None = "",
) -> object:
    return SimpleNamespace(name=name, value=value, dist=_Distribution(root, files, record=record))


def test_discovers_record_manifest_without_importing_owner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Resolve a manifest through wheel metadata without importing its package."""
    manifest = "vendor/device/data/lib/device.qdmi.json"
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "__init__.py").write_text("raise RuntimeError\n")
    path = tmp_path / manifest
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    monkeypatch.setattr(_qdmi_discovery, "entry_points", lambda **_: [_entry(tmp_path, [manifest])])

    discovered: list[Path] = []
    _qdmi_discovery.discover_qdmi_manifests(discovered.append)

    assert discovered == [path]
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


def test_skips_file_list_without_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reject file lists that do not come from wheel RECORD metadata."""
    manifest = "vendor/device/data/device.qdmi.json"
    monkeypatch.setattr(
        _qdmi_discovery,
        "entry_points",
        lambda **_: [_entry(tmp_path, [manifest], record=None)],
    )

    with pytest.warns(RuntimeWarning, match="no RECORD") as warnings:
        _qdmi_discovery.discover_qdmi_manifests(lambda _: pytest.fail("must skip"))
    assert len(warnings) == 1


@pytest.mark.parametrize(
    ("files", "name", "value"),
    [
        (None, "device.qdmi.json", "vendor.device"),
        (
            ["vendor/device/a/device.qdmi.json", "vendor/device/b/device.qdmi.json"],
            "device.qdmi.json",
            "vendor.device",
        ),
        (["../device.qdmi.json"], "device.qdmi.json", "vendor.device"),
        (["other/device.qdmi.json"], "device.qdmi.json", "vendor.device"),
        (["vendor/device/data/device.qdmi.json"], "../device.qdmi.json", "vendor.device"),
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
    """Warn once and skip malformed or ambiguous manifest metadata."""
    monkeypatch.setattr(
        _qdmi_discovery,
        "entry_points",
        lambda **_: [_entry(tmp_path, files, name=name, value=value)],
    )
    with pytest.warns(RuntimeWarning, match="Skipping QDMI manifest") as warnings:
        _qdmi_discovery.discover_qdmi_manifests(lambda _: pytest.fail("must skip"))
    assert len(warnings) == 1
