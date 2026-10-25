# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for vector and matrix DD exports."""

from __future__ import annotations

import shutil
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.core.dd import DDPackage

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("matrix", [False, True])
def test_svg_without_graphviz_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, matrix: bool) -> None:
    """Render both DD types with PyGraphviz and retain the DOT sidecar."""
    package = DDPackage(1)
    dd = (
        package.from_matrix(np.array([[1, 1j], [1, -1j]]) / np.sqrt(2))
        if matrix
        else package.from_vector(np.array([1, 1j]) / np.sqrt(2))
    )
    monkeypatch.setenv("PATH", "")
    directory = tmp_path / "folder.with.dot"
    directory.mkdir()
    filename = directory / "diagram ü with spaces;literal"
    dd.to_svg(str(filename), edge_labels=True, format_as_polar=False)

    assert filename.with_suffix(".dot").read_text(encoding="utf-8") == dd.to_dot(
        edge_labels=True, format_as_polar=False
    )
    svg = filename.with_suffix(".svg").read_text(encoding="utf-8")
    assert "<svg " in svg
    assert 'class="edge"' in svg
    assert "+i" in svg
    assert "√2" in svg


def test_svg_native_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Render literal paths with the native Graphviz command."""
    monkeypatch.setitem(sys.modules, "pygraphviz", None)
    monkeypatch.chdir(tmp_path)
    if sys.platform == "win32":
        if shutil.which("dot.exe") is None:
            pytest.skip("Graphviz dot.exe is not installed")
    else:
        monkeypatch.setenv("PATH", str(tmp_path))
        dot = tmp_path / "dot"
        dot.write_text('#!/bin/sh\nprintf "native dot\\n" > "$4"\n', encoding="utf-8")
        dot.chmod(0o755)
    package = DDPackage(1)
    dd = package.zero_state(1)
    directory = tmp_path / "folder.with.dot"
    directory.mkdir()
    filename = directory / "diagram ü with spaces;literal"

    dd.to_svg(str(filename))

    svg = filename.with_suffix(".svg").read_text(encoding="utf-8")
    if sys.platform == "win32":
        assert "<svg" in svg
    else:
        assert svg.strip() == "native dot"
    assert filename.with_suffix(".dot").read_text(encoding="utf-8") == dd.to_dot()

    if sys.platform != "win32":
        dot.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Graphviz dot exited with code 7"):
            dd.to_svg(str(filename))


def test_svg_native_fallback_without_dot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Report a missing Graphviz executable."""
    monkeypatch.setitem(sys.modules, "pygraphviz", None)
    monkeypatch.setenv("PATH", "")
    monkeypatch.chdir(tmp_path)
    package = DDPackage(1)
    dd = package.zero_state(1)

    with pytest.raises(RuntimeError, match="Cannot start Graphviz dot"):
        dd.to_svg("diagram.svg")


def test_svg_unwritable_dot_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Report a DOT file that cannot be opened before rendering."""
    monkeypatch.setitem(sys.modules, "pygraphviz", None)
    package = DDPackage(1)
    dd = package.zero_state(1)

    with pytest.raises(RuntimeError, match="Cannot open DOT file"):
        dd.to_svg(str(tmp_path / "missing" / "diagram.svg"))
