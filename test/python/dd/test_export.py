# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for shared vector and matrix DD exports."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.core.dd import DDPackage

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("colored", [False, True])
@pytest.mark.parametrize("classic", [False, True])
@pytest.mark.parametrize("edge_labels", [False, True])
def test_dot_options(*, matrix: bool, colored: bool, classic: bool, edge_labels: bool) -> None:
    """Keep export options available for both DD types."""
    package = DDPackage(1)
    dd = (
        package.from_matrix(np.array([[1, 1j], [1, -1j]]) / np.sqrt(2))
        if matrix
        else package.from_vector(np.array([1, 1j]) / np.sqrt(2))
    )
    dot = dd.to_dot(colored=colored, classic=classic, edge_labels=edge_labels)
    edges = "\n".join(line for line in dot.splitlines() if "->" in line)
    assert (' color="' in edges) == colored
    assert ("shape=circle" in dot) == classic
    assert ('label=<<font point-size="8">' in edges) == edge_labels


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
    dd.to_svg(str(tmp_path / "diagram.svg"), edge_labels=True, format_as_polar=False)

    assert (tmp_path / "diagram.dot").read_text(encoding="utf-8") == dd.to_dot(edge_labels=True, format_as_polar=False)
    svg = (tmp_path / "diagram.svg").read_text(encoding="utf-8")
    assert "<svg " in svg
    assert 'class="edge"' in svg
    assert "+i" in svg
    assert "√2" in svg


def test_svg_native_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the native dot command when PyGraphviz is absent."""
    monkeypatch.setitem(sys.modules, "pygraphviz", None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    dot = tmp_path / ("dot.bat" if sys.platform == "win32" else "dot")
    dot.write_text(
        '@echo off\n> "%~4" echo native dot\n'
        if sys.platform == "win32"
        else '#!/bin/sh\nprintf "native dot\\n" > "$4"\n',
        encoding="utf-8",
    )
    dot.chmod(0o755)
    package = DDPackage(1)
    dd = package.zero_state(1)

    dd.to_svg("diagram.svg")

    assert (tmp_path / "diagram.svg").read_text(encoding="utf-8").strip() == "native dot"
    assert (tmp_path / "diagram.dot").read_text(encoding="utf-8") == dd.to_dot()
