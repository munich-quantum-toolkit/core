# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for shared vector and matrix DD exports."""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import pygraphviz
import pytest

from mqt.core.dd import DDPackage

if TYPE_CHECKING:
    from pathlib import Path
    from typing import NoReturn

    from mqt.core.dd import MatrixDD, VectorDD


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


@pytest.fixture(params=[False, True], ids=["vector", "matrix"])
def exported_dd(request: pytest.FixtureRequest) -> VectorDD | MatrixDD:
    """Create either DD type for export tests.

    Returns:
        A vector or matrix DD with complex edge weights.
    """
    package = DDPackage(1)
    return (
        package.from_matrix(np.array([[1, 1j], [1, -1j]]) / np.sqrt(2))
        if request.param
        else package.from_vector(np.array([1, 1j]) / np.sqrt(2))
    )


@pytest.mark.parametrize(
    ("filename", "options"),
    [
        ("diagram", {}),
        ("diagram.other", {"colored": False, "classic": True, "edge_labels": True, "format_as_polar": False}),
        ("diagram.svg", {"memory": True}),
    ],
)
def test_svg_without_graphviz_executable(
    exported_dd: VectorDD | MatrixDD,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    options: dict[str, bool],
) -> None:
    """Render real SVGs and retain DOT options without a Graphviz executable."""
    monkeypatch.setenv("PATH", "")
    exported_dd.to_svg(str(tmp_path / filename), **options)

    dot = (tmp_path / "diagram.dot").read_text(encoding="utf-8")
    assert dot == exported_dd.to_dot(**options)
    svg = (tmp_path / "diagram.svg").read_text(encoding="utf-8")
    assert "<svg " in svg
    assert 'xmlns="http://www.w3.org/2000/svg"' in svg
    assert 'class="edge"' in svg
    assert ">1</text>" in svg
    if not options.get("memory"):
        assert ">q</text>" in svg
    if options.get("edge_labels"):
        assert "+i" in svg
        assert "√2" in svg


@pytest.mark.parametrize("version", [None, "1.14"])
def test_svg_native_fallback(
    exported_dd: VectorDD | MatrixDD, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str | None
) -> None:
    """Keep the native dot command for absent or older PyGraphviz installations."""
    module = ModuleType("pygraphviz") if version else None
    if module is not None:
        monkeypatch.setattr(module, "__version__", version, raising=False)
    monkeypatch.setitem(sys.modules, "pygraphviz", module)
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

    exported_dd.to_svg("diagram.other")

    assert (tmp_path / "diagram.svg").read_text(encoding="utf-8").strip() == "native dot"
    assert (tmp_path / "diagram.dot").read_text(encoding="utf-8") == exported_dd.to_dot()


@pytest.mark.parametrize(
    ("error", "statement"),
    [
        (ModuleNotFoundError, 'raise ModuleNotFoundError("broken dependency", name="graphviz_dependency")'),
        (ModuleNotFoundError, 'raise ModuleNotFoundError("broken dependency")'),
        (ImportError, 'raise ImportError("broken extension")'),
    ],
)
def test_svg_import_errors(
    exported_dd: VectorDD | MatrixDD,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: type[ImportError],
    statement: str,
) -> None:
    """Report broken installed packages instead of silently using native dot."""
    monkeypatch.delitem(sys.modules, "pygraphviz")
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "pygraphviz.py").write_text(statement, encoding="utf-8")
    monkeypatch.setenv("PATH", "")

    with pytest.raises(error, match="broken"):
        exported_dd.to_svg(str(tmp_path / "diagram.svg"))


def test_svg_render_errors(exported_dd: VectorDD | MatrixDD, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve renderer errors for callers to diagnose."""

    def fail_render(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "SVG rendering failed"
        raise OSError(msg)

    monkeypatch.setattr(pygraphviz.AGraph, "draw", fail_render)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(OSError, match="SVG rendering failed"):
        exported_dd.to_svg(str(tmp_path / "diagram.svg"))


def test_import_without_pygraphviz() -> None:
    """Import Core and use DDs in a fresh interpreter without PyGraphviz."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['pygraphviz'] = None; "
                "from mqt.core.dd import DDPackage; "
                "package = DDPackage(1); assert package.zero_state(1).to_dot()"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
