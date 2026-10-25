# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for shared vector and matrix DD exports."""

from __future__ import annotations

import shutil
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.core.dd import DDPackage, MatrixDD, VectorDD

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
    """Render paths with spaces and report Graphviz failures."""
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
    filename = directory / "測定 with spaces;literal"

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

        dot.unlink()
        with pytest.raises(RuntimeError, match="Cannot find Graphviz dot"):
            dd.to_svg(str(filename))


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("binary", [False, True])
def test_serialized_bytes_and_errors(*, matrix: bool, binary: bool) -> None:
    """Preserve serialized values and translate malformed-input exceptions."""
    package = DDPackage(2)
    values = np.array([[1, 2j], [-3, -4j]]) if matrix else np.array([1, 2j, -3, -4j])
    dd = package.from_matrix(values) if matrix else package.from_vector(values)
    payload = dd.to_bytes(binary=binary)
    restored = type(dd).from_bytes(package, payload, binary=binary)
    actual = restored.get_matrix(1) if isinstance(restored, MatrixDD) else restored.get_vector()
    np.testing.assert_allclose(actual, values, rtol=1e-14, atol=0)
    malformed = payload + b"\0" if binary else b"1\ninvalid\n"
    with pytest.raises(RuntimeError):
        type(dd).from_bytes(package, malformed, binary=binary)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (b"\n1\n", ValueError),
        (b"invalid\n1\n", ValueError),
        (b"+-1\n1\n", ValueError),
        (b"2\n1\n", RuntimeError),
        (b"1x\n1\n", RuntimeError),
        (b"1\ninvalid\n", RuntimeError),
        (b"2147483648\n1\n", IndexError),
        (b"1\n1e400\n", IndexError),
        (b"1\n1e-400\n", IndexError),
    ],
)
def test_serialized_text_error_categories(payload: bytes, error: type[Exception]) -> None:
    """Retain numeric syntax, numeric range, and serialized-format error classes."""
    package = DDPackage(1)
    with pytest.raises(error):
        VectorDD.from_bytes(package, payload, binary=False)
