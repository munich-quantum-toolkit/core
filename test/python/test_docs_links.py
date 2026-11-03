# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check navigation validation independently of the documentation generators."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.linkcheck


@pytest.fixture(scope="module")
def check_links() -> Callable[[Path], list[str]]:
    """Load the documentation link checker.

    Returns:
        The link-checking function from the source checkout.
    """
    return runpy.run_path(str(Path(__file__).parents[2] / "scripts" / "check_docs_links.py"))["check_links"]


def test_generated_navigation(tmp_path: Path, check_links: Callable[[Path], list[str]]) -> None:
    """Accept valid relative and legacy anchors; reject missing generated targets."""
    (tmp_path / "index.html").write_text(
        '<a href="nested/page.html#hello%20world">valid</a>'
        '<a href="nested/page.html#legacy">legacy</a>'
        '<a href="https://example.org/missing">external</a>'
        '<a href="missing.html">missing file</a>'
        '<a href="nested/page.html#absent">missing anchor</a>'
        '<script src="missing.js"></script>'
        '<link rel="stylesheet" href="missing.css">'
        '<script src="https://example.org/external.js"></script>'
        '<link rel="stylesheet" href="style.css">',
        encoding="utf-8",
    )
    (tmp_path / "nested").mkdir()
    (tmp_path / "style.css").touch()
    (tmp_path / "nested" / "page.html").write_text(
        '<h1 id="hello world">Hello</h1><a name="legacy"></a><a href="../index.html">home</a>',
        encoding="utf-8",
    )
    assert check_links(tmp_path) == [
        "index.html: missing file: missing.html",
        "index.html: missing anchor: nested/page.html#absent",
        "index.html: missing file: missing.js",
        "index.html: missing file: missing.css",
    ]


def test_missing_build(tmp_path: Path, check_links: Callable[[Path], list[str]]) -> None:
    """An empty output directory must not count as a validated site."""
    assert check_links(tmp_path) == [f"No HTML pages found in {tmp_path}"]
