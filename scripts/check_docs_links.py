#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check local links, scripts, and stylesheets in Sphinx and Doxygen HTML."""

from __future__ import annotations

import sys
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    """Collect HTML link destinations and both modern and legacy anchors."""

    def __init__(self) -> None:
        """Initialize per-page anchor and link collections."""
        super().__init__()
        self.anchors: set[str] = set()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record an element's anchors and hyperlinks."""
        attributes = dict(attrs)
        if anchor := attributes.get("id"):
            self.anchors.add(anchor)
        if tag == "a" and (anchor := attributes.get("name")):
            self.anchors.add(anchor)
        if tag in {"a", "link"} and (href := attributes.get("href")):
            self.links.append(href)
        elif tag == "script" and (src := attributes.get("src")):
            self.links.append(src)


def check_links(root: Path) -> list[str]:
    """Return missing-file and anchor diagnostics for a completed HTML build."""

    @lru_cache
    def resolve_target(path: Path) -> tuple[Path, bool]:
        target = path.resolve()
        if target.is_dir():
            target /= "index.html"
        return target, target.is_file()

    root = root.resolve()
    pages = {}
    for path in sorted(root.rglob("*.html")):
        page = Page()
        page.feed(path.read_text(encoding="utf-8"))
        pages[path] = page
    if not pages:
        return [f"No HTML pages found in {root}"]
    errors = []
    for path, page in pages.items():
        for href in page.links:
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            target, exists = resolve_target(path.parent / unquote(url.path)) if url.path else (path, True)
            if not exists:
                errors.append(f"{path.relative_to(root)}: missing file: {href}")
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].anchors:
                errors.append(f"{path.relative_to(root)}: missing anchor: {href}")
    return errors


if __name__ == "__main__":
    diagnostics = check_links(Path(sys.argv[1]))
    for diagnostic in diagnostics:
        sys.stderr.write(f"{diagnostic}\n")
    sys.exit(bool(diagnostics))
