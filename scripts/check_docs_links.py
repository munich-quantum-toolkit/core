#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check local links in the complete generated HTML site, including Doxygen."""

from __future__ import annotations

import sys
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
        if tag == "a":
            if anchor := attributes.get("name"):
                self.anchors.add(anchor)
            if href := attributes.get("href"):
                self.links.append(href)


def check_links(root: Path) -> list[str]:
    """Return diagnostics for missing local files or HTML anchors."""
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
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                errors.append(f"{path.relative_to(root)}: missing file: {href}")
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].anchors:
                errors.append(f"{path.relative_to(root)}: missing anchor: {href}")
    return errors


if __name__ == "__main__":
    diagnostics = check_links(Path(sys.argv[1]))
    for diagnostic in diagnostics:
        sys.stderr.write(f"{diagnostic}\n")
    sys.exit(bool(diagnostics))
