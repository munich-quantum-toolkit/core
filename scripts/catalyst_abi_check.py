#!/usr/bin/env -S uv run --script --quiet
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging>=24"]
# ///

"""Check the newest released Catalyst headers against the vendored device ABI."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import urllib.request
from pathlib import Path

from packaging.version import Version

LOGGER = logging.getLogger(__name__)
VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "catalyst"


def main() -> int:
    """Report changed ABI headers.

    Returns:
        One when headers changed, otherwise zero.
    """
    snapshot = max(VENDOR.iterdir(), key=lambda path: Version(path.name))
    provenance = json.loads((snapshot / "PROVENANCE.json").read_text())
    with urllib.request.urlopen("https://pypi.org/pypi/pennylane-catalyst/json", timeout=30) as response:
        release = Version(json.load(response)["info"]["version"])
    source = provenance["source"]
    repository = source["repository"].removeprefix("https://github.com/")
    changed = []
    for name, expected in provenance["files"].items():
        if not name.startswith("include/"):
            continue
        path = f"{source['directory']}/{name.removeprefix('include/')}"
        url = f"https://raw.githubusercontent.com/{repository}/v{release}/{path}"
        with urllib.request.urlopen(url, timeout=30) as response:
            if hashlib.sha256(response.read()).hexdigest() != expected:
                changed.append(name)
    if changed:
        LOGGER.error("Catalyst %s differs from vendored %s: %s", release, snapshot.name, ", ".join(changed))
        return 1
    LOGGER.info("Catalyst %s matches the vendored %s device interface.", release, snapshot.name)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
