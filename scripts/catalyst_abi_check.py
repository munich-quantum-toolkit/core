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
# ///

"""Check the newest released Catalyst headers against the vendored device ABI."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import urllib.request
from pathlib import Path

LOGGER = logging.getLogger(__name__)
VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "catalyst"


def _check_headers() -> int:
    """Compare the released and vendored device headers.

    Returns:
        Zero for unchanged headers or one for drift.
    """
    snapshot = max(VENDOR.iterdir(), key=lambda path: tuple(map(int, path.name.split("."))))
    provenance = json.loads((snapshot / "PROVENANCE.json").read_text())
    with urllib.request.urlopen("https://pypi.org/pypi/pennylane-catalyst/json", timeout=30) as response:
        release = json.load(response)["info"]["version"]
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


def main() -> int:
    """Check the ABI and distinguish download failures from header drift.

    Returns:
        Zero for unchanged headers, one for drift, or two for a download failure.
    """
    try:
        return _check_headers()
    except OSError as error:
        LOGGER.log(logging.ERROR, "Could not check Catalyst ABI: %s", error)
        return 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
