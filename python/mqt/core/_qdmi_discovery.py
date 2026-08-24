# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Discover installed QDMI manifests without importing provider packages."""

from __future__ import annotations

import warnings
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_ENTRY_POINT_GROUP = "mqt.core.qdmi.manifests"


def _manifest_path(entry: EntryPoint) -> Path:
    name = entry.name
    if not name.endswith(".qdmi.json") or PurePosixPath(name).name != name or "\\" in name:
        msg = f"invalid manifest basename {name!r}"
        raise ValueError(msg)

    module_parts = entry.value.split(".")
    if not module_parts or not all(part.isidentifier() for part in module_parts):
        msg = f"invalid module anchor {entry.value!r}"
        raise ValueError(msg)

    distribution = entry.dist
    if distribution is None or distribution.read_text("RECORD") is None or distribution.files is None:
        msg = "distribution has no RECORD file list"
        raise ValueError(msg)

    prefix = PurePosixPath(*module_parts)
    matches = []
    for file in distribution.files:
        candidate = PurePosixPath(str(file))
        if ".." in candidate.parts or candidate.name != name:
            continue
        try:
            candidate.relative_to(prefix)
        except ValueError:
            continue
        matches.append(file)
    if len(matches) != 1:
        msg = f"expected one {name!r} below {prefix}, found {len(matches)}"
        raise ValueError(msg)
    return Path(str(distribution.locate_file(matches[0])))


def discover_qdmi_manifests(add_manifest: Callable[[Path], None]) -> None:
    """Stage each valid, explicitly advertised package manifest."""
    try:
        entries = tuple(entry_points(group=_ENTRY_POINT_GROUP))
    except Exception as error:  # ruff: ignore[blind-except]
        warnings.warn(
            f"Skipping QDMI manifest discovery: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    for entry in entries:
        try:
            add_manifest(_manifest_path(entry))
        except Exception as error:  # ruff: ignore[blind-except]
            warnings.warn(
                f"Skipping QDMI manifest entry point {entry.name!r}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
