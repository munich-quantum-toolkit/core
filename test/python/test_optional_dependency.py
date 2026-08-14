# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for optional dependency checks."""

from __future__ import annotations

import pytest

from mqt.core._compat import optional  # ruff:ignore[import-private-name]
from mqt.core._compat.optional import is_module_available  # ruff:ignore[import-private-name]


def test_available_module() -> None:
    """An importable module is available."""
    assert is_module_available("sys")


def test_unavailable_module() -> None:
    """A missing module is unavailable."""
    assert not is_module_available("this_module_does_not_exist_xyz123")


def test_broken_module_import_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """An import error inside an installed dependency remains visible."""

    def fail_import(_module: str) -> None:
        msg = "No module named 'transitive_dependency'"
        raise ModuleNotFoundError(msg, name="transitive_dependency")

    monkeypatch.setattr(optional, "import_module", fail_import)
    with pytest.raises(ModuleNotFoundError, match="transitive_dependency"):
        is_module_available("installed_but_broken")
