# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test optional plugin dependency detection."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


@pytest.mark.parametrize(
    ("dependency", "availability_flag"), [("pennylane", "HAS_PENNYLANE"), ("qiskit", "HAS_QISKIT")]
)
def test_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
    availability_flag: str,
) -> None:
    """Treat a missing top-level dependency as an unavailable plugin."""
    plugin = importlib.import_module(f"mqt.core.plugins.{dependency}")
    original_import_module = importlib.import_module

    def import_without_dependency(name: str, package: str | None = None) -> ModuleType:
        if name == dependency:
            raise ModuleNotFoundError(name=dependency)
        return original_import_module(name, package)

    try:
        with monkeypatch.context() as context:
            context.setattr(importlib, "import_module", import_without_dependency)
            reloaded_plugin = importlib.reload(plugin)
            assert getattr(reloaded_plugin, availability_flag) is False
    finally:
        importlib.reload(plugin)


@pytest.mark.parametrize("dependency", ["pennylane", "qiskit"])
def test_nested_import_failure_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
) -> None:
    """Propagate a missing dependency imported by an installed plugin."""
    plugin = importlib.import_module(f"mqt.core.plugins.{dependency}")
    original_import_module = importlib.import_module

    def import_with_nested_failure(name: str, package: str | None = None) -> ModuleType:
        if name == dependency:
            raise ModuleNotFoundError(name="nested_dependency")
        return original_import_module(name, package)

    try:
        with monkeypatch.context() as context:
            context.setattr(importlib, "import_module", import_with_nested_failure)
            with pytest.raises(ModuleNotFoundError) as exc_info:
                importlib.reload(plugin)
            assert exc_info.value.name == "nested_dependency"
    finally:
        importlib.reload(plugin)
