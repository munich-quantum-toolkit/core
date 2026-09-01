# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the public Python package surface."""

from __future__ import annotations

import importlib

import pytest

import mqt.core


@pytest.mark.parametrize(
    "module",
    [
        "mqt.core.ir",
        "mqt.core.load",
        "mqt.core.plugins.qiskit.mqt_to_qiskit",
        "mqt.core.plugins.qiskit.qiskit_to_mqt",
    ],
)
def test_legacy_circuit_modules_are_absent(module: str) -> None:
    """Omit removed legacy modules."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_load_is_not_exported_from_mqt_core() -> None:
    """Omit load from top-level exports."""
    assert not hasattr(mqt.core, "load")


def test_classic_qiskit_converters_are_not_exported() -> None:
    """Omit classic Qiskit converters from plugin exports."""
    plugin = importlib.import_module("mqt.core.plugins.qiskit")
    assert not hasattr(plugin, "mqt_to_qiskit")
    assert not hasattr(plugin, "qiskit_to_mqt")
