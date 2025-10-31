# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for MQT plugin setup with PennyLane and Catalyst.

These tests only check that the MQT plugin is correctly installed and
can be used in various ways with PennyLane (they do NOT execute any pass).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pennylane as qml
from catalyst import pipeline
from catalyst.passes import apply_pass, apply_pass_plugin

from mqt.core.plugins.catalyst import get_catalyst_plugin_abs_path

if TYPE_CHECKING:
    from pennylane.measurements.state import StateMP


def test_mqt_plugin() -> None:
    """Generate MLIR for the MQT plugin.

    Does not execute the pass.
    """
    plugin_path = str(get_catalyst_plugin_abs_path())

    @apply_pass("mqt-core-round-trip")
    @qml.qnode(qml.device("null.qubit", wires=0))
    def qnode() -> StateMP:
        return qml.state()

    @qml.qjit(pass_plugins={plugin_path}, dialect_plugins={plugin_path}, target="mlir")
    def module() -> StateMP:
        return qnode()

    assert "mqt-core-round-trip" in module.mlir


def test_mqt_plugin_no_preregistration() -> None:
    """Generate MLIR for the MQT plugin.

    No need to register the plugin ahead of time in the qjit decorator.
    """
    plugin_path = str(get_catalyst_plugin_abs_path())

    @apply_pass_plugin(plugin_path, "mqt-core-round-trip")
    @qml.qnode(qml.device("null.qubit", wires=0))
    def qnode() -> StateMP:
        return qml.state()

    @qml.qjit(target="mlir")
    def module() -> StateMP:
        return qnode()

    assert "mqt-core-round-trip" in module.mlir


def test_mqt_entry_point() -> None:
    """Generate MLIR for the MQT plugin via entry-point."""

    @apply_pass("mqt.mqt-core-round-trip")
    @qml.qnode(qml.device("null.qubit", wires=0))
    def qnode() -> StateMP:
        return qml.state()

    @qml.qjit(target="mlir")
    def module() -> StateMP:
        return qnode()

    assert "mqt-core-round-trip" in module.mlir


def test_mqt_dictionary() -> None:
    """Generate MLIR for the MQT plugin via entry-point."""

    @pipeline({"mqt.mqt-core-round-trip": {}})
    @qml.qnode(qml.device("null.qubit", wires=0))
    def qnode() -> StateMP:
        return qml.state()

    @qml.qjit(target="mlir")
    def module() -> StateMP:
        return qnode()

    assert "mqt-core-round-trip" in module.mlir
