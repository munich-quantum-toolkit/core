# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for MQT plugin execution with PennyLane and Catalyst.

These tests check that the MQT plugin is correctly installed and
can be executed with PennyLane.
"""

from __future__ import annotations

import catalyst
import pennylane as qml
import pytest
from catalyst.passes import apply_pass

from mqt.core.plugins.catalyst import get_catalyst_plugin_abs_path

plugin_available: bool = True
try:
    plugin_path: str = str(get_catalyst_plugin_abs_path())
except ImportError:
    plugin_available = False


@pytest.mark.skipif(not plugin_available, reason="MQT Plugin is not installed")
def test_mqtopt_conversion() -> None:
    """Execute the conversion passes to and from MQTOpt dialect."""

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def circuit() -> None:
        qml.Hadamard(wires=[0])
        qml.CNOT(wires=[0, 1])
        catalyst.measure(0)
        catalyst.measure(1)

    @qml.qjit(target="mlir", autograph=True)
    def module() -> None:
        return circuit()

    # This will execute the pass and return the final MLIR
    assert module.mlir_opt


@pytest.mark.skipif(not plugin_available, reason="MQT Plugin is not installed")
def test_mqtopt_roundtrip() -> None:
    """Execute the full roundtrip including MQT Core IR.

    Executes the conversion passes to and from MQTOpt dialect AND
    the roundtrip through MQT Core IR.
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.mqt-core-round-trip")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def circuit() -> None:
        qml.Hadamard(wires=[0])
        qml.CNOT(wires=[0, 1])
        catalyst.measure(0)
        catalyst.measure(1)

    @qml.qjit(target="mlir", autograph=True)
    def module() -> None:
        return circuit()

    # This will execute the pass and return the final MLIR
    assert module.mlir_opt
