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

from typing import TYPE_CHECKING

import pennylane as qml
import pytest
from catalyst.passes import apply_pass
from mqt.catalyst import get_catalyst_plugin_abs_path

if TYPE_CHECKING:
    from pennylane.measurements.state import StateMP

plugin_available: bool = True
try:
    plugin_path: str = str(get_catalyst_plugin_abs_path())
except ImportError:
    plugin_available = False


@pytest.mark.skipif(not plugin_available, reason="MQT Plugin is not installed")
def test_mqt_plugin() -> None:
    """Generate MLIR for the MQT plugin.

    Execute the full pipeline, including the MQT pass.
    """

    @apply_pass("mqt-core-round-trip")
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def circuit() -> StateMP:
        qml.Hadamard(wires=[0])
        qml.CNOT(wires=[0, 1])
        return qml.state()

    @qml.qjit(pass_plugins={plugin_path}, dialect_plugins={plugin_path}, target="mlir")
    def module() -> StateMP:
        return circuit()

    # This will execute the pass and return the final MLIR
    assert module.mlir_opt
