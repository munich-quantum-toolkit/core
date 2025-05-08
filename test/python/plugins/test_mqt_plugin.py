# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for MQT plugin.

The MQT plugin may be found here:
https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone
"""

from __future__ import annotations

import pennylane as qml
import pytest
from catalyst import pipeline
from catalyst.passes import apply_pass, apply_pass_plugin

have_mqt_plugin = True

try:
    from mqt_plugin import MQTCoreRoundTrip, getMQTPluginAbsolutePath

    plugin = getMQTPluginAbsolutePath()
except ImportError:
    have_mqt_plugin = False


@pytest.mark.skipif(not have_mqt_plugin, reason="MQT Plugin is not installed")
def test_MQT_plugin() -> None:
    """Generate MLIR for the MQT plugin. Do not execute code.
    The code execution test is in the lit test. See that test
    for more information as to why that is the case.
    """

    @apply_pass("mqt-core-round-trip")
    @qml.qnode(qml.device("lightning.qubit", wires=0))
    def qnode():
        return qml.state()

    @qml.qjit(pass_plugins={plugin}, dialect_plugins={plugin}, target="mlir")
    def module():
        return qnode()

    # It would be nice if we were able to combine lit tests with pytest
    assert "mqt-core-round-trip" in module.mlir


@pytest.mark.skipif(not have_mqt_plugin, reason="MQT Plugin is not installed")
def test_MQT_plugin_no_preregistration() -> None:
    """Generate MLIR for the MQT plugin, no need to register the
    plugin ahead of time in the qjit decorator.
    """

    @apply_pass_plugin(plugin, "mqt-core-round-trip")
    @qml.qnode(qml.device("lightning.qubit", wires=0))
    def qnode():
        return qml.state()

    @qml.qjit(target="mlir")
    def module():
        return qnode()

    # It would be nice if we were able to combine lit tests with
    # pytest
    assert "mqt-core-round-trip" in module.mlir


@pytest.mark.skipif(not have_mqt_plugin, reason="MQT Plugin is not installed")
def test_MQT_entry_point() -> None:
    """Generate MLIR for the MQT plugin via entry-point."""

    @apply_pass("mqt.mqt-core-round-trip")
    @qml.qnode(qml.device("lightning.qubit", wires=0))
    def qnode():
        return qml.state()

    @qml.qjit(target="mlir")
    def module():
        return qnode()

    # It would be nice if we were able to combine lit tests with
    # pytest
    assert "mqt-core-round-trip" in module.mlir


@pytest.mark.skipif(not have_mqt_plugin, reason="MQT Plugin is not installed")
def test_MQT_dictionary() -> None:
    """Generate MLIR for the MQT plugin via entry-point."""

    # @qjit(keep_intermediate=True)
    @pipeline({"mqt.mqt-core-round-trip": {}})
    @qml.qnode(qml.device("lightning.qubit", wires=0))
    def qnode():
        return qml.state()

    @qml.qjit(target="mlir")
    def module():
        return qnode()

    # It would be nice if we were able to combine lit tests with
    # pytest
    assert "mqt-core-round-trip" in module.mlir


@pytest.mark.skipif(not have_mqt_plugin, reason="MQT Plugin is not installed")
def test_MQT_plugin_decorator() -> None:
    """Generate MLIR for the MQT plugin."""

    @MQTCoreRoundTrip
    @qml.qnode(qml.device("lightning.qubit", wires=0))
    def qnode():
        return qml.state()

    @qml.qjit(target="mlir")
    def module():
        return qnode()

    assert "mqt-core-round-trip" in module.mlir


if __name__ == "__main__":
    pytest.main(["-x", __file__])
