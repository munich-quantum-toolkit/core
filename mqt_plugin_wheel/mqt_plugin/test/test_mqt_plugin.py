# Copyright (c) 2025 Chair for Design Automation, TUM
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


from catalyst import pipeline
from catalyst.debug import get_compilation_stage
from catalyst.passes import apply_pass, cancel_inverses, merge_rotations


def print_attr(f, attr, *args, aot: bool = False, **kwargs):
    """Print function attribute"""
    name = f"TEST {f.__name__}"
    print("\n" + "-" * len(name))
    print(f"{name}\n")
    res = None
    if not aot:
        res = f(*args, **kwargs)
    print(getattr(f, attr))
    return res


def print_mlir(f, *args, **kwargs):
    """Print mlir code of a function"""
    return print_attr(f, "mlir", *args, **kwargs)


def flush_peephole_opted_mlir_to_iostream(QJIT):
    """
    The QJIT compiler does not offer a direct interface to access an intermediate mlir in the pipeline.
    The `QJIT.mlir` is the mlir before any passes are run, i.e. the "0_<qnode_name>.mlir".
    Since the QUANTUM_COMPILATION_PASS is located in the middle of the pipeline, we need
    to retrieve it with keep_intermediate=True and manually access the "2_QuantumCompilationPass.mlir".
    Then we delete the kept intermediates to avoid pollution of the workspace
    """
    print(get_compilation_stage(QJIT, "QuantumCompilationPass"))


#
# pipeline
#


def test_pipeline_lowering():
    """
    Basic pipeline lowering on one qnode.
    """
    my_pipeline = {"mqt.mqt-core-round-trip": {}}

    @qml.qjit(keep_intermediate=True)
    @pipeline(my_pipeline)
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def test_pipeline_lowering_workflow(x):
        qml.Hadamard(wires=[1])
        return

    # CHECK: transform.named_sequence @__transform_main
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "remove-chained-self-inverse" to {{%.+}}
    # CHECK-NEXT: {{%.+}} = transform.apply_registered_pass "merge-rotations" to {{%.+}}
    # CHECK-NEXT: transform.yield
    print_mlir(test_pipeline_lowering_workflow, 1.2)

    # CHECK: {{%.+}} = call @test_pipeline_lowering_workflow_0(
    # CHECK: func.func public @test_pipeline_lowering_workflow_0(
    # CHECK: {{%.+}} = quantum.custom "RX"({{%.+}}) {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    # CHECK-NOT: {{%.+}} = quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    test_pipeline_lowering_workflow(42.42)
    flush_peephole_opted_mlir_to_iostream(test_pipeline_lowering_workflow)


test_pipeline_lowering()


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
