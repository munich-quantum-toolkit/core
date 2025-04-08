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
from catalyst import measure


def test_pipeline_lowering():
    """
    Basic pipeline lowering on one qnode.
    """
    my_pipeline = {"mqt.quantum-to-mqtopt": {}, "mqt.mqt-core-round-trip": {}, "mqt.mqtopt-to-quantum": {}}
    num_qubits = 3

    @qml.qjit(keep_intermediate=True, pass_plugins={plugin}, dialect_plugins={plugin}, target="mlir", verbose=True)
    @pipeline(my_pipeline)
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def test_ghz_circuit():
        qml.Hadamard(wires=[0])
        for i in range(1, num_qubits):
            qml.CNOT(wires=[0, i])
        measurements = [measure(wires=i) for i in range(num_qubits)]
        return measurements

    print(test_ghz_circuit.mlir)

    test_ghz_circuit()
    flush_peephole_opted_mlir_to_iostream(test_ghz_circuit)


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


# if __name__ == "__main__":
#    pytest.main(["-x", __file__])


import pennylane as qml
import timeit
from catalyst import measure

my_pipeline = {"mqt.quantum-to-mqtopt": {}, "mqt.mqt-core-round-trip": {}, "mqt.mqtopt-to-quantum": {}}


def _compile(num_qubits: int):
    custom_pipeline = [("run_only_plugin", ["builtin.module(apply-transform-sequence)"])]

    @qml.qjit(target="mlir", pipelines=custom_pipeline)
    def foo():
        dev = qml.device("lightning.qubit", wires=num_qubits)

        @pipeline(my_pipeline)
        @qml.qnode(dev)
        def workflow():
            qml.Hadamard(wires=[0])
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            meas = [measure(wires=i) for i in range(num_qubits)]
            return meas

        return workflow()

    _ = foo.mlir
    return


for n in range(2, 5000, 50):
    duration = timeit.timeit(lambda: _compile(n), number=10)
    print(f"n={n}: {duration:.6f} seconds")
