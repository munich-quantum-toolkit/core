# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import pennylane as qml
from pennylane.tape import QuantumTape


# Define GHZ circuit
def ghz_circuit() -> None:
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])


# Prepare conversion
with QuantumTape() as tape:
    ghz_circuit()

# Convert to OpenQASM
qasm_str = tape.to_openqasm()

# ... continue with MQT mapping ...

# Convert OpenQASM back to PennyLane
# mapped_ghz = qml.from_qasm(mapped_qasm)

############################################################
############################################################

import mqt.qmap.pyqmap as qmap
from mqt.core import QuantumComputation

# Convert OpenQASM to MQT Core IR
mqt_qc = QuantumComputation.from_qasm_str(qasm_str)

n_qubits = 3
cmap = {(0, 1), (1, 0), (1, 2), (2, 1)}
arch = qmap.Architecture(n_qubits, cmap)
config = qmap.Configuration()

# Map circuit using QMAP
mapped_qc, _ = qmap.map(mqt_qc, arch, config)

# Convert mapped circuit to OpenQASM
mapped_qasm = QuantumComputation.qasm2_str(mqt_qc)

############################################################
############################################################
mapped_circuit = qml.from_qasm(mapped_qasm)


############################################################
############################################################

import pennylane as qml
from catalyst import measure
from mqt_plugin import QMAP, plugin


@qml.qjit(pass_plugins={plugin}, dialect_plugins={plugin})
@QMAP({"cMap": [(0, 1), (1, 0), (1, 2), (2, 1)]})
@qml.qnode(qml.device("lightning.qubit", wires=3))
def test_ghz_circuit():
    qml.Hadamard(wires=[0])
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])
    return [measure(i) for i in range(3)]


test_ghz_circuit.mlir_opt

###
# %r0 = qntm.alloc(0)      : !qntm.reg
# %q0 = qntm.extract %r0[0]: !qntm.reg->!qntm.bit
# %q1 = qntm.extract %r0[1]: !qntm.reg->!qntm.bit
# %q2 = qntm.extract %r0[2]: !qntm.reg->!qntm.bit
#
# //...
#
# %q3   = qntm.custom"H"() %q0          : !qntm.bit
# %q4:2 = qntm.custom"CNOT"() %q3, %q1  : !qntm.bit,!qntm.bit
# %q5:2 = qntm.custom"CNOT"() %q4#0, %q2: !qntm.bit,!qntm.bit
#
# //...
#
# %r2 = qntm.insert %r0[0], %q5#0: !qntm.reg,!qntm.bit->!qntm.reg
# %r3 = qntm.insert %r6[1], %q4#1: !qntm.reg,!qntm.bit->!qntm.reg
# %r4 = qntm.insert %r7[2], %q5#1: !qntm.reg,!qntm.bit->!qntm.reg
# qntm.dealloc %r4               : !qntm.reg
#
# %r1, %q0 = mqtopt.extractQubit(%r0, %idx0) :
# (!mqtopt.QubitRegister, i64) ->
# (!mqtopt.QubitRegister, !mqtopt.Qubit)
#
# %r2, %q1 = mqtopt.extractQubit(%r1, %idx1) :
# (!mqtopt.QubitRegister, i64) ->
# (!mqtopt.QubitRegister, !mqtopt.Qubit)
#
# // ...
#
# %q3 = mqtopt.H() %q0 : !mqtopt.Qubit
# %q4:2 = mqtopt.x() %q1 ctrl %q3 : !mqtopt.Qubit, !mqtopt.Qubit
