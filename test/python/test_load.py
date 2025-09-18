# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the circuit IO functionality."""

from __future__ import annotations

from pathlib import Path

from mqt.core import load
from mqt.core.ir import QuantumComputation


def test_loading_quantum_computation() -> None:
    """Test that directly loading a ``QuantumComputation`` works."""
    qc = QuantumComputation(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.mcx({0, 1}, 2)
    qc.measure(range(3), range(3))

    qc_loaded = load(qc)
    print(qc_loaded)

    # check that the same object is returned
    assert qc is qc_loaded


def test_loading_file() -> None:
    """Test whether importing a simple QASM file works."""
    qasm = "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\nmeasure q -> c;\n"
    with Path("test_1.qasm").open("w", encoding="utf-8") as f:
        f.write(qasm)

    # load the file
    qc = load("test_1.qasm")
    print(qc)

    # check the result
    assert isinstance(qc, QuantumComputation)
    qc_qasm = qc.qasm2_str()

    assert qasm in qc_qasm

    # remove the file
    Path("test_1.qasm").unlink()


def test_loading_file_from_path() -> None:
    """Test whether importing a simple QASM file works."""
    qasm = "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\nmeasure q -> c;\n"
    path = Path("test_2.qasm")
    with path.open("w", encoding="utf-8") as f:
        f.write(qasm)

    # load the file
    qc = load(path)
    print(qc)

    # check the result
    assert isinstance(qc, QuantumComputation)
    qc_qasm = qc.qasm2_str()

    assert qasm in qc_qasm

    # remove the file
    path.unlink()


def test_loading_nonexistent_file() -> None:
    """Test whether trying to load a non-existent file raises an error.

    Raises:
        AssertionError: If no error is raised.
    """
    try:
        load("nonexistent.qasm")
    except FileNotFoundError:
        pass
    else:
        msg = "No error was raised when trying to load a non-existent file."
        raise AssertionError(msg)


def test_loading_qiskit_circuit() -> None:
    """Test whether importing a Qiskit circuit works."""
    from qiskit import QuantumCircuit
    from qiskit.qasm2 import dumps

    qiskit_circuit = QuantumCircuit(2, 2)
    qiskit_circuit.h(0)
    qiskit_circuit.cx(0, 1)
    qiskit_circuit.measure(range(2), range(2))
    qasm = dumps(qiskit_circuit)

    # load the circuit
    qc = load(qiskit_circuit)
    print(qc)

    # check the result
    assert isinstance(qc, QuantumComputation)
    qc_qasm = qc.qasm2_str()

    # remove any whitespace from both QASM strings and check for equality
    assert "".join(qasm.split()) in "".join(qc_qasm.split())


def test_loading_qasm2_string() -> None:
    """Test whether importing a QASM2 string works."""
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\nmeasure q -> c;\n'

    # load the circuit
    qc = load(qasm)
    print(qc)

    # check the result
    assert isinstance(qc, QuantumComputation)
    qc_qasm = qc.qasm2_str()

    assert qasm in qc_qasm


def test_loading_qasm3_string() -> None:
    """Test whether importing a QASM3 string works."""
    qasm = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];\nc = measure q;\n'

    # load the circuit
    qc = load(qasm)
    print(qc)

    # check the result
    assert isinstance(qc, QuantumComputation)
    qc_qasm = qc.qasm3_str()

    assert qasm in qc_qasm
