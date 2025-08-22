# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the quantum computation IR."""

from __future__ import annotations

from mqt.core.ir import QuantumComputation


def test_bell_state_circuit() -> None:
    """Test the creation of a Bell state circuit."""
    qc = QuantumComputation()
    q = qc.add_qubit_register(2)
    c = qc.add_classical_register(2)

    qc.h(q[0])
    qc.cx(q[0], q[1])
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])

    qasm = qc.qasm3_str()
    expected = """
        // i 0 1
        // o 0 1
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
    """
    # Remove all whitespace from both strings before comparison
    assert "".join(qasm.split()) == "".join(expected.split())
