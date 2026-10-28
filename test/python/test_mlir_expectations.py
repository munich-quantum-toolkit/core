# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Program-level expectation evaluation without exposing simulator state."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.core.mlir import JeffProgram, OpenQASMProgram, QCOProgram, QCProgram, expectation_values


@pytest.mark.parametrize("kind", ["source", "openqasm", "qc", "qco", "jeff"])
def test_expectations_preserve_program_and_terminal_coherence(kind: str) -> None:
    """Shared input adapters evaluate the uncollapsed state without consuming it."""
    source = 'OPENQASM 3; include "stdgates.inc"; qubit[3] q; h q[0]; s q[0]; x q[2]; bit c = measure q[0];'
    program: str | QCProgram | QCOProgram | JeffProgram | OpenQASMProgram = source
    if kind != "source":
        program = QCProgram.from_openqasm_str(source)
        if kind == "openqasm":
            program = program.to_openqasm3()
        elif kind in {"qco", "jeff"}:
            program = program.to_qco()
            if kind == "jeff":
                program = program.to_jeff()
    original = program.ir if isinstance(program, QCOProgram) else None
    observables = [{"IIY": 1.0}, {"ZII": 1.0}, {"IIY": 0.5, "ZII": 0.25}, {}]
    expected = [1, -1, 0.25, 0]
    np.testing.assert_allclose(expectation_values(program, observables), expected, atol=1e-12)
    np.testing.assert_allclose(expectation_values(program, observables), expected, atol=1e-12)
    if isinstance(program, QCOProgram):
        assert program.ir == original
        np.testing.assert_allclose(program.expectation_values(observables), expected, atol=1e-12)


def test_expectations_bind_qasm_parameters() -> None:
    """Existing named parameter binding supplies the native preparation."""
    program = QCProgram.from_openqasm_str(
        'OPENQASM 3.0; include "stdgates.inc"; input float[64] theta; qubit q; ry(theta) q;'
    ).to_qco()
    with pytest.raises(ValueError, match=r"argument|parameter|binding"):
        expectation_values(program, [{"Z": 1.0}])
    program.bind_parameters({"theta": np.pi / 3})
    np.testing.assert_allclose(expectation_values(program, [{"Z": 1.0}]), [0.5], atol=1e-12)


@pytest.mark.parametrize("observable", [{"XX": 1.0}, {"A": 1.0}, {"X": np.inf}, {"X": np.nan}])
def test_expectations_reject_invalid_observables(observable: dict[str, float]) -> None:
    """Native validation also protects callers outside Qiskit."""
    program = QCProgram.from_openqasm_str("OPENQASM 3.0; qubit q;").to_qco()
    with pytest.raises(ValueError, match=r"Pauli|observable|coefficient"):
        expectation_values(program, [observable])
    assert expectation_values(program, [{"Z": 1.0}]) == [1.0]


def test_expectations_report_overflow() -> None:
    """Finite inputs cannot produce a successful infinite expectation."""
    program = QCProgram.from_openqasm_str("OPENQASM 3.0; qubit q;").to_qco()
    coefficient = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="finite"):
        expectation_values(program, [{"I": coefficient, "Z": coefficient}])
    assert expectation_values(program, [{"Z": 1.0}]) == [1.0]


def test_expectations_scalar_and_empty_observables() -> None:
    """Zero-qubit states and empty observable batches retain their dimensions."""
    scalar = "OPENQASM 3;"
    assert expectation_values(scalar, [{"": 2.5}, {}]) == [2.5, 0.0]
    assert expectation_values(scalar, []) == []
