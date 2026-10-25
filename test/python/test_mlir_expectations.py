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

from mqt.core.mlir import QCProgram


def test_expectations_preserve_program_and_terminal_coherence() -> None:
    """Terminal readout does not collapse the expectation state."""
    program = QCProgram.from_openqasm_str(
        'OPENQASM 3.0; include "stdgates.inc"; qubit[3] q; h q[0]; s q[0]; x q[2]; bit[1] c = measure q[0];'
    ).to_qco()
    original = program.ir
    observables = [{"IIY": 1.0}, {"ZII": 1.0}, {"IIY": 0.5, "ZII": 0.25}, {}]
    for _ in range(2):
        np.testing.assert_allclose(program.expectation_values(observables), [1, -1, 0.25, 0], atol=1e-12)
        assert program.ir == original
    assert program.expectation_values([]) == []


def test_expectations_bind_qasm_parameters() -> None:
    """Existing named parameter binding supplies the native preparation."""
    program = QCProgram.from_openqasm_str(
        'OPENQASM 3.0; include "stdgates.inc"; input float[64] theta; qubit q; ry(theta) q;'
    ).to_qco()
    with pytest.raises(ValueError, match=r"argument|parameter|binding"):
        program.expectation_values([{"Z": 1.0}])
    program.bind_parameters({"theta": np.pi / 3})
    np.testing.assert_allclose(program.expectation_values([{"Z": 1.0}]), [0.5], atol=1e-12)


@pytest.mark.parametrize("observable", [{"XX": 1.0}, {"A": 1.0}, {"X": np.inf}, {"X": np.nan}])
def test_expectations_reject_invalid_observables(observable: dict[str, float]) -> None:
    """Native validation also protects callers outside Qiskit."""
    program = QCProgram.from_openqasm_str("OPENQASM 3.0; qubit q;").to_qco()
    with pytest.raises(ValueError, match=r"Pauli|observable|coefficient"):
        program.expectation_values([observable])
    assert program.expectation_values([{"Z": 1.0}]) == [1.0]


def test_expectations_reject_reset() -> None:
    """A trajectory is not returned as an exact ensemble expectation."""
    program = QCProgram.from_openqasm_str("OPENQASM 3.0; qubit q; reset q;").to_qco()
    with pytest.raises(ValueError, match=r"statevector|reset|terminal"):
        program.expectation_values([{"Z": 1.0}])


def test_expectations_report_overflow() -> None:
    """Finite inputs cannot produce a successful infinite expectation."""
    program = QCProgram.from_openqasm_str("OPENQASM 3.0; qubit q;").to_qco()
    coefficient = np.finfo(np.float64).max
    with pytest.raises(OverflowError, match="finite"):
        program.expectation_values([{"I": coefficient, "Z": coefficient}])
    assert program.expectation_values([{"Z": 1.0}]) == [1.0]


def test_expectations_scalar_and_compact_states() -> None:
    """Scalar states and wide compact states need no dense export."""
    scalar = QCProgram.from_openqasm_str("OPENQASM 3.0;").to_qco()
    assert scalar.expectation_values([{"": 2.5}, {}]) == [2.5, 0.0]
    program = QCProgram.from_openqasm_str(
        'OPENQASM 3.0; include "stdgates.inc"; qubit[64] q; h q[0]; '
        + " ".join(f"cx q[0], q[{index}];" for index in range(1, 64))
    ).to_qco()
    np.testing.assert_allclose(
        program.expectation_values([{"X" * 64: 1.0}, {"Z" + "I" * 62 + "Z": 1.0}]), [1, 1], atol=1e-12
    )
