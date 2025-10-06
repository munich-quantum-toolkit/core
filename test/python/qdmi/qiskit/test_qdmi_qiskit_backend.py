# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QiskitBackend."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

from mqt.core.qdmi.qiskit import (
    QiskitBackend,
    UnsupportedOperationError,
    clear_operation_translators,
    register_operation_translator,
)

if TYPE_CHECKING:
    from mqt.core.qdmi.qiskit import (
        InstructionContext,
        ProgramInstruction,
    )

_qiskit_present = importlib.util.find_spec("qiskit") is not None

pytestmark = pytest.mark.skipif(not _qiskit_present, reason="qiskit not installed")

if _qiskit_present:
    from qiskit import QuantumCircuit


def setup_module() -> None:  # noqa: D103
    clear_operation_translators(keep_defaults=True)
    # Register minimal translators used in tests

    def _cz(ctx: InstructionContext) -> list[ProgramInstruction]:
        from mqt.core.qdmi.qiskit import ProgramInstruction

        return [ProgramInstruction(name="cz", qubits=ctx.qubits)]

    register_operation_translator("cz", _cz, overwrite=True)


def test_backend_instantiation() -> None:
    """Backend exposes capabilities hash and target qubit count."""
    backend = QiskitBackend()
    assert backend.capabilities_hash
    assert backend.target.num_qubits > 0


def test_single_circuit_run_counts() -> None:
    """Running a circuit yields deterministic all-zero counts with specified shots."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    backend = QiskitBackend()
    job = backend.run(qc, shots=256)
    counts = job.get_counts()
    assert list(counts.values()) == [256]
    assert list(counts.keys()) == ["00"]


def test_unsupported_operation() -> None:
    """Unsupported operation raises the expected error type."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)  # 'x' translator not registered
    qc.measure(0, 0)
    backend = QiskitBackend()
    with pytest.raises(UnsupportedOperationError):
        backend.run(qc)
