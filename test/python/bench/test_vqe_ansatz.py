# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the fixed-repetition VQE ansatz benchmark."""

from __future__ import annotations

import json

import numpy as np
import pytest
from qiskit.circuit.library import real_amplitudes
from qiskit.quantum_info import Statevector

from mqt.core.bench import vqe_ansatz
from mqt.core.mlir import sample


def test_vqe_ansatz_json_roundtrip() -> None:
    """Expose resolved options and the absence of a built-in reference."""
    benchmark = vqe_ansatz.VQEAnsatz(vqe_ansatz.Options(qubits=3))
    assert benchmark.options.qubits == 3
    assert benchmark.options.depth == 1
    assert benchmark.num_parameters == 6
    assert benchmark.output.name == "result"
    assert benchmark.output.width == 3
    assert json.loads(benchmark.manifest_json)["reference"] == {"kind": "none"}
    for restored in (
        vqe_ansatz.VQEAnsatz.from_instance_specification_json(benchmark.instance_specification_json),
        vqe_ansatz.VQEAnsatz.from_manifest_json(benchmark.manifest_json),
    ):
        assert restored.case_id == benchmark.case_id


@pytest.mark.parametrize(("qubits", "depth"), [(1, 2), (3, 1), (4, 3)])
def test_vqe_ansatz_matches_real_amplitudes(qubits: int, depth: int) -> None:
    """Match Qiskit's linear ansatz, including signs and final rotations."""
    benchmark = vqe_ansatz.VQEAnsatz(vqe_ansatz.Options(qubits=qubits, depth=depth))
    angles = np.linspace(-1.3, 2.1, benchmark.num_parameters)
    program = benchmark.generate()
    program.bind_parameters({"thetas": angles})
    qco = program.to_qco()
    qco.unroll_quantum_loops()
    circuit = qco.to_qiskit()
    circuit.remove_final_measurements()
    reference = real_amplitudes(qubits, entanglement="linear", reps=depth).assign_parameters(angles)
    assert Statevector.from_instruction(circuit).equiv(Statevector.from_instruction(reference))


def test_vqe_ansatz_reuses_template_for_sampling() -> None:
    """Bind independent angle arrays without changing the template."""
    benchmark = vqe_ansatz.VQEAnsatz(vqe_ansatz.Options(qubits=3))
    template = benchmark.generate()
    program = template.copy()
    program.bind_parameters({"thetas": [np.pi / 2, 0, 0, 0, 0, 0]})
    counts = sample(program, shots=4096, seed=17)
    assert set(counts) == {"000", "111"}
    assert counts["111"] / 4096 == pytest.approx(0.5, abs=0.03)
    assert template.parameters == ["thetas"]
    program = template.copy()
    program.bind_parameters({"thetas": np.zeros(benchmark.num_parameters)})
    assert sample(program, shots=16, seed=17) == {"000": 16}


def test_vqe_ansatz_errors() -> None:
    """Propagate invalid options and angle-array lengths through Python."""
    with pytest.raises(ValueError, match="qubits"):
        vqe_ansatz.VQEAnsatz(vqe_ansatz.Options(qubits=0))
    with pytest.raises(ValueError, match="depth"):
        vqe_ansatz.VQEAnsatz(vqe_ansatz.Options(qubits=3, depth=0))
    program = vqe_ansatz.VQEAnsatz(vqe_ansatz.Options(qubits=3)).generate()
    with pytest.raises(ValueError, match="requires 6 elements"):
        program.bind_parameters({"thetas": [0.0]})
