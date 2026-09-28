# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""W-state generation and evaluation through the existing benchmark API."""

from __future__ import annotations

import json

import pytest

from mqt.core import mlir
from mqt.core.bench import w_state


def test_w_state_reference_and_json() -> None:
    """Expose the new family's parameters, ideal probabilities, and manifest."""
    benchmark = w_state.WState(w_state.Options(qubits=3))
    assert benchmark.options.qubits == 3
    assert benchmark.output.name == "result"
    assert benchmark.output.width == 3
    assert benchmark.probability("010") == pytest.approx(1 / 3)
    assert benchmark.probability("111") == 0
    counts = {"001": 10, "010": 10, "100": 10}
    evaluation = benchmark.evaluate(counts)
    assert evaluation.total_variation_distance == pytest.approx(0)
    assert evaluation.squared_hellinger_fidelity == pytest.approx(1)
    manifest = json.loads(benchmark.manifest_json)
    assert manifest["reference"]["model"] == "w_state"
    assert w_state.WState.from_manifest_json(benchmark.manifest_json).case_id == benchmark.case_id
    assert (
        w_state.WState.from_instance_specification_json(benchmark.instance_specification_json).manifest_json
        == benchmark.manifest_json
    )
    with pytest.raises(ValueError, match="qubits must be positive"):
        w_state.WState(w_state.Options(qubits=0))


def test_w_state_generation_and_sampling() -> None:
    """Sample W states through QC and serialized jeff programs."""
    assert mlir.sample(w_state.WState(w_state.Options(qubits=1)).generate(), shots=32) == {"1": 32}
    benchmark = w_state.WState(w_state.Options(qubits=3))
    qc = benchmark.generate()
    jeff = qc.to_qco(copy=True).to_jeff()
    for program in (qc, mlir.JeffProgram.from_bytes(jeff.to_bytes())):
        counts = mlir.sample(program, shots=4096, seed=17)
        assert set(counts) == {"001", "010", "100"}
        assert sum(counts.values()) == 4096
        assert benchmark.evaluate(counts).total_variation_distance < 0.03
