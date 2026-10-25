# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the QAOA benchmark for MaxCut."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mqt.core.bench import qaoa
from mqt.core.mlir import sample


def test_qaoa_reference_and_json() -> None:
    """Expose the default depth, cut evaluation, and verification manifest."""
    benchmark = qaoa.QAOA(qaoa.Options(qubits=4))
    assert benchmark.options.qubits == 4
    assert benchmark.options.depth == 1
    assert benchmark.output.name == "result"
    assert benchmark.output.width == 4
    assert benchmark.cut_size("0101") == 4
    evaluation = benchmark.evaluate({"0101": 3, "0000": 1})
    assert evaluation.shots == 4
    assert evaluation.mean_cut == pytest.approx(3)
    assert evaluation.approximation_ratio == pytest.approx(0.75)
    assert json.loads(benchmark.manifest_json)["reference"]["kind"] == "verification"
    for restored in (
        qaoa.QAOA.from_instance_specification_json(benchmark.instance_specification_json),
        qaoa.QAOA.from_manifest_json(benchmark.manifest_json),
    ):
        assert restored.case_id == benchmark.case_id


def test_qaoa_binding_and_sampling() -> None:
    """Bind one template to distinct angles and score their cuts."""
    benchmark = qaoa.QAOA(qaoa.Options(qubits=4, depth=2))
    template = benchmark.generate()
    program = template.copy()
    program.bind_parameters({"gammas": [np.pi / 4, 0.0]})
    assert program.parameters == ["betas"]
    program.bind_parameters({"betas": [np.pi / 8, 0.0]})
    assert not program.parameters
    counts = sample(program, shots=4096, seed=17)
    assert benchmark.evaluate(counts).mean_cut == pytest.approx(3.0, abs=0.1)

    assert template.parameters == ["gammas", "betas"]
    template.bind_parameters({"gammas": np.zeros(2), "betas": np.zeros(2)})
    counts = sample(template, shots=4096, seed=17)
    assert benchmark.evaluate(counts).mean_cut == pytest.approx(2.0, abs=0.1)


def test_qaoa_errors() -> None:
    """Propagate invalid parameters and outcomes through the binding."""
    with pytest.raises(ValueError, match="depth"):
        qaoa.QAOA(qaoa.Options(qubits=4, depth=0))
    with pytest.raises(ValueError, match="width"):
        qaoa.QAOA(qaoa.Options(qubits=4)).cut_size("010")
