# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the multiplexer benchmark."""

from __future__ import annotations

import json

import pytest

from mqt.core.bench import multiplexer

from .utils import assert_generates


def _make_benchmark() -> multiplexer.Multiplexer:
    return multiplexer.Multiplexer(multiplexer.Options(qubits=3))


def test_multiplexer_reference() -> None:
    """Expose the output and reference distribution."""
    benchmark = _make_benchmark()
    assert benchmark.output.name == "result"
    assert benchmark.output.width == 3
    assert benchmark.probability("000") == pytest.approx(0.25)
    assert benchmark.probability("001") == 0


def test_multiplexer_evaluation() -> None:
    """Evaluate counts against the multiplexer reference distribution."""
    benchmark = _make_benchmark()
    evaluation = benchmark.evaluate({"000": 10})
    assert evaluation.total_variation_distance == pytest.approx(0.75)
    assert evaluation.squared_hellinger_fidelity == pytest.approx(0.25)
    assert evaluation.success_probability is None


def test_multiplexer_json_roundtrip() -> None:
    """Preserve the benchmark identity through both JSON representations."""
    benchmark = _make_benchmark()
    assert json.loads(benchmark.instance_specification_json)["parameters"] == {"qubits": 3}

    instance_copy = multiplexer.Multiplexer.from_instance_specification_json(benchmark.instance_specification_json)
    manifest_copy = multiplexer.Multiplexer.from_manifest_json(benchmark.manifest_json)
    assert instance_copy.case_id == manifest_copy.case_id == benchmark.case_id


def test_multiplexer_generation() -> None:
    """Generate a multiplexer program through the Python binding."""
    assert_generates(_make_benchmark().generate())
