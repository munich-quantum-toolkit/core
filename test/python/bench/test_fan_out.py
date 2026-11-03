# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the quantum fan-out benchmark."""

from __future__ import annotations

import json

import pytest

from mqt.core.bench import fan_out

from .utils import assert_generates


def _make_benchmark() -> fan_out.FanOut:
    return fan_out.FanOut(fan_out.Options(controls=3))


def test_fan_out_reference() -> None:
    """Expose the output and reference distribution."""
    benchmark = _make_benchmark()
    assert benchmark.options.controls == 3
    assert benchmark.output.name == "result"
    assert benchmark.output.width == 4
    assert benchmark.probability("0000") == pytest.approx(0.125)
    assert benchmark.probability("0001") == 0
    assert benchmark.probability("0010") == pytest.approx(0.0625)
    assert benchmark.probability("0111") == pytest.approx(0.125)


def test_fan_out_evaluation() -> None:
    """Evaluate counts against the fan-out reference distribution."""
    evaluation = _make_benchmark().evaluate({"0000": 10})
    assert evaluation.total_variation_distance == pytest.approx(0.875)
    assert evaluation.squared_hellinger_fidelity == pytest.approx(0.125)
    assert evaluation.success_probability is None


def test_fan_out_json_roundtrip() -> None:
    """Preserve the benchmark identity through both JSON representations."""
    benchmark = _make_benchmark()
    assert json.loads(benchmark.instance_specification_json)["parameters"] == {"controls": 3}

    instance_copy = fan_out.FanOut.from_instance_specification_json(benchmark.instance_specification_json)
    manifest_copy = fan_out.FanOut.from_manifest_json(benchmark.manifest_json)
    assert instance_copy.case_id == manifest_copy.case_id == benchmark.case_id


def test_fan_out_generation() -> None:
    """Generate a fan-out program through the Python binding."""
    assert_generates(_make_benchmark().generate())
