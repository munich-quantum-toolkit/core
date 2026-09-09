# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the repeat-until-success benchmark."""

from __future__ import annotations

import json
import math

import pytest

from mqt.core.bench import repeat_until_success

from .utils import assert_generates


def test_repeat_until_success_reference() -> None:
    """Expose the output and phase-sensitive reference distribution."""
    benchmark = repeat_until_success.RepeatUntilSuccess()
    assert benchmark.output.name == "result"
    assert benchmark.output.width == 1
    assert benchmark.probability("0") == pytest.approx(0.5 + math.sqrt(2) / 3)
    assert benchmark.probability("1") == pytest.approx(0.5 - math.sqrt(2) / 3)


def test_repeat_until_success_evaluation() -> None:
    """Evaluate deterministic counts against the reference distribution."""
    evaluation = repeat_until_success.RepeatUntilSuccess().evaluate({"0": 10})
    assert evaluation.total_variation_distance == pytest.approx(0.5 - math.sqrt(2) / 3)
    assert evaluation.squared_hellinger_fidelity == pytest.approx(0.5 + math.sqrt(2) / 3)
    assert evaluation.success_probability is None


def test_repeat_until_success_json_roundtrip() -> None:
    """Preserve the benchmark identity through both JSON representations."""
    benchmark = repeat_until_success.RepeatUntilSuccess()
    assert json.loads(benchmark.instance_specification_json)["parameters"] == {}

    instance_copy = repeat_until_success.RepeatUntilSuccess.from_instance_specification_json(
        benchmark.instance_specification_json
    )
    manifest_copy = repeat_until_success.RepeatUntilSuccess.from_manifest_json(benchmark.manifest_json)
    assert instance_copy.case_id == manifest_copy.case_id == benchmark.case_id


def test_repeat_until_success_generation() -> None:
    """Generate a repeat-until-success program."""
    assert_generates(repeat_until_success.RepeatUntilSuccess().generate())
