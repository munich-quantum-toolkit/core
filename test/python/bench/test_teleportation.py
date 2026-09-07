# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the teleportation benchmark."""

from __future__ import annotations

import json

import pytest

from mqt.core.bench import teleportation

from .utils import assert_generates


def test_teleportation_reference_json_and_generation() -> None:
    """Expose the fixed quantum teleportation benchmark without options."""
    benchmark = teleportation.Teleportation()
    assert benchmark.output.name == "result"
    assert benchmark.output.width == 1
    assert benchmark.probability("0") == 1
    assert benchmark.probability("1") == 0

    evaluation = benchmark.evaluate({"0": 128})
    assert evaluation.total_variation_distance == pytest.approx(0)
    assert evaluation.squared_hellinger_fidelity == pytest.approx(1)
    assert evaluation.success_probability == 1
    assert json.loads(benchmark.instance_specification_json)["parameters"] == {}

    instance_copy = teleportation.Teleportation.from_instance_specification_json(benchmark.instance_specification_json)
    manifest_copy = teleportation.Teleportation.from_manifest_json(benchmark.manifest_json)
    assert instance_copy.case_id == manifest_copy.case_id == benchmark.case_id

    assert_generates(benchmark.generate())
