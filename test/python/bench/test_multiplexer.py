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


def test_multiplexer_reference_json_and_generation() -> None:
    """Expose the fixed-angle quantum multiplexer as one typed family."""
    benchmark = multiplexer.Multiplexer(multiplexer.Options(qubits=3))
    assert benchmark.output.name == "result"
    assert benchmark.output.width == 3
    assert benchmark.probability("000") == pytest.approx(0.25)
    assert benchmark.probability("001") == 0

    evaluation = benchmark.evaluate({"000": 10})
    assert evaluation.total_variation_distance == pytest.approx(0.75)
    assert evaluation.squared_hellinger_fidelity == pytest.approx(0.25)
    assert evaluation.success_probability is None
    assert json.loads(benchmark.instance_specification_json)["parameters"] == {"qubits": 3}

    instance_copy = multiplexer.Multiplexer.from_instance_specification_json(benchmark.instance_specification_json)
    manifest_copy = multiplexer.Multiplexer.from_manifest_json(benchmark.manifest_json)
    assert instance_copy.case_id == manifest_copy.case_id == benchmark.case_id

    assert_generates(benchmark.generate())
