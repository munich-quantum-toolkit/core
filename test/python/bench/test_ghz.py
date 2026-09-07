# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the ghz benchmark."""

from __future__ import annotations

import pytest

from mqt.core import bench
from mqt.core.bench import ghz

from .utils import assert_generates


def test_ghz_options_reference_and_json_roundtrip() -> None:
    """Keep GHZ parameters typed and preserve one semantic case through JSON."""
    with pytest.raises(TypeError):
        ghz.Options(3)  # ty: ignore[missing-argument, too-many-positional-arguments]

    options = ghz.Options(
        qubits=3,
        topology=ghz.Topology.STAR,
        basis=ghz.Basis.X,
    )
    with pytest.raises(AttributeError):
        options.qubits = 4  # ty: ignore[invalid-assignment]

    benchmark = ghz.GHZ(options)
    assert isinstance(benchmark.output, bench.Output)
    assert benchmark.output.name == "result"
    assert benchmark.output.width == 3
    assert benchmark.probability("011") == pytest.approx(0.25)
    assert benchmark.probability("111") == 0

    evaluation = benchmark.evaluate({"000": 50, "011": 50})
    assert isinstance(evaluation, bench.Evaluation)
    assert evaluation.total_variation_distance == pytest.approx(0.5)
    assert evaluation.squared_hellinger_fidelity == pytest.approx(0.5)
    assert evaluation.success_probability is None

    instance_copy = ghz.GHZ.from_instance_specification_json(benchmark.instance_specification_json)
    manifest_copy = ghz.GHZ.from_manifest_json(benchmark.manifest_json)
    assert instance_copy.instance_specification_json == benchmark.instance_specification_json
    assert manifest_copy.manifest_json == benchmark.manifest_json
    assert instance_copy.case_id == manifest_copy.case_id == benchmark.case_id
    assert_generates(benchmark.generate())
