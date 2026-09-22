# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the weak-measurement Grover benchmark."""

from __future__ import annotations

import json

import pytest

from mqt.core.bench import grover_weak_measurement

from .utils import assert_generates


def _make_benchmark() -> grover_weak_measurement.Grover:
    return grover_weak_measurement.Grover(grover_weak_measurement.Options(marked_bitstring="10"))


def test_resolves_default_measurement_strength() -> None:
    """Resolve the default while accepting an explicit measurement strength."""
    options = grover_weak_measurement.Options(marked_bitstring="10")
    benchmark = grover_weak_measurement.Grover(options)

    assert options.measurement_strength is None
    assert benchmark.options.measurement_strength == pytest.approx(0.5)

    explicit = grover_weak_measurement.Grover(
        grover_weak_measurement.Options(marked_bitstring="10", measurement_strength=0.25)
    )
    assert explicit.options.measurement_strength == pytest.approx(0.25)


def test_reference() -> None:
    """Use the marked bitstring as the deterministic reference result."""
    benchmark = _make_benchmark()
    assert benchmark.qubits == 2
    assert benchmark.output.name == "result"
    assert benchmark.output.width == 2
    assert benchmark.probability("10") == 1
    assert benchmark.probability("00") == 0


def test_evaluation() -> None:
    """Report samples of the marked bitstring as successful."""
    evaluation = _make_benchmark().evaluate({"10": 20})
    assert evaluation.total_variation_distance == 0
    assert evaluation.squared_hellinger_fidelity == 1
    assert evaluation.success_probability == 1


def test_json_roundtrip() -> None:
    """Preserve the resolved measurement strength in both JSON forms."""
    benchmark = _make_benchmark()
    assert json.loads(benchmark.instance_specification_json)["parameters"] == {
        "marked_bitstring": "10",
        "measurement_strength": 0.5,
    }

    instance_copy = grover_weak_measurement.Grover.from_instance_specification_json(
        benchmark.instance_specification_json
    )
    manifest_copy = grover_weak_measurement.Grover.from_manifest_json(benchmark.manifest_json)
    assert instance_copy.case_id == manifest_copy.case_id == benchmark.case_id


def test_generation() -> None:
    """Generate a weak-measurement Grover program."""
    assert_generates(_make_benchmark().generate())
