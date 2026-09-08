# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the controlled multiplication modulo N benchmark."""

from __future__ import annotations

import json

import pytest

from mqt.core.bench import controlled_multiplication_modulo_n

from .utils import assert_generates


def _make_benchmark() -> controlled_multiplication_modulo_n.ControlledMultiplicationModuloN:
    return controlled_multiplication_modulo_n.ControlledMultiplicationModuloN(
        controlled_multiplication_modulo_n.Options(multiplier="011", modulus="101")
    )


def _exact_counts() -> dict[str, int]:
    products = ("000", "011", "001", "100", "010", "000", "011", "001")
    counts: dict[str, int] = {}
    for multiplicand, product in zip((f"{value:03b}" for value in range(8)), products, strict=True):
        counts[f"0{multiplicand}0000"] = 1
        counts[f"1{multiplicand}0{product}"] = 1
    return counts


def test_controlled_multiplication_modulo_n_reference() -> None:
    """Expose the options, output, and exact modular products."""
    benchmark = _make_benchmark()
    assert benchmark.options.multiplier == "011"
    assert benchmark.options.modulus == "101"
    assert benchmark.output.name == "result"
    assert benchmark.output.width == 8
    for outcome in _exact_counts():
        assert benchmark.probability(outcome) == pytest.approx(1 / 16)
    assert benchmark.probability("10010000") == 0


def test_controlled_multiplication_modulo_n_evaluation() -> None:
    """Evaluate exact counts against the modular-product reference."""
    evaluation = _make_benchmark().evaluate(_exact_counts())
    assert evaluation.total_variation_distance == pytest.approx(0)
    assert evaluation.squared_hellinger_fidelity == pytest.approx(1)
    assert evaluation.success_probability is None


def test_controlled_multiplication_modulo_n_json_roundtrip() -> None:
    """Preserve the benchmark identity through both JSON representations."""
    benchmark = _make_benchmark()
    assert json.loads(benchmark.instance_specification_json)["parameters"] == {
        "modulus": "101",
        "multiplier": "011",
    }

    instance_copy = controlled_multiplication_modulo_n.ControlledMultiplicationModuloN.from_instance_specification_json(
        benchmark.instance_specification_json
    )
    manifest_copy = controlled_multiplication_modulo_n.ControlledMultiplicationModuloN.from_manifest_json(
        benchmark.manifest_json
    )
    assert instance_copy.case_id == manifest_copy.case_id == benchmark.case_id


def test_controlled_multiplication_modulo_n_generation() -> None:
    """Generate a controlled modular-multiplication program."""
    assert_generates(_make_benchmark().generate())
