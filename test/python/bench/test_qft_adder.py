# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the qft_adder benchmark."""

from __future__ import annotations

import json

import pytest

from mqt.core.bench import qft_adder

from .utils import assert_generates


@pytest.mark.parametrize("method", [qft_adder.Method.REGISTER, qft_adder.Method.CONSTANT])
@pytest.mark.parametrize("overflow", [qft_adder.Overflow.WRAP, qft_adder.Overflow.CARRY])
def test_qft_adder_reference_json_and_generation(method: qft_adder.Method, overflow: qft_adder.Overflow) -> None:
    """Expose both operand representations with the same overflow contract."""
    benchmark = qft_adder.QFTAdder(qft_adder.Options(addend="110", accumulator="011", method=method, overflow=overflow))
    expected_sum = "1001" if overflow == qft_adder.Overflow.CARRY else "001"
    expected = ("110" if method == qft_adder.Method.REGISTER else "") + expected_sum
    assert benchmark.options.addend == "110"
    assert benchmark.options.accumulator == "011"
    assert benchmark.output.width == len(expected)
    assert benchmark.expected_result == expected
    assert benchmark.probability(expected) == 1
    assert benchmark.evaluate({expected: 8}).success_probability == 1
    parameters = json.loads(benchmark.instance_specification_json)["parameters"]
    assert parameters["addend"] == "110"
    assert parameters["accumulator"] == "011"
    copy = qft_adder.QFTAdder.from_instance_specification_json(benchmark.instance_specification_json)
    manifest_copy = qft_adder.QFTAdder.from_manifest_json(benchmark.manifest_json)
    assert copy.case_id == manifest_copy.case_id == benchmark.case_id
    assert_generates(benchmark.generate())


def test_qft_adder_superposition_reference() -> None:
    """Keep the observable correlation for a partly superposed addend."""
    benchmark = qft_adder.QFTAdder(qft_adder.Options(addend="1+0", accumulator="001"))
    assert benchmark.expected_result is None
    assert benchmark.probability("100101") == pytest.approx(0.5)
    assert benchmark.probability("110111") == pytest.approx(0.5)
    assert benchmark.probability("000001") == 0
    assert benchmark.probability("100100") == 0
    evaluation = benchmark.evaluate({"100101": 1, "110111": 1})
    assert evaluation.total_variation_distance == 0
    assert evaluation.success_probability is None
