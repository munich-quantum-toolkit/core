# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for typed benchmark instances and analytic references."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from mqt.core import bench, mlir


def assert_generates(benchmark: bench.BV | bench.GHZ | bench.Grover | bench.QFT | bench.QPE) -> None:
    """Exercise the shared Python-to-MLIR generation boundary."""
    program = benchmark.generate()
    assert isinstance(program, mlir.QCProgram)
    assert "qc." in program.ir
    assert isinstance(program.to_qco(), mlir.QCOProgram)


def test_bv_methods_share_the_hidden_string_reference() -> None:
    """Expose static and dynamic Bernstein--Vazirani as one family."""
    for method in (bench.BVMethod.STATIC, bench.BVMethod.DYNAMIC):
        benchmark = bench.BV(bench.BVOptions(hidden_bitstring="101", method=method))
        assert benchmark.probability("101") == 1
        assert benchmark.evaluate({"101": 10}).success_probability == 1
        assert bench.BV.from_manifest_json(benchmark.manifest_json).case_id == benchmark.case_id
        assert_generates(benchmark)


def test_ghz_options_reference_and_json_roundtrip() -> None:
    """Keep GHZ parameters typed and preserve one semantic case through JSON."""
    with pytest.raises(TypeError):
        bench.GHZOptions(3)  # ty: ignore[missing-argument, too-many-positional-arguments]

    options = bench.GHZOptions(
        qubits=3,
        topology=bench.GHZTopology.STAR,
        basis=bench.GHZBasis.X,
    )
    with pytest.raises(AttributeError):
        options.qubits = 4  # ty: ignore[invalid-assignment]

    benchmark = bench.GHZ(options)
    assert benchmark.output.name == "result"
    assert benchmark.output.width == 3
    assert benchmark.probability("011") == pytest.approx(0.25)
    assert benchmark.probability("111") == 0

    evaluation = benchmark.evaluate({"000": 50, "011": 50})
    assert evaluation.total_variation_distance == pytest.approx(0.5)
    assert evaluation.squared_hellinger_fidelity == pytest.approx(0.5)
    assert evaluation.success_probability is None

    instance_copy = bench.GHZ.from_instance_specification_json(benchmark.instance_specification_json)
    manifest_copy = bench.GHZ.from_manifest_json(benchmark.manifest_json)
    assert instance_copy.instance_specification_json == benchmark.instance_specification_json
    assert manifest_copy.manifest_json == benchmark.manifest_json
    assert instance_copy.case_id == manifest_copy.case_id == benchmark.case_id
    assert_generates(benchmark)


def test_grover_resolves_iterations_and_reports_success() -> None:
    """Expose Grover's resolved default and marked-outcome score."""
    options = bench.GroverOptions(marked_bitstring="10")
    benchmark = bench.Grover(options)

    assert options.iterations is None
    assert benchmark.options.iterations == 1
    assert benchmark.qubits == 2
    assert benchmark.probability("10") == pytest.approx(1)
    assert benchmark.evaluate({"10": 20}).success_probability == pytest.approx(1)

    copy = bench.Grover.from_manifest_json(benchmark.manifest_json)
    assert copy.instance_specification_json == benchmark.instance_specification_json
    assert copy.case_id == benchmark.case_id
    assert_generates(benchmark)


def test_qft_methods_share_the_periodic_reference() -> None:
    """Expose standard and semiclassical QFT as one family."""
    for method in (bench.QFTMethod.STANDARD, bench.QFTMethod.SEMICLASSICAL):
        benchmark = bench.QFT(bench.QFTOptions(qubits=3, period_exponent=1, method=method))
        assert benchmark.probability("000") == pytest.approx(0.5)
        assert benchmark.probability("100") == pytest.approx(0.5)
        assert (
            bench.QFT.from_instance_specification_json(benchmark.instance_specification_json).case_id
            == benchmark.case_id
        )
        assert_generates(benchmark)


def test_qpe_accepts_fraction_and_native_phase() -> None:
    """Use exact rational input without a free-form parameter dictionary."""
    options = bench.QPEOptions(
        precision=2,
        phase=Fraction(3, 24),
        method=bench.QPEMethod.ITERATIVE,
    )
    assert options.phase == Fraction(1, 8)

    benchmark = bench.QPE(options)
    assert benchmark.probability("00") == pytest.approx((2 + 2**0.5) / 8)
    assert benchmark.probability("01") == pytest.approx((2 + 2**0.5) / 8)
    assert json.loads(benchmark.instance_specification_json)["parameters"]["phase"] == {
        "denominator": 8,
        "numerator": 1,
    }

    instance_copy = bench.QPE.from_instance_specification_json(benchmark.instance_specification_json)
    assert instance_copy.options.phase == Fraction(1, 8)
    assert instance_copy.options.method is bench.QPEMethod.ITERATIVE
    assert instance_copy.case_id == benchmark.case_id

    phase = bench.Phase(numerator=9, denominator=8)
    native_options = bench.QPEOptions(precision=3, phase=phase)
    assert phase.numerator == 1
    assert phase.denominator == 8
    assert native_options.phase == Fraction(1, 8)
    assert_generates(benchmark)


def test_qpe_rejects_untyped_phase_input() -> None:
    """Reject generic dictionaries at the typed Python boundary."""
    with pytest.raises(TypeError, match=r"fractions\.Fraction or Phase"):
        bench.QPEOptions(
            precision=3,
            phase={"numerator": 1, "denominator": 8},  # ty: ignore[invalid-argument-type]
        )


def test_qpe_normalizes_arbitrary_fraction() -> None:
    """Normalize arbitrary-size fractions before entering the native type."""
    negative = bench.QPEOptions(precision=3, phase=Fraction(-1, 8))
    large = bench.QPEOptions(precision=3, phase=Fraction(2**80 + 1, 8))
    assert negative.phase == Fraction(7, 8)
    assert large.phase == Fraction(1, 8)

    with pytest.raises(ValueError, match="denominator must fit in 64 bits"):
        bench.QPEOptions(precision=3, phase=Fraction(1, 2**80 + 1))
