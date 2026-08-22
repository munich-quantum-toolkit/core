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

from mqt.core import benchmarks, mlir


def test_ghz_options_reference_and_json_roundtrip() -> None:
    """Keep GHZ parameters typed and preserve one semantic case through JSON."""
    with pytest.raises(TypeError):
        benchmarks.GHZOptions(3)  # ty: ignore[missing-argument, too-many-positional-arguments]

    options = benchmarks.GHZOptions(
        qubits=3,
        topology=benchmarks.GHZTopology.STAR,
        basis=benchmarks.GHZBasis.X,
    )
    with pytest.raises(AttributeError):
        options.qubits = 4  # ty: ignore[invalid-assignment]

    benchmark = benchmarks.GHZ(options)
    assert benchmark.output.name == "result"
    assert benchmark.output.width == 3
    assert benchmark.probability("011") == pytest.approx(0.25)
    assert benchmark.probability("111") == 0

    evaluation = benchmark.evaluate({"000": 50, "011": 50})
    assert evaluation.total_variation_distance == pytest.approx(0.5)
    assert evaluation.squared_hellinger_fidelity == pytest.approx(0.5)
    assert evaluation.success_probability is None

    request_copy = benchmarks.GHZ.from_request_json(benchmark.request_json)
    manifest_copy = benchmarks.GHZ.from_manifest_json(benchmark.manifest_json)
    assert request_copy.request_json == benchmark.request_json
    assert manifest_copy.manifest_json == benchmark.manifest_json
    assert request_copy.case_id == manifest_copy.case_id == benchmark.case_id
    program = benchmark.generate()
    assert isinstance(program, mlir.QCProgram)
    assert "qc." in program.ir
    assert isinstance(program.to_qco(), mlir.QCOProgram)


def test_grover_resolves_iterations_and_reports_success() -> None:
    """Expose Grover's resolved default and marked-outcome score."""
    options = benchmarks.GroverOptions(marked_bitstring="10")
    benchmark = benchmarks.Grover(options)

    assert options.iterations is None
    assert benchmark.options.iterations == 1
    assert benchmark.qubits == 2
    assert benchmark.probability("10") == pytest.approx(1)
    assert benchmark.evaluate({"10": 20}).success_probability == pytest.approx(1)

    copy = benchmarks.Grover.from_manifest_json(benchmark.manifest_json)
    assert copy.request_json == benchmark.request_json
    assert copy.case_id == benchmark.case_id
    program = benchmark.generate()
    assert isinstance(program, mlir.QCProgram)
    assert "qc." in program.ir


def test_qpe_accepts_fraction_and_native_phase() -> None:
    """Use exact rational input without a free-form parameter dictionary."""
    options = benchmarks.QPEOptions(
        precision=2,
        phase=Fraction(3, 24),
        method=benchmarks.QPEMethod.ITERATIVE,
    )
    assert options.phase == Fraction(1, 8)

    benchmark = benchmarks.QPE(options)
    assert benchmark.probability("00") == pytest.approx((2 + 2**0.5) / 8)
    assert benchmark.probability("01") == pytest.approx((2 + 2**0.5) / 8)
    assert json.loads(benchmark.request_json)["parameters"]["phase"] == {
        "denominator": 8,
        "numerator": 1,
    }

    request_copy = benchmarks.QPE.from_request_json(benchmark.request_json)
    assert request_copy.options.phase == Fraction(1, 8)
    assert request_copy.options.method is benchmarks.QPEMethod.ITERATIVE
    assert request_copy.case_id == benchmark.case_id

    phase = benchmarks.Phase(numerator=9, denominator=8)
    native_options = benchmarks.QPEOptions(precision=3, phase=phase)
    assert phase.numerator == 1
    assert phase.denominator == 8
    assert native_options.phase == Fraction(1, 8)
    program = benchmark.generate()
    assert isinstance(program, mlir.QCProgram)
    assert "qc." in program.ir


def test_qpe_rejects_untyped_phase_input() -> None:
    """Reject generic dictionaries at the typed Python boundary."""
    with pytest.raises(TypeError, match=r"fractions\.Fraction or Phase"):
        benchmarks.QPEOptions(
            precision=3,
            phase={"numerator": 1, "denominator": 8},  # ty: ignore[invalid-argument-type]
        )


def test_qpe_normalizes_arbitrary_fraction() -> None:
    """Normalize arbitrary-size fractions before entering the native type."""
    negative = benchmarks.QPEOptions(precision=3, phase=Fraction(-1, 8))
    large = benchmarks.QPEOptions(precision=3, phase=Fraction(2**80 + 1, 8))
    assert negative.phase == Fraction(7, 8)
    assert large.phase == Fraction(1, 8)

    with pytest.raises(ValueError, match="denominator must fit in 64 bits"):
        benchmarks.QPEOptions(precision=3, phase=Fraction(1, 2**80 + 1))
