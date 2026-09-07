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
from mqt.core.bench import (
    bv,
    ghz,
    grover,
    multiplexer,
    qft,
    qft_adder,
    qpe,
    teleportation,
)


def assert_generates(
    benchmark: (
        bv.BV
        | ghz.GHZ
        | grover.Grover
        | multiplexer.Multiplexer
        | qft.QFT
        | qft_adder.QFTAdder
        | qpe.QPE
        | teleportation.Teleportation
    ),
) -> None:
    """Exercise the shared Python-to-MLIR generation boundary."""
    program = benchmark.generate()
    assert isinstance(program, mlir.QCProgram)
    assert "qc." in program.ir
    assert isinstance(program.to_qco(), mlir.QCOProgram)


def test_bv_methods_share_the_hidden_string_reference() -> None:
    """Expose static and dynamic Bernstein--Vazirani as one family."""
    for method in (bv.Method.STATIC, bv.Method.DYNAMIC):
        benchmark = bv.BV(bv.Options(hidden_bitstring="101", method=method))
        assert benchmark.probability("101") == 1
        assert benchmark.evaluate({"101": 10}).success_probability == 1
        assert bv.BV.from_manifest_json(benchmark.manifest_json).case_id == benchmark.case_id
        assert_generates(benchmark)


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
    assert_generates(benchmark)


def test_grover_resolves_iterations_and_reports_success() -> None:
    """Expose Grover's resolved default and marked-outcome score."""
    options = grover.Options(marked_bitstring="10")
    benchmark = grover.Grover(options)

    assert options.iterations is None
    assert benchmark.options.iterations == 1
    assert benchmark.qubits == 2
    assert benchmark.probability("10") == pytest.approx(1)
    assert benchmark.evaluate({"10": 20}).success_probability == pytest.approx(1)

    copy = grover.Grover.from_manifest_json(benchmark.manifest_json)
    assert copy.instance_specification_json == benchmark.instance_specification_json
    assert copy.case_id == benchmark.case_id
    assert_generates(benchmark)


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

    shots = 16_384
    counts = mlir.sample(benchmark.generate(), shots=shots, seed=17)
    assert sum(counts.values()) == shots
    assert benchmark.evaluate(counts).total_variation_distance < 0.03
    assert_generates(benchmark)


def test_qft_methods_share_the_periodic_reference() -> None:
    """Expose standard and semiclassical QFT as one family."""
    for method in (qft.Method.STANDARD, qft.Method.SEMICLASSICAL):
        benchmark = qft.QFT(qft.Options(qubits=3, period_exponent=1, method=method))
        assert benchmark.probability("000") == pytest.approx(0.5)
        assert benchmark.probability("100") == pytest.approx(0.5)
        assert (
            qft.QFT.from_instance_specification_json(benchmark.instance_specification_json).case_id == benchmark.case_id
        )
        shots = 16_384
        counts = mlir.sample(benchmark.generate(), shots=shots, seed=17)
        assert sum(counts.values()) == shots
        assert benchmark.evaluate(counts).total_variation_distance < 0.03
        assert_generates(benchmark)


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
    assert mlir.sample(benchmark.generate(), shots=128, seed=17) == {expected: 128}
    assert_generates(benchmark)


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
    counts = mlir.sample(benchmark.generate(), shots=16_384, seed=17)
    assert benchmark.evaluate(counts).total_variation_distance < 0.03


@pytest.mark.parametrize(
    ("addend", "expected"),
    [("0", "01"), ("1", "10"), ("001", "0010"), ("110", "0111"), ("111", "1000")],
)
def test_qft_adder_preserves_leading_zeros_and_carry(addend: str, expected: str) -> None:
    """Keep the input width and final carry in constant addition."""
    benchmark = qft_adder.QFTAdder(
        qft_adder.Options(
            addend=addend,
            accumulator="0" * (len(addend) - 1) + "1",
            method=qft_adder.Method.CONSTANT,
            overflow=qft_adder.Overflow.CARRY,
        )
    )
    assert mlir.sample(benchmark.generate(), shots=128, seed=17) == {expected: 128}


def test_qpe_accepts_fraction_and_native_phase() -> None:
    """Use exact rational input without a free-form parameter dictionary."""
    options = qpe.Options(
        precision=2,
        phase=Fraction(3, 24),
        method=qpe.Method.ITERATIVE,
    )
    assert options.phase == Fraction(1, 8)

    benchmark = qpe.QPE(options)
    assert benchmark.probability("00") == pytest.approx((2 + 2**0.5) / 8)
    assert benchmark.probability("01") == pytest.approx((2 + 2**0.5) / 8)
    assert json.loads(benchmark.instance_specification_json)["parameters"]["phase"] == {
        "denominator": 8,
        "numerator": 1,
    }

    instance_copy = qpe.QPE.from_instance_specification_json(benchmark.instance_specification_json)
    assert instance_copy.options.phase == Fraction(1, 8)
    assert instance_copy.options.method is qpe.Method.ITERATIVE
    assert instance_copy.case_id == benchmark.case_id

    phase = qpe.Phase(numerator=9, denominator=8)
    native_options = qpe.Options(precision=3, phase=phase)
    assert phase.numerator == 1
    assert phase.denominator == 8
    assert native_options.phase == Fraction(1, 8)
    assert_generates(benchmark)


@pytest.mark.parametrize("method", [qpe.Method.STANDARD, qpe.Method.ITERATIVE])
@pytest.mark.parametrize("phase", [Fraction(3, 8), Fraction(1, 3)])
def test_qpe_dd_sampling_matches_reference(method: qpe.Method, phase: Fraction) -> None:
    """Execute exact and inexact phases with both inverse-QFT implementations."""
    benchmark = qpe.QPE(qpe.Options(precision=3, phase=phase, method=method))
    shots = 16_384
    counts = mlir.sample(benchmark.generate(), shots=shots, seed=17)
    assert sum(counts.values()) == shots
    assert benchmark.evaluate(counts).total_variation_distance < 0.03


def test_qpe_rejects_untyped_phase_input() -> None:
    """Reject generic dictionaries at the typed Python boundary."""
    with pytest.raises(TypeError, match=r"fractions\.Fraction or Phase"):
        qpe.Options(
            precision=3,
            phase={"numerator": 1, "denominator": 8},  # ty: ignore[invalid-argument-type]
        )


def test_qpe_normalizes_arbitrary_fraction() -> None:
    """Normalize arbitrary-size fractions before entering the native type."""
    negative = qpe.Options(precision=3, phase=Fraction(-1, 8))
    large = qpe.Options(precision=3, phase=Fraction(2**80 + 1, 8))
    assert negative.phase == Fraction(7, 8)
    assert large.phase == Fraction(1, 8)

    with pytest.raises(ValueError, match="denominator must fit in 64 bits"):
        qpe.Options(precision=3, phase=Fraction(1, 2**80 + 1))


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

    shots = 128
    counts = mlir.sample(benchmark.generate(), shots=shots, seed=17)
    assert counts == {"0": shots}
    assert_generates(benchmark)
