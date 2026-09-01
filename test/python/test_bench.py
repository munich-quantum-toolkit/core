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
from mqt.core.bench import bv, ghz, grover, qft, qpe


def assert_generates(benchmark: bv.BV | ghz.GHZ | grover.Grover | qft.QFT | qpe.QPE) -> None:
    """Exercise the shared Python-to-MLIR generation boundary."""
    program = benchmark.generate()
    assert isinstance(program, mlir.QCProgram)
    assert "qc." in program.ir
    assert isinstance(program.to_qco(), mlir.QCOProgram)


def test_benchmark_namespace_groups_family_types() -> None:
    """Keep shared types at the root and family types in direct submodules."""
    assert bench.Output.__module__ == "mqt.core.bench"
    assert bench.Evaluation.__module__ == "mqt.core.bench"

    assert bv.__name__ == "mqt.core.bench.bv"
    assert bv.BV.__module__ == bv.__name__
    assert bv.Options.__module__ == bv.__name__
    assert bv.Method.__module__ == bv.__name__

    assert ghz.__name__ == "mqt.core.bench.ghz"
    assert ghz.GHZ.__module__ == ghz.__name__
    assert ghz.Options.__module__ == ghz.__name__
    assert ghz.Topology.__module__ == ghz.__name__
    assert ghz.Basis.__module__ == ghz.__name__

    assert grover.__name__ == "mqt.core.bench.grover"
    assert grover.Grover.__module__ == grover.__name__
    assert grover.Options.__module__ == grover.__name__

    assert qft.__name__ == "mqt.core.bench.qft"
    assert qft.QFT.__module__ == qft.__name__
    assert qft.Options.__module__ == qft.__name__
    assert qft.Method.__module__ == qft.__name__

    assert qpe.__name__ == "mqt.core.bench.qpe"
    assert qpe.Phase.__module__ == qpe.__name__
    assert qpe.QPE.__module__ == qpe.__name__
    assert qpe.Options.__module__ == qpe.__name__
    assert qpe.Method.__module__ == qpe.__name__

    root_family_names = (
        "Basis",
        "BV",
        "BVMethod",
        "BVOptions",
        "GHZ",
        "GHZBasis",
        "GHZOptions",
        "GHZTopology",
        "Grover",
        "GroverOptions",
        "Method",
        "Options",
        "Phase",
        "QFT",
        "QFTMethod",
        "QFTOptions",
        "QPE",
        "QPEMethod",
        "QPEOptions",
        "Topology",
    )
    assert all(not hasattr(bench, name) for name in root_family_names)
    for family_module in (bv, ghz, grover, qft, qpe):
        assert not hasattr(family_module, "Output")
        assert not hasattr(family_module, "Evaluation")

    legacy_family_names = (
        (bv, ("BVMethod", "BVOptions")),
        (ghz, ("GHZBasis", "GHZOptions", "GHZTopology")),
        (grover, ("GroverOptions",)),
        (qft, ("QFTMethod", "QFTOptions")),
        (qpe, ("QPEMethod", "QPEOptions")),
    )
    for family_module, names in legacy_family_names:
        assert all(not hasattr(family_module, name) for name in names)


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


def test_importing_mlir_preserves_benchmark_overflow_error() -> None:
    """Preserve nanobind's ``OverflowError`` mapping after loading MLIR."""
    benchmark = ghz.GHZ(ghz.Options(qubits=2))
    with pytest.raises(OverflowError, match="total shot count exceeds size_t"):
        benchmark.evaluate({"00": 2**64 - 1, "11": 1})


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


def test_qft_methods_share_the_periodic_reference() -> None:
    """Expose standard and semiclassical QFT as one family."""
    for method in (qft.Method.STANDARD, qft.Method.SEMICLASSICAL):
        benchmark = qft.QFT(qft.Options(qubits=3, period_exponent=1, method=method))
        assert benchmark.probability("000") == pytest.approx(0.5)
        assert benchmark.probability("100") == pytest.approx(0.5)
        assert (
            qft.QFT.from_instance_specification_json(benchmark.instance_specification_json).case_id == benchmark.case_id
        )
        assert_generates(benchmark)


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
