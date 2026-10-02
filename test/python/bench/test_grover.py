# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the grover benchmark."""

from __future__ import annotations

import pytest

from mqt.core.bench import grover

from .utils import assert_generates


def _make_benchmark() -> grover.Grover:
    return grover.Grover(grover.Options(marked_bitstring="10"))


def test_grover_resolves_default_iterations() -> None:
    """Resolve the default iteration count when constructing the benchmark."""
    options = grover.Options(marked_bitstring="10")
    benchmark = grover.Grover(options)

    assert options.iterations is None
    assert benchmark.options.iterations == 1


def test_grover_reference() -> None:
    """Use the marked bitstring as the deterministic reference result."""
    benchmark = _make_benchmark()
    assert benchmark.qubits == 2
    assert benchmark.probability("10") == pytest.approx(1)


def test_grover_evaluation() -> None:
    """Report samples of the marked bitstring as successful."""
    benchmark = _make_benchmark()
    assert benchmark.evaluate({"10": 20}).success_probability == pytest.approx(1)


def test_grover_manifest_roundtrip() -> None:
    """Preserve the benchmark identity and instance through its manifest."""
    benchmark = _make_benchmark()
    copy = grover.Grover.from_manifest_json(benchmark.manifest_json)
    assert copy.instance_specification_json == benchmark.instance_specification_json
    assert copy.case_id == benchmark.case_id


def test_grover_generation() -> None:
    """Generate a Grover program through the Python binding."""
    assert_generates(_make_benchmark().generate())
