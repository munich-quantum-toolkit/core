# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the qft benchmark."""

from __future__ import annotations

import pytest

from mqt.core.bench import qft

from .utils import assert_generates

METHODS = (qft.Method.STANDARD, qft.Method.SEMICLASSICAL)


def _make_benchmark(method: qft.Method) -> qft.QFT:
    return qft.QFT(qft.Options(qubits=3, period_exponent=1, method=method))


@pytest.mark.parametrize("method", METHODS)
def test_qft_reference(method: qft.Method) -> None:
    """Use the same periodic reference distribution for both methods."""
    benchmark = _make_benchmark(method)
    assert benchmark.probability("000") == pytest.approx(0.5)
    assert benchmark.probability("100") == pytest.approx(0.5)


@pytest.mark.parametrize("method", METHODS)
def test_qft_instance_specification_roundtrip(method: qft.Method) -> None:
    """Preserve the benchmark identity through its instance specification."""
    benchmark = _make_benchmark(method)
    copy = qft.QFT.from_instance_specification_json(benchmark.instance_specification_json)
    assert copy.case_id == benchmark.case_id


@pytest.mark.parametrize("method", METHODS)
def test_qft_generation(method: qft.Method) -> None:
    """Generate both QFT methods through the Python binding."""
    assert_generates(_make_benchmark(method).generate())
