# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the bv benchmark."""

from __future__ import annotations

import pytest

from mqt.core.bench import bv

from .utils import assert_generates

METHODS = (bv.Method.STATIC, bv.Method.DYNAMIC)


def _make_benchmark(method: bv.Method) -> bv.BV:
    return bv.BV(bv.Options(hidden_bitstring="101", method=method))


@pytest.mark.parametrize("method", METHODS)
def test_bv_reference(method: bv.Method) -> None:
    """Use the hidden bitstring as the deterministic reference result."""
    assert _make_benchmark(method).probability("101") == 1


@pytest.mark.parametrize("method", METHODS)
def test_bv_evaluation(method: bv.Method) -> None:
    """Report successful samples for both methods."""
    assert _make_benchmark(method).evaluate({"101": 10}).success_probability == 1


@pytest.mark.parametrize("method", METHODS)
def test_bv_manifest_roundtrip(method: bv.Method) -> None:
    """Preserve the benchmark identity through its manifest."""
    benchmark = _make_benchmark(method)
    assert bv.BV.from_manifest_json(benchmark.manifest_json).case_id == benchmark.case_id


@pytest.mark.parametrize("method", METHODS)
def test_bv_generation(method: bv.Method) -> None:
    """Generate both Bernstein--Vazirani methods."""
    assert_generates(_make_benchmark(method).generate())
