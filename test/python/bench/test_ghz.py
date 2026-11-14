# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the ghz benchmark."""

from __future__ import annotations

import importlib

import pytest

from mqt.core.bench import ghz


def _make_benchmark() -> ghz.GHZ:
    return ghz.GHZ(
        ghz.Options(
            qubits=3,
            topology=ghz.Topology.STAR,
            basis=ghz.Basis.X,
        )
    )


def test_ghz_options_are_keyword_only_and_read_only() -> None:
    """Keep GHZ options keyword-only and immutable."""
    with pytest.raises(TypeError):
        ghz.Options(3)  # ty: ignore[missing-argument, too-many-positional-arguments]

    options = ghz.Options(
        qubits=3,
        topology=ghz.Topology.STAR,
        basis=ghz.Basis.X,
    )
    with pytest.raises(AttributeError):
        options.qubits = 4  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("outcome", ["01", "01x"])
def test_probability_rejects_invalid_outcomes(outcome: str) -> None:
    """Validate Python strings before calling the native scalar query."""
    with pytest.raises(ValueError, match="outcome"):
        _make_benchmark().probability(outcome)


def test_importing_mlir_preserves_benchmark_overflow_error() -> None:
    """Keep benchmark overflow translation after loading the MLIR module."""
    importlib.import_module("mqt.core.mlir")
    benchmark = ghz.GHZ(ghz.Options(qubits=2))
    with pytest.raises(OverflowError, match="total shot count exceeds size_t"):
        benchmark.evaluate({"00": 2**64 - 1, "11": 1})
