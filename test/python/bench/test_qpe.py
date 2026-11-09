# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the qpe benchmark."""

from __future__ import annotations

from fractions import Fraction

import pytest

from mqt.core.bench import qpe


def test_qpe_accepts_fraction() -> None:
    """Normalize exact rational input in the benchmark options."""
    options = qpe.Options(
        precision=2,
        phase=Fraction(3, 24),
        method=qpe.Method.ITERATIVE,
    )
    assert options.phase == Fraction(1, 8)


def test_qpe_accepts_native_phase() -> None:
    """Normalize the native phase type before storing it in the options."""
    phase = qpe.Phase(numerator=9, denominator=8)
    native_options = qpe.Options(precision=3, phase=phase)
    assert phase.numerator == 1
    assert phase.denominator == 8
    assert native_options.phase == Fraction(1, 8)


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
