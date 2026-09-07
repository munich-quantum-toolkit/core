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


def test_qft_methods_share_the_periodic_reference() -> None:
    """Expose standard and semiclassical QFT as one family."""
    for method in (qft.Method.STANDARD, qft.Method.SEMICLASSICAL):
        benchmark = qft.QFT(qft.Options(qubits=3, period_exponent=1, method=method))
        assert benchmark.probability("000") == pytest.approx(0.5)
        assert benchmark.probability("100") == pytest.approx(0.5)
        assert (
            qft.QFT.from_instance_specification_json(benchmark.instance_specification_json).case_id == benchmark.case_id
        )
        assert_generates(benchmark.generate())
