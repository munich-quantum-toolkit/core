# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QAE workflow callbacks and result conversions."""

from __future__ import annotations

import pytest

from mqt.core.bench import qae


def test_estimation_callbacks() -> None:
    """Pass typed experiments to the executor and recover endpoint estimates."""
    problem = qae.Problem(good_states=["10"], amplitude=1.0)

    def run(benchmark: qae.QAE) -> dict[str, int]:
        assert benchmark.options.problem.good_states == ["10"]
        return {"100" if benchmark.options.precision else "10": 256}

    assert qae.phase_estimation(problem, run, precision=3) == pytest.approx(1.0)
    assert qae.maximum_likelihood(problem, run, rounds=2) == pytest.approx(1.0)
    low, high = qae.iterative(problem, run, epsilon=0.01)
    assert 0.98 <= low <= high
    assert high == pytest.approx(1.0)


def test_iterative_preserves_callback_exception() -> None:
    """Propagate the original exception from an adaptive round."""
    error = RuntimeError("sampling failed")

    def fail(_: qae.QAE) -> dict[str, int]:
        raise error

    with pytest.raises(RuntimeError) as caught:
        qae.iterative(qae.Problem(good_states=["0"], amplitude=0.3), fail, epsilon=0.1)
    assert caught.value is error
