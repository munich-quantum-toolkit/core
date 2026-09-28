# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shor bindings and the callback-based factoring workflow."""

from __future__ import annotations

import json

import pytest

from mqt.core.bench import shor

from .utils import assert_generates


def test_shor_instance() -> None:
    """Preserve the configured circuit and its verification reference."""
    benchmark = shor.Shor(shor.Options(number=21))
    assert benchmark.options.base == 2
    assert benchmark.output.name == "result"
    assert benchmark.output.width == 10
    assert json.loads(benchmark.instance_specification_json)["parameters"] == {"number": 21, "base": 2}
    assert json.loads(benchmark.manifest_json)["reference"]["kind"] == "verification"
    for restored in (
        shor.Shor.from_instance_specification_json(benchmark.instance_specification_json),
        shor.Shor.from_manifest_json(benchmark.manifest_json),
    ):
        assert restored.case_id == benchmark.case_id
    evaluation = benchmark.evaluate({"0010101011": 3, "0000000000": 1})
    assert evaluation.factors == (3, 7)
    assert evaluation.success_probability == pytest.approx(0.75)
    assert_generates(benchmark.generate())


def test_factor_callback() -> None:
    """Pass resolved instances to Python and return verified native results."""
    bases = []

    def run(benchmark: shor.Shor) -> dict[str, int]:
        assert benchmark.options.number == 21
        bases.append(benchmark.options.base)
        return {"0010101011": 64}

    result = shor.factor(21, run)
    assert result.status == shor.FactorStatus.SUCCESS
    assert result.factors == (3, 7)
    assert result.attempts == 1
    assert bases == [2]
    exhausted = shor.factor(21, lambda _: {"0000000000": 1}, max_attempts=1)
    assert exhausted.status == shor.FactorStatus.ATTEMPTS_EXHAUSTED
    assert exhausted.factors is None
    assert exhausted.attempts == 1


@pytest.mark.parametrize(("number", "factors"), [(2, None), (31, None), (14, (2, 7)), (27, (3, 9))])
def test_factor_prechecks(number: int, factors: tuple[int, int] | None) -> None:
    """Classical prechecks require no execution."""

    def unexpected(_: shor.Shor) -> dict[str, int]:
        pytest.fail("classical precheck called the executor")

    result = shor.factor(number, unexpected)
    assert result.factors == factors
    assert result.status == (shor.FactorStatus.PRIME if factors is None else shor.FactorStatus.SUCCESS)
    assert result.attempts == 0


def test_shor_errors() -> None:
    """Propagate validation and executor failures across the binding."""
    with pytest.raises(ValueError, match="coprime"):
        shor.Shor(shor.Options(number=21, base=3))
    with pytest.raises(ValueError, match="width"):
        shor.Shor(shor.Options(number=21)).evaluate({"0": 1})

    def failed(_: shor.Shor) -> dict[str, int]:
        msg = "execution failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="execution failed"):
        shor.factor(21, failed)
