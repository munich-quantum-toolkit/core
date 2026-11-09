# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Benchmark construction, interchange, execution, and evaluation in Python."""

from __future__ import annotations

from fractions import Fraction

import pytest

from mqt.core.bench import (
    bv,
    fan_out,
    ghz,
    grover,
    grover_weak_measurement,
    magic_state_distillation,
    modular_multiplier,
    multiplexer,
    qft,
    qft_adder,
    qpe,
    repeat_until_success,
    teleportation,
    w_state,
)
from mqt.core.mlir import JeffProgram, sample

ReferenceBenchmark = (
    bv.BV
    | fan_out.FanOut
    | ghz.GHZ
    | grover.Grover
    | grover_weak_measurement.Grover
    | magic_state_distillation.MagicStateDistillation
    | modular_multiplier.ModularMultiplier
    | multiplexer.Multiplexer
    | qft.QFT
    | qft_adder.QFTAdder
    | qpe.QPE
    | repeat_until_success.RepeatUntilSuccess
    | teleportation.Teleportation
    | w_state.WState
)


@pytest.mark.parametrize(
    "benchmark",
    [
        pytest.param(bv.BV(bv.Options(hidden_bitstring="101", method=bv.Method.DYNAMIC)), id="bv"),
        pytest.param(fan_out.FanOut(fan_out.Options(controls=2)), id="fan-out"),
        pytest.param(ghz.GHZ(ghz.Options(qubits=3, topology=ghz.Topology.STAR, basis=ghz.Basis.X)), id="ghz"),
        pytest.param(grover.Grover(grover.Options(marked_bitstring="10")), id="grover"),
        pytest.param(
            grover_weak_measurement.Grover(grover_weak_measurement.Options(marked_bitstring="10")),
            id="weak-grover",
        ),
        pytest.param(magic_state_distillation.MagicStateDistillation(), id="distillation"),
        pytest.param(
            modular_multiplier.ModularMultiplier(
                modular_multiplier.Options(multiplier="10", modulus="11", multiplicand="+0", control="+")
            ),
            id="multiplier",
        ),
        pytest.param(multiplexer.Multiplexer(multiplexer.Options(qubits=3)), id="multiplexer"),
        pytest.param(qft.QFT(qft.Options(qubits=3, period_exponent=1, method=qft.Method.SEMICLASSICAL)), id="qft"),
        pytest.param(
            qft_adder.QFTAdder(qft_adder.Options(addend="11", accumulator="01", overflow=qft_adder.Overflow.CARRY)),
            id="adder",
        ),
        pytest.param(qpe.QPE(qpe.Options(precision=2, phase=Fraction(1, 8), method=qpe.Method.ITERATIVE)), id="qpe"),
        pytest.param(repeat_until_success.RepeatUntilSuccess(), id="repeat-until-success"),
        pytest.param(teleportation.Teleportation(), id="teleportation"),
        pytest.param(w_state.WState(w_state.Options(qubits=3)), id="w-state"),
    ],
)
def test_benchmark_workflow(benchmark: ReferenceBenchmark) -> None:
    """Execute a restored instance and score real outcomes through the binding."""
    benchmark_type = type(benchmark)
    instance = benchmark_type.from_instance_specification_json(benchmark.instance_specification_json)
    restored = benchmark_type.from_manifest_json(instance.manifest_json)
    assert restored.case_id == benchmark.case_id
    assert restored.manifest_json == benchmark.manifest_json
    program = restored.generate().to_qco().to_jeff()
    counts = sample(JeffProgram.from_bytes(program.to_bytes()), shots=512, seed=17)
    assert all(len(bits) == benchmark.output.width and benchmark.probability(bits) > 0 for bits in counts)
    evaluation = benchmark.evaluate(counts)
    assert evaluation.shots == sum(counts.values()) == 512
    # Binding smoke checks; native sampling tests distinguish the reference distributions.
    assert evaluation.total_variation_distance < 0.2
    assert evaluation.squared_hellinger_fidelity > 0.9
