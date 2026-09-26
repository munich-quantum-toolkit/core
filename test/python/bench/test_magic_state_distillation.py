# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Public interfaces and execution of concatenated magic-state distillation."""

from __future__ import annotations

import json

import pytest

from mqt.core.bench import magic_state_distillation
from mqt.core.mlir import compile_program, sample, submit_program
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device

from .utils import assert_generates


@pytest.mark.parametrize("levels", [1, 2, 3, 4])
def test_distillation_roundtrip(levels: int) -> None:
    """Preserve the level count and semantic identity through both JSON forms."""
    benchmark = magic_state_distillation.MagicStateDistillation(magic_state_distillation.Options(levels=levels))
    assert benchmark.options.levels == levels
    assert benchmark.output.name == "result"
    assert benchmark.output.width == 2
    assert json.loads(benchmark.instance_specification_json)["parameters"] == {"levels": levels}
    instance = magic_state_distillation.MagicStateDistillation.from_instance_specification_json(
        benchmark.instance_specification_json
    )
    manifest = magic_state_distillation.MagicStateDistillation.from_manifest_json(benchmark.manifest_json)
    assert instance.case_id == manifest.case_id == benchmark.case_id
    if levels != 1:
        assert benchmark.case_id != magic_state_distillation.MagicStateDistillation().case_id
    assert_generates(benchmark.generate())


@pytest.mark.parametrize("levels", [0, 5])
def test_distillation_invalid_levels(levels: int) -> None:
    """Reject unsupported concatenation levels."""
    with pytest.raises(ValueError, match="levels must be between 1 and 4"):
        magic_state_distillation.MagicStateDistillation(magic_state_distillation.Options(levels=levels))


def test_distillation_reference() -> None:
    """Success requires both acceptance and the correct root state."""
    benchmark = magic_state_distillation.MagicStateDistillation()
    assert benchmark.probability("00") == 1
    for outcome in ("01", "10", "11"):
        assert benchmark.probability(outcome) == 0
    evaluation = benchmark.evaluate({"00": 5, "01": 1, "10": 1, "11": 1})
    assert evaluation.success_probability == pytest.approx(0.625)
    assert evaluation.total_variation_distance == pytest.approx(0.375)
    assert evaluation.squared_hellinger_fidelity == pytest.approx(0.625)


def test_distillation_direct_sampling() -> None:
    """Sample the 15-qubit program with the public DD interface."""
    benchmark = magic_state_distillation.MagicStateDistillation()
    assert sample(benchmark.generate(), shots=16, seed=17) == {"00": 16}


@pytest.mark.parametrize("program_format", [None, ProgramFormat.QASM3])
def test_distillation_ddsim(program_format: ProgramFormat | None) -> None:
    """Compile and execute the same circuit through both DDSIM payload paths."""
    benchmark = magic_state_distillation.MagicStateDistillation()
    device = open_device("mqt.ddsim.default")
    compiled = compile_program(benchmark.generate(), target=device, program_format=program_format)
    if program_format is None:
        assert compiled.program_format in {ProgramFormat.QIR_ADAPTIVE_MODULE, ProgramFormat.QIR_ADAPTIVE_STRING}
    job = submit_program(compiled, target=device, num_shots=16, custom1=17)
    job.wait()
    assert job.get_counts() == {"00": 16}
