# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Public interfaces and execution of concatenated magic-state distillation."""

from __future__ import annotations

import pytest

from mqt.core.bench import magic_state_distillation
from mqt.core.mlir import sample, submit_program
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.builtin_driver import open_device


def test_distillation_direct_sampling() -> None:
    """Sample both direct rotations and injected lower-level magic states."""
    benchmark = magic_state_distillation.MagicStateDistillation(magic_state_distillation.Options(levels=2))
    assert sample(benchmark.generate(), shots=16, seed=17) == {"00": 16}


@pytest.mark.parametrize("program_format", [None, ProgramFormat.QIR_ADAPTIVE_MODULE])
def test_distillation_ddsim(program_format: ProgramFormat | None) -> None:
    """Compile and execute the same circuit through both DDSIM payload paths."""
    benchmark = magic_state_distillation.MagicStateDistillation()
    device = open_device("mqt.ddsim.default", custom1="5")
    job = submit_program(benchmark.generate(), target=device, program_format=program_format, num_shots=16, custom1=17)
    assert job.program_format == (ProgramFormat.QASM3 if program_format is None else program_format)
    job.wait()
    assert job.get_counts() == {"00": 16}
