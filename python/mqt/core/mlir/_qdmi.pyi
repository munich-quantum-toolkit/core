# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared native QDMI execution support."""

from collections.abc import Sequence

import mqt.core.mlir
import mqt.core.qdmi

class Context:
    """Retain a QDMI device, target snapshot, and default job parameters."""

    def __init__(self, device: mqt.core.qdmi.Device) -> None:
        """Create an execution context from an open device."""

    @property
    def environment(self) -> mqt.core.mlir.TargetEnvironment[mqt.core.mlir.CompiledProgram]:
        """Target environment used to compile programs for this context."""

    def prepare(
        self, programs: tuple[mqt.core.mlir.CompiledProgram, ...]
    ) -> dict[mqt.core.qdmi.ProgramFormat, mqt.core.mlir.TargetEnvironment]:
        """Validate all format groups before submitting any program.

        Programs compiled against this context reuse its target snapshot. Other groups
        obtain a fresh destination snapshot. Returns destinations indexed by format.
        """

    def submit(self, program: mqt.core.mlir.CompiledProgram, num_shots: int) -> mqt.core.qdmi.Job:
        """Submit one compiled program with the retained job parameters.

        Args:
            program: Program compiled for this context's target environment.
            num_shots: Nonnegative number of execution shots.

        Returns:
            The submitted QDMI job, without waiting for completion.
        """

    def try_submit(
        self,
        programs: Sequence[mqt.core.mlir.CompiledProgram],
        num_shots: int,
        *,
        custom1: str | bool | float | bytes | None = None,
        custom2: str | bool | float | bytes | None = None,
        custom3: str | bool | float | bytes | None = None,
        custom4: str | bool | float | bytes | None = None,
        custom5: str | bool | float | bytes | None = None,
        destination: mqt.core.mlir.TargetEnvironment | None = None,
    ) -> mqt.core.qdmi.Job | None:
        """Submit a compatible group, returning None if the provider declines it.

        Args:
            programs: Programs with one payload format.
            num_shots: Nonnegative shot count applied to every program.
            custom1: First provider-specific job parameter.
            custom2: Second provider-specific job parameter.
            custom3: Third provider-specific job parameter.
            custom4: Fourth provider-specific job parameter.
            custom5: Fifth provider-specific job parameter.
            destination: Validated destination from prepare, or this context's snapshot.

        A supplied custom parameter replaces the retained parameter set for this call.
        The context itself is unchanged. With no supplied custom parameters, use the
        retained defaults.
        """
