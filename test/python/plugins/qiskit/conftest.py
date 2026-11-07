# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Keep Python device doubles at the native snapshot boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack, cast

import pytest

from mqt.core.mlir import CompilerTarget, PayloadSpecification, TargetEnvironment
from mqt.core.mlir import _submission_environment as native_submission_environment  # ruff:ignore[import-private-name] Native-boundary device doubles.
from mqt.core.mlir import _try_submit_programs as native_submit  # ruff:ignore[import-private-name] Native-boundary device doubles.
from mqt.core.plugins.qiskit import job as job_module
from mqt.core.qdmi import Device, Job, ProgramFormat

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.core.mlir import CompiledProgram
    from mqt.core.typing import QDMIJobParameters


@pytest.fixture
def mock_device_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshot Python device doubles; genuine devices use the native adapter."""
    native_snapshot = CompilerTarget.from_device
    native_environment = TargetEnvironment.from_device

    def snapshot(device: Device) -> CompilerTarget:
        if isinstance(device, Device):
            return native_snapshot(device)
        operations = []
        for operation in device.operations():
            arity = operation.qubits_num()
            placements = operation.site_pairs() if arity == 2 else operation.sites()
            tuples = (
                []
                if placements is None
                else [
                    CompilerTarget.SiteTuple([site.index() for site in pair])
                    for pair in (placements if arity == 2 else [(site,) for site in placements])
                ]
            )
            operations.append(
                CompilerTarget.OperationCapability(
                    operation.name(), arity, operation.parameters_num(), site_tuples=tuples
                )
            )
        coupling = device.coupling_map()
        return CompilerTarget(
            device.name(),
            [CompilerTarget.Site(site.index()) for site in device.sites()],
            connectivity=CompilerTarget.Connectivity.all_to_all()
            if coupling is None
            else CompilerTarget.Connectivity([(a.index(), b.index()) for a, b in coupling]),
            native_operations=CompilerTarget.NativeOperations(operations),
        )

    monkeypatch.setattr(CompilerTarget, "from_device", snapshot)

    def environment(device: Device, *, program_format: ProgramFormat | None = None) -> TargetEnvironment:
        if isinstance(device, Device):
            return native_environment(device, program_format=program_format)
        formats = device.supported_program_formats()
        chosen = program_format or next(iter(formats), None)
        if chosen is None:
            msg = "Device reports no supported program formats"
            raise ValueError(msg)
        if chosen not in formats:
            msg = "Device does not support the requested program format"
            raise ValueError(msg)
        return TargetEnvironment(snapshot(device), PayloadSpecification.from_program_format(chosen))

    monkeypatch.setattr(TargetEnvironment, "from_device", environment)

    def submission_environment(device: Device, program_format: ProgramFormat) -> TargetEnvironment:
        if isinstance(device, Device):
            return native_submission_environment(device, program_format)
        return environment(device, program_format=program_format)

    def submit_programs(
        device: Device,
        programs: Sequence[CompiledProgram],
        num_shots: int,
        *,
        destination: TargetEnvironment | None = None,
        **parameters: Unpack[QDMIJobParameters],
    ) -> Job | None:
        if isinstance(device, Device):
            return native_submit(device, programs, num_shots, destination=destination, **parameters)
        assert destination is not None
        for program in programs:
            program._validate_environment(destination)  # ruff:ignore[private-member-access] Mirror native preflight for device doubles.
        if len(programs) == 1:
            return device.submit_job(programs[0].payload, programs[0].program_format, num_shots, **parameters)
        return device.try_submit_job(
            cast("Sequence[str] | Sequence[bytes]", [program.payload for program in programs]),
            programs[0].program_format,
            num_shots,
            **parameters,
        )

    monkeypatch.setattr(job_module, "_submission_environment", submission_environment)
    monkeypatch.setattr(job_module, "_try_submit_programs", submit_programs)
