# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compact test support for the QDMI PennyLane plugin."""

from __future__ import annotations

import math
import re
from collections import Counter
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

from mqt.core.qdmi import Device as QDMIDevice
from mqt.core.qdmi import Job as QDMIJob
from mqt.core.qdmi import ProgramFormat

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import pytest


def _site(index: int) -> QDMIDevice.Site:
    """Return a site mock with one stable index."""
    site = Mock()
    site.index.return_value = index
    return cast("QDMIDevice.Site", site)


def operation(
    name: str,
    wires: int,
    parameters: int = 0,
    *,
    sites: Sequence[int] | None = None,
    site_pairs: Sequence[tuple[int, int]] | None = None,
) -> QDMIDevice.Operation:
    """Return the operation metadata consumed by the converter."""
    metadata = Mock()
    metadata.name.return_value = name
    metadata.qubits_num.return_value = wires
    metadata.parameters_num.return_value = parameters
    metadata.sites.return_value = None if sites is None else [_site(index) for index in sites]
    metadata.site_pairs.return_value = (
        None if site_pairs is None else [(_site(first), _site(second)) for first, second in site_pairs]
    )
    return cast("QDMIDevice.Operation", metadata)


def _standard_operations(program_format: ProgramFormat) -> list[QDMIDevice.Operation]:
    """Return the gate set used by execution tests."""
    qasm2 = program_format == ProgramFormat.OPENQASM2
    return [
        operation("id" if qasm2 else "i", 1),
        operation("x", 1),
        operation("y", 1),
        operation("z", 1),
        operation("h", 1),
        operation("s", 1),
        operation("sdg", 1),
        operation("t", 1),
        operation("tdg", 1),
        operation("rx", 1, 1),
        operation("ry", 1, 1),
        operation("rz", 1, 1),
        operation("u1" if qasm2 else "p", 1, 1),
        operation("cx", 2),
        operation("cz", 2),
        operation("swap", 2),
        operation("ccx", 3),
        operation("cswap", 3),
    ]


def bell_results(_program: str, shots: int) -> list[str]:
    """Return an even Bell-state histogram."""
    zeros = shots // 2
    return ["00"] * zeros + ["11"] * (shots - zeros)


def rotation_results(program: str, shots: int) -> list[str]:
    """Return deterministic counts for the final RY instruction."""
    matches = re.findall(r"ry\(([-+0-9.eE]+)\)", program)
    angle = float(matches[-1]) if matches else 0.0
    ones = round(shots * math.sin(angle / 2) ** 2)
    return ["0"] * (shots - ones) + ["1"] * ones


class StubDevice:
    """Small QDMI boundary stub with configurable capabilities and results."""

    def __init__(
        self,
        operations: Sequence[QDMIDevice.Operation],
        formats: Sequence[ProgramFormat],
        *,
        qubits: int = 2,
        coupling_map: Sequence[tuple[int, int]] | None = None,
        result_factory: Callable[[str, int], Sequence[str]] = bell_results,
        expose_shots: bool = True,
        program_features: Sequence[object] | None = (),
    ) -> None:
        """Store the advertised capabilities and result behavior."""
        self._operations = list(operations)
        self._formats = list(formats)
        self._qubits = qubits
        self._coupling_map = coupling_map
        self._result_factory = result_factory
        self._expose_shots = expose_shots
        self._program_features = program_features
        self.submissions: list[tuple[str, ProgramFormat, int, Mapping[str, object]]] = []
        self.events: list[str] = []

    @staticmethod
    def name() -> str:
        """Return the stable stub name."""
        return "stub.qdmi"

    def operations(self) -> list[QDMIDevice.Operation]:
        """Return the advertised operations."""
        return self._operations

    def supported_program_formats(self) -> list[ProgramFormat]:
        """Return the advertised program formats."""
        return self._formats

    def try_program_features(self, _program_format: ProgramFormat) -> list[object] | None:
        """Return known feature records, or unknown metadata."""
        if self._program_features is None:
            return None
        return [
            SimpleNamespace(id=feature, value=0, constraint_id="", constraint_value=0)
            if isinstance(feature, str)
            else feature
            for feature in self._program_features
        ]

    def qubits_num(self) -> int:
        """Return the device width."""
        return self._qubits

    def coupling_map(self) -> list[tuple[QDMIDevice.Site, QDMIDevice.Site]] | None:
        """Return the optional topology."""
        if self._coupling_map is None:
            return None
        return [(_site(first), _site(second)) for first, second in self._coupling_map]

    def submit_job(
        self,
        program: str,
        program_format: ProgramFormat,
        num_shots: int,
        **parameters: object,
    ) -> QDMIJob:
        """Record one submission and return an immediately completed job.

        Returns:
            The completed job mock.
        """
        self.submissions.append((program, program_format, num_shots, parameters))
        shots = list(self._result_factory(program, num_shots))
        job = Mock()
        job_id = str(len(self.submissions))
        job.id = job_id
        self.events.append(f"submit:{job_id}")

        def wait() -> bool:
            self.events.append(f"wait:{job_id}")
            return True

        def cancel() -> None:
            self.events.append(f"cancel:{job_id}")

        job.wait.side_effect = wait
        job.cancel.side_effect = cancel
        job.check.return_value = QDMIJob.Status.DONE
        if self._expose_shots:
            job.get_shots.return_value = shots
        else:
            job.get_shots.side_effect = RuntimeError("Not supported")
        job.get_counts.return_value = dict(Counter(shots))
        return cast("QDMIJob", job)


def stub_device(
    *,
    program_format: ProgramFormat = ProgramFormat.OPENQASM3,
    operations: Sequence[QDMIDevice.Operation] | None = None,
    qubits: int = 2,
    result_factory: Callable[[str, int], Sequence[str]] = bell_results,
    expose_shots: bool = True,
    program_features: Sequence[object] | None = (),
) -> StubDevice:
    """Return a stub with the ordinary execution-test gate set."""
    return StubDevice(
        _standard_operations(program_format) if operations is None else operations,
        [program_format],
        qubits=qubits,
        result_factory=result_factory,
        expose_shots=expose_shots,
        program_features=program_features,
    )


def patch_open_device(monkeypatch: pytest.MonkeyPatch, device: StubDevice) -> None:
    """Route fresh stable-ID opens to a test double."""
    monkeypatch.setattr(
        "mqt.core.plugins.pennylane.device.open_device",
        lambda *_args, **_kwargs: cast("QDMIDevice", device),
    )
