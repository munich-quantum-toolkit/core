# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test doubles for the QDMI PennyLane plugin."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pennylane as qp

from mqt.core import fomac

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


@dataclass(frozen=True)
class FakeSite:
    """A QDMI site test double."""

    site_index: int

    def index(self) -> int:
        """Return the site index."""
        return self.site_index


class FakeOperation:
    """A QDMI operation test double."""

    def __init__(
        self,
        name: str,
        wires: int,
        parameters: int,
        *,
        sites: Sequence[int] | None = None,
        site_pairs: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        """Initialize operation metadata and optional loci."""
        self._name = name
        self._wires = wires
        self._parameters = parameters
        self._sites = None if sites is None else [FakeSite(index) for index in sites]
        self._site_pairs = (
            None if site_pairs is None else [(FakeSite(first), FakeSite(second)) for first, second in site_pairs]
        )

    def name(self) -> str:
        """Return the operation spelling."""
        return self._name

    def qubits_num(self) -> int:
        """Return the operation arity."""
        return self._wires

    def parameters_num(self) -> int:
        """Return the number of parameters."""
        return self._parameters

    def sites(self) -> list[FakeSite] | None:
        """Return advertised one-qubit sites."""
        return self._sites

    def site_pairs(self) -> list[tuple[FakeSite, FakeSite]] | None:
        """Return advertised two-qubit loci."""
        return self._site_pairs


class FakeJob:
    """A completed QDMI job test double."""

    def __init__(
        self,
        job_id: str,
        shots: Sequence[str],
        *,
        expose_shots: bool = True,
    ) -> None:
        """Initialize completed result data."""
        self.id = job_id
        self._shots = list(shots)
        self._expose_shots = expose_shots

    def wait(self) -> bool:
        """Report immediate completion.

        Returns:
            Always ``True`` for this completed test job.
        """
        return self._shots is not None

    def check(self) -> fomac.Job.Status:
        """Return the successful terminal state."""
        assert self._shots is not None
        return fomac.Job.Status.DONE

    def get_shots(self) -> list[str]:
        """Return shots or emulate a histogram-only device.

        Raises:
            RuntimeError: If this test job exposes only its histogram.
        """
        if not self._expose_shots:
            msg = "Not supported"
            raise RuntimeError(msg)
        return self._shots

    def get_counts(self) -> dict[str, int]:
        """Aggregate the stored shots.

        Returns:
            Counts keyed by QDMI bit string.
        """
        counts: dict[str, int] = {}
        for shot in self._shots:
            counts[shot] = counts.get(shot, 0) + 1
        return counts


class FakeDevice:
    """A gate-based QDMI device test double."""

    def __init__(
        self,
        operations: Sequence[FakeOperation],
        formats: Sequence[fomac.ProgramFormat],
        *,
        qubits: int = 2,
        coupling_map: Sequence[tuple[int, int]] | None = None,
        result_factory: Callable[[str, int], Sequence[str]] | None = None,
        expose_shots: bool = True,
    ) -> None:
        """Initialize capabilities and deterministic result generation."""
        self._name = "fake.qdmi"
        self._operations = list(operations)
        self._formats = list(formats)
        self._qubits = qubits
        self._coupling_map = coupling_map
        self._result_factory = result_factory or bell_results
        self._expose_shots = expose_shots
        self.submissions: list[tuple[str, fomac.ProgramFormat, int, Mapping[str, object]]] = []

    def name(self) -> str:
        """Return the test-device name."""
        return self._name

    def operations(self) -> list[FakeOperation]:
        """Return advertised operations."""
        return self._operations

    def supported_program_formats(self) -> list[fomac.ProgramFormat]:
        """Return advertised program formats."""
        return self._formats

    def qubits_num(self) -> int:
        """Return the device width."""
        return self._qubits

    def coupling_map(self) -> list[tuple[FakeSite, FakeSite]] | None:
        """Return the optional topology."""
        if self._coupling_map is None:
            return None
        return [(FakeSite(first), FakeSite(second)) for first, second in self._coupling_map]

    def submit_job(
        self,
        program: str,
        program_format: fomac.ProgramFormat,
        num_shots: int,
        **parameters: object,
    ) -> FakeJob:
        """Record one sequential submission and return its generated results.

        Returns:
            An immediately completed fake job.
        """
        self.submissions.append((program, program_format, num_shots, parameters))
        shots = self._result_factory(program, num_shots)
        return FakeJob(str(len(self.submissions)), shots, expose_shots=self._expose_shots)


def operation(
    name: str,
    wires: int,
    parameters: int = 0,
    *,
    sites: Sequence[int] | None = None,
    site_pairs: Sequence[tuple[int, int]] | None = None,
) -> FakeOperation:
    """Construct a typed fake operation.

    Returns:
        The configured operation test double.
    """
    return FakeOperation(name, wires, parameters, sites=sites, site_pairs=site_pairs)


def standard_operations(*, qasm2: bool = False) -> list[FakeOperation]:
    """Return operations sufficient for ordinary preprocessing tests.

    Returns:
        A QASM3 or QASM2-flavoured operation set.
    """
    cnot = "cx"
    identity = "id" if qasm2 else "i"
    phase = "u1" if qasm2 else "p"
    return [
        operation(identity, 1),
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
        operation(phase, 1, 1),
        operation(cnot, 2),
        operation("cz", 2),
        operation("swap", 2),
        operation("ccx", 3),
        operation("cswap", 3),
    ]


def bell_results(_program: str, shots: int) -> list[str]:
    """Return an even Bell-state histogram.

    Returns:
        ``00`` and ``11`` shots in equal proportions.
    """
    zeros = shots // 2
    return ["00"] * zeros + ["11"] * (shots - zeros)


def rotation_results(program: str, shots: int) -> list[str]:
    """Sample the one-qubit probability encoded by the last RY instruction.

    Returns:
        Deterministic counts matching the rounded RY probability.
    """
    matches = re.findall(r"ry\(([-+0-9.eE]+)\)", program)
    angle = float(matches[-1]) if matches else 0.0
    ones = round(shots * math.sin(angle / 2) ** 2)
    return ["0"] * (shots - ones) + ["1"] * ones


def qasm3_device(
    *,
    operations: Sequence[FakeOperation] | None = None,
    qubits: int = 2,
    result_factory: Callable[[str, int], Sequence[str]] | None = None,
    expose_shots: bool = True,
) -> FakeDevice:
    """Construct a typical QASM3 fake device.

    Returns:
        The configured fake QDMI device.
    """
    return FakeDevice(
        standard_operations() if operations is None else operations,
        [fomac.ProgramFormat.QASM3],
        qubits=qubits,
        result_factory=result_factory,
        expose_shots=expose_shots,
    )


def qasm2_device() -> FakeDevice:
    """Construct a typical QASM2-only fake device.

    Returns:
        The configured fake QDMI device.
    """
    return FakeDevice(
        standard_operations(qasm2=True),
        [fomac.ProgramFormat.QASM2],
    )


def require_pennylane_alias() -> str:
    """Exercise the required PennyLane import alias in this helper.

    Returns:
        The installed PennyLane version.
    """
    return qp.__version__
