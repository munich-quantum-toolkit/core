# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the program serializer registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mqt.core.plugins.qiskit import serializers
from mqt.core.qdmi import ProgramFormat

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qiskit.circuit import QuantumCircuit

    from mqt.core.plugins.qiskit import QDMIBackend


def _serializer(circuit: QuantumCircuit, backend: QDMIBackend) -> str:  # ruff:ignore[unused-function-argument]
    """Returns a fixed program.

    Args:
        circuit: The circuit to serialize.
        backend: The backend that runs the circuit.

    Returns:
        A fixed program string.
    """
    return "program"


@pytest.fixture
def private_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the test an empty registry and restore the shared one afterwards.

    The registry is a module-level dictionary, so a test that registered into it
    directly would change what later tests see. ``monkeypatch`` puts the shared
    registry back when the test ends.
    """
    monkeypatch.setattr(serializers, "_SERIALIZERS", {})
    monkeypatch.setattr(serializers, "_ENTRY_POINTS_LOADED", True)


def _load_entry_points(monkeypatch: pytest.MonkeyPatch, entry_points: Sequence[object]) -> None:
    """Make the registry load one fixed set of entry points on next use.

    Args:
        monkeypatch: The monkeypatch fixture.
        entry_points: The entry points the registry should discover.
    """
    monkeypatch.setattr(serializers, "entry_points", lambda group: entry_points)  # ruff:ignore[unused-lambda-argument]
    monkeypatch.setattr(serializers, "_ENTRY_POINTS_LOADED", False)
    monkeypatch.setattr(serializers, "_SERIALIZERS", {})


class _FakeEntryPoint:
    """An entry point that yields a fixed object, or raises."""

    def __init__(self, name: str, value: str, target: object | None = None) -> None:
        """Initialize the entry point.

        Args:
            name: Entry point name, expected to be a program format name.
            value: Entry point value, used in diagnostics.
            target: What loading returns. Loading raises when this is None.
        """
        self.name = name
        self.value = value
        self._target = target

    def load(self) -> object:
        """Returns the target, or raises if there is none.

        Raises:
            ImportError: If the entry point has no target.
        """
        if self._target is None:
            msg = "no such module"
            raise ImportError(msg)
        return self._target


def test_register_and_look_up_a_serializer(private_registry: None) -> None:  # ruff:ignore[unused-function-argument]
    """A registered serializer is the one the lookup returns."""
    serializers.register_program_serializer(ProgramFormat.CUSTOM1, _serializer)

    assert serializers.program_serializer(ProgramFormat.CUSTOM1) is _serializer


def test_second_serializer_for_one_format_is_rejected(private_registry: None) -> None:  # ruff:ignore[unused-function-argument]
    """Two serializers for one format need an explicit override."""
    serializers.register_program_serializer(ProgramFormat.CUSTOM1, _serializer)

    with pytest.raises(ValueError, match="already registered"):
        serializers.register_program_serializer(ProgramFormat.CUSTOM1, _serializer)


def test_replace_overrides_an_existing_serializer(private_registry: None) -> None:  # ruff:ignore[unused-function-argument]
    """``replace=True`` puts the new serializer in place of the old one."""

    def other(circuit: QuantumCircuit, backend: QDMIBackend) -> str:  # ruff:ignore[unused-function-argument]
        return "other program"

    serializers.register_program_serializer(ProgramFormat.CUSTOM1, _serializer)
    serializers.register_program_serializer(ProgramFormat.CUSTOM1, other, replace=True)

    assert serializers.program_serializer(ProgramFormat.CUSTOM1) is other


def test_unregistering_an_unknown_format_does_nothing(private_registry: None) -> None:  # ruff:ignore[unused-function-argument]
    """Removing a serializer that was never registered is a no-op."""
    serializers.unregister_program_serializer(ProgramFormat.CUSTOM1)

    assert serializers.program_serializer(ProgramFormat.CUSTOM1) is None


@pytest.mark.parametrize("fmt", [ProgramFormat.CALIBRATION, ProgramFormat.BATCH_JOB])
def test_format_without_program_payload_is_rejected(fmt: ProgramFormat, private_registry: None) -> None:  # ruff:ignore[unused-function-argument]
    """A format that carries no program cannot have a serializer."""
    with pytest.raises(ValueError, match="carries no program payload"):
        serializers.register_program_serializer(fmt, _serializer)


def test_serializer_is_discovered_from_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registry loads a serializer advertised through the entry point group."""
    _load_entry_points(monkeypatch, [_FakeEntryPoint("CUSTOM2", "pkg.mod:serializer", _serializer)])

    assert serializers.program_serializer(ProgramFormat.CUSTOM2) is _serializer


def test_entry_point_with_unknown_format_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entry point that does not name a program format is skipped."""
    _load_entry_points(monkeypatch, [_FakeEntryPoint("NOT_A_FORMAT", "pkg.mod:serializer", _serializer)])

    with pytest.warns(UserWarning, match="does not name a program format"):
        assert serializers.program_serializer(ProgramFormat.CUSTOM2) is None


def test_entry_point_for_payloadless_format_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entry point for a format without a program payload is skipped."""
    _load_entry_points(monkeypatch, [_FakeEntryPoint("CALIBRATION", "pkg.mod:serializer", _serializer)])

    with pytest.warns(UserWarning, match="carries no program payload"):
        assert serializers.program_serializer(ProgramFormat.CALIBRATION) is None


def test_entry_point_that_fails_to_load_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """A serializer that cannot be imported is skipped without hiding the others."""
    _load_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint("CUSTOM2", "broken.mod:serializer"),
            _FakeEntryPoint("CUSTOM3", "pkg.mod:serializer", _serializer),
        ],
    )

    with pytest.warns(UserWarning, match="Failed to load the program serializer for CUSTOM2"):
        assert serializers.program_serializer(ProgramFormat.CUSTOM2) is None

    assert serializers.program_serializer(ProgramFormat.CUSTOM3) is _serializer


def test_runtime_registration_beats_an_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    """A serializer registered at run time wins over one from an entry point."""

    def other(circuit: QuantumCircuit, backend: QDMIBackend) -> str:  # ruff:ignore[unused-function-argument]
        return "other program"

    _load_entry_points(monkeypatch, [_FakeEntryPoint("CUSTOM2", "pkg.mod:serializer", _serializer)])
    serializers.register_program_serializer(ProgramFormat.CUSTOM2, other)

    assert serializers.program_serializer(ProgramFormat.CUSTOM2) is other


def test_preferred_program_formats_orders_a_shuffled_list() -> None:
    """The preference tuple decides the order, not the order the device reports."""
    reported = [
        ProgramFormat.QASM2,
        ProgramFormat.QPY,
        ProgramFormat.IQM_JSON,
        ProgramFormat.QASM3,
        ProgramFormat.QIR_BASE_STRING,
    ]

    assert serializers.preferred_program_formats(reported) == [
        ProgramFormat.IQM_JSON,
        ProgramFormat.QPY,
        ProgramFormat.QASM3,
        ProgramFormat.QIR_BASE_STRING,
        ProgramFormat.QASM2,
    ]


def test_preferred_program_formats_drops_formats_without_payload() -> None:
    """A format that carries no program cannot be serialized into."""
    reported = [ProgramFormat.CALIBRATION, ProgramFormat.QASM3, ProgramFormat.BATCH_JOB]

    assert serializers.preferred_program_formats(reported) == [ProgramFormat.QASM3]


def test_preferred_program_formats_puts_unnamed_formats_last(monkeypatch: pytest.MonkeyPatch) -> None:
    """A format the preference tuple does not name keeps its reported position at the end."""
    monkeypatch.setattr(serializers, "PROGRAM_FORMAT_PREFERENCE", (ProgramFormat.QASM3,))
    reported = [ProgramFormat.QPY, ProgramFormat.QASM2, ProgramFormat.QASM3]

    assert serializers.preferred_program_formats(reported) == [
        ProgramFormat.QASM3,
        ProgramFormat.QPY,
        ProgramFormat.QASM2,
    ]
