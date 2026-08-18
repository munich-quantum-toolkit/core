# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the program serializer registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from mqt.core.plugins.qiskit import serializers
from mqt.core.qdmi import ProgramFormat

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from importlib.metadata import EntryPoint

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


def _registry(entry_points: Sequence[EntryPoint] = ()) -> serializers._ProgramSerializerRegistry:
    """Returns a registry of its own that discovers one fixed set of entry points.

    Each registry owns its serializers and its load state, so a test never
    touches what another test sees.

    Args:
        entry_points: The entry points the registry should discover.
    """
    return serializers._ProgramSerializerRegistry(lambda: entry_points)  # ruff:ignore[private-member-access]


class _FakeEntryPoint:
    """An entry point that yields a fixed object, or raises."""

    def __init__(
        self,
        name: str,
        value: str,
        target: object | None = None,
        on_load: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the entry point.

        Args:
            name: Entry point name, expected to be a program format name.
            value: Entry point value, used in diagnostics.
            target: What loading returns. Loading raises when this is None.
            on_load: Called while loading, standing in for the import side
                effects of the package that advertises the entry point.
        """
        self.name = name
        self.value = value
        self._target = target
        self._on_load = on_load

    def load(self) -> object:
        """Returns the target, or raises if there is none.

        Raises:
            ImportError: If the entry point has no target.
        """
        if self._on_load is not None:
            self._on_load()
        if self._target is None:
            msg = "no such module"
            raise ImportError(msg)
        return self._target


def _entry_point(
    name: str,
    value: str,
    target: object | None = None,
    on_load: Callable[[], None] | None = None,
) -> EntryPoint:
    """Returns a stand-in entry point, typed as the real thing for the registry.

    Args:
        name: Entry point name, expected to be a program format name.
        value: Entry point value, used in diagnostics.
        target: What loading returns. Loading raises when this is None.
        on_load: Called while loading, standing in for import side effects.
    """
    return cast("EntryPoint", _FakeEntryPoint(name, value, target, on_load))


def test_register_and_look_up_a_serializer() -> None:
    """A registered serializer is the one the lookup returns."""
    registry = _registry()
    registry.register(ProgramFormat.CUSTOM1, _serializer)

    assert registry.get(ProgramFormat.CUSTOM1) is _serializer


def test_second_serializer_for_one_format_is_rejected() -> None:
    """Two serializers for one format need an explicit override."""
    registry = _registry()
    registry.register(ProgramFormat.CUSTOM1, _serializer)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(ProgramFormat.CUSTOM1, _serializer)


def test_replace_overrides_an_existing_serializer() -> None:
    """``replace=True`` puts the new serializer in place of the old one."""

    def other(circuit: QuantumCircuit, backend: QDMIBackend) -> str:  # ruff:ignore[unused-function-argument]
        return "other program"

    registry = _registry()
    registry.register(ProgramFormat.CUSTOM1, _serializer)
    registry.register(ProgramFormat.CUSTOM1, other, replace=True)

    assert registry.get(ProgramFormat.CUSTOM1) is other


def test_unregistering_an_unknown_format_does_nothing() -> None:
    """Removing a serializer that was never registered is a no-op."""
    registry = _registry()
    registry.unregister(ProgramFormat.CUSTOM1)

    assert registry.get(ProgramFormat.CUSTOM1) is None


@pytest.mark.parametrize("fmt", [ProgramFormat.CALIBRATION, ProgramFormat.BATCH_JOB])
def test_non_circuit_format_is_rejected(fmt: ProgramFormat) -> None:
    """A format that does not carry a serialized circuit cannot have a serializer."""
    with pytest.raises(ValueError, match="does not carry a serialized circuit"):
        _registry().register(fmt, _serializer)


def test_module_functions_share_one_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public functions read and write the same process-wide registry."""
    monkeypatch.setattr(serializers, "_REGISTRY", _registry())
    serializers.register_program_serializer(ProgramFormat.CUSTOM1, _serializer)

    assert serializers.program_serializer(ProgramFormat.CUSTOM1) is _serializer

    serializers.unregister_program_serializer(ProgramFormat.CUSTOM1)

    assert serializers.program_serializer(ProgramFormat.CUSTOM1) is None


def test_serializer_is_discovered_from_entry_point() -> None:
    """The registry loads a serializer advertised through the entry point group."""
    registry = _registry([_entry_point("CUSTOM2", "pkg.mod:serializer", _serializer)])

    assert registry.get(ProgramFormat.CUSTOM2) is _serializer


def test_entry_points_are_read_once() -> None:
    """A second lookup does not read the entry points again."""
    calls = 0

    def discover() -> list[EntryPoint]:
        nonlocal calls
        calls += 1
        return [_entry_point("CUSTOM2", "pkg.mod:serializer", _serializer)]

    registry = serializers._ProgramSerializerRegistry(discover)  # ruff:ignore[private-member-access]
    registry.get(ProgramFormat.CUSTOM2)
    registry.get(ProgramFormat.CUSTOM2)

    assert calls == 1


def test_entry_point_with_unknown_format_warns() -> None:
    """An entry point that does not name a program format is skipped."""
    registry = _registry([_entry_point("NOT_A_FORMAT", "pkg.mod:serializer", _serializer)])

    with pytest.warns(UserWarning, match="does not name a program format"):
        assert registry.get(ProgramFormat.CUSTOM2) is None


def test_entry_point_for_non_circuit_format_warns() -> None:
    """An entry point for a format that carries no serialized circuit is skipped."""
    registry = _registry([_entry_point("CALIBRATION", "pkg.mod:serializer", _serializer)])

    with pytest.warns(UserWarning, match="does not carry a serialized circuit"):
        assert registry.get(ProgramFormat.CALIBRATION) is None


def test_entry_point_that_fails_to_load_warns() -> None:
    """A serializer that cannot be imported is skipped without hiding the others."""
    registry = _registry([
        _entry_point("CUSTOM2", "broken.mod:serializer"),
        _entry_point("CUSTOM3", "pkg.mod:serializer", _serializer),
    ])

    with pytest.warns(UserWarning, match="Failed to load the program serializer for CUSTOM2"):
        assert registry.get(ProgramFormat.CUSTOM2) is None

    assert registry.get(ProgramFormat.CUSTOM3) is _serializer


def test_runtime_registration_beats_an_entry_point() -> None:
    """A serializer registered at run time wins over one from an entry point."""

    def other(circuit: QuantumCircuit, backend: QDMIBackend) -> str:  # ruff:ignore[unused-function-argument]
        return "other program"

    registry = _registry([_entry_point("CUSTOM2", "pkg.mod:serializer", _serializer)])
    registry.register(ProgramFormat.CUSTOM2, other)

    assert registry.get(ProgramFormat.CUSTOM2) is other


def test_registration_during_discovery_wins() -> None:
    """A serializer registered while an entry point loads keeps precedence."""

    def other(circuit: QuantumCircuit, backend: QDMIBackend) -> str:  # ruff:ignore[unused-function-argument]
        return "other program"

    def register_other() -> None:
        registry.register(ProgramFormat.CUSTOM2, other)

    registry = _registry([_entry_point("CUSTOM2", "pkg.mod:serializer", _serializer, register_other)])

    assert registry.get(ProgramFormat.CUSTOM2) is other


def test_lookup_during_discovery_does_not_start_a_second_pass() -> None:
    """A re-entrant lookup returns without reading the entry points again."""
    calls = 0
    seen: list[object] = []

    def look_up_again() -> None:
        seen.append(registry.get(ProgramFormat.CUSTOM2))

    def discover() -> list[EntryPoint]:
        nonlocal calls
        calls += 1
        return [_entry_point("CUSTOM2", "pkg.mod:serializer", _serializer, look_up_again)]

    registry = serializers._ProgramSerializerRegistry(discover)  # ruff:ignore[private-member-access]

    assert registry.get(ProgramFormat.CUSTOM2) is _serializer
    # Discovery ran once, and the re-entrant lookup saw no half-built result.
    assert calls == 1
    assert seen == [None]


def test_discovery_is_retried_after_it_aborts() -> None:
    """A failure that stops discovery leaves the registry cold rather than empty."""
    attempts = 0

    def discover() -> list[EntryPoint]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            msg = "distribution metadata is unreadable"
            raise RuntimeError(msg)
        return [_entry_point("CUSTOM2", "pkg.mod:serializer", _serializer)]

    registry = serializers._ProgramSerializerRegistry(discover)  # ruff:ignore[private-member-access]

    with pytest.raises(RuntimeError, match="unreadable"):
        registry.get(ProgramFormat.CUSTOM2)

    assert registry.get(ProgramFormat.CUSTOM2) is _serializer
    assert attempts == 2


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


def test_preferred_program_formats_drops_non_circuit_formats() -> None:
    """A format that carries no serialized circuit cannot be serialized into."""
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
