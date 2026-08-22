# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for exact-payload Qiskit serializer selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from mqt.core.plugins.qiskit import serializers
from mqt.core.qdmi import PayloadDescriptor, ProgramEncoding

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

    from qiskit.circuit import QuantumCircuit

    from mqt.core.plugins.qiskit import QDMIBackend

CUSTOM = PayloadDescriptor("custom", (1, 0, 0))


def _serializer(_circuit: QuantumCircuit, _backend: QDMIBackend) -> str:
    return "program"


class _EntryPoint:
    def __init__(self, target: object | None) -> None:
        self.name = "custom"
        self.value = "pkg.mod:serializer"
        self._target = target

    def load(self) -> object:
        if self._target is None:
            msg = "no such module"
            raise ImportError(msg)
        return self._target


def _registry(*targets: object | None) -> serializers._ProgramSerializerRegistry:
    points = [cast("EntryPoint", _EntryPoint(target)) for target in targets]
    return serializers._ProgramSerializerRegistry(lambda: points)  # ruff:ignore[private-member-access]


def test_register_replace_and_unregister() -> None:
    """Use the complete descriptor as the registry key."""
    registry = _registry()
    registry.register(CUSTOM, _serializer)
    assert registry.get(CUSTOM) is _serializer
    with pytest.raises(ValueError, match="already registered"):
        registry.register(CUSTOM, _serializer)

    def replacement(_circuit: QuantumCircuit, _backend: QDMIBackend) -> str:
        return "replacement"

    registry.register(CUSTOM, replacement, replace=True)
    assert registry.get(CUSTOM) is replacement
    registry.unregister(CUSTOM)
    assert registry.get(CUSTOM) is None


def test_descriptor_variants_do_not_alias() -> None:
    """Keep version, profile, and encoding in serializer identity."""
    registry = _registry()
    registry.register(CUSTOM, _serializer)
    assert registry.get(PayloadDescriptor("custom", (1, 1, 0))) is None
    assert registry.get(PayloadDescriptor("custom", (1, 0, 0), "native")) is None
    assert registry.get(PayloadDescriptor("custom", (1, 0, 0), encoding=ProgramEncoding.BINARY)) is None


def test_entry_point_exports_descriptor_and_serializer() -> None:
    """Discover the descriptor from the loaded entry-point value."""
    assert _registry((CUSTOM, _serializer)).get(CUSTOM) is _serializer


def test_invalid_entry_point_is_skipped() -> None:
    """Do not derive descriptor identity from the entry-point name."""
    registry = _registry(_serializer)
    with pytest.warns(UserWarning, match="must export"):
        assert registry.get(CUSTOM) is None


def test_failing_entry_point_does_not_break_lookup() -> None:
    """Skip a package that cannot import."""
    registry = _registry(None, (CUSTOM, _serializer))
    with pytest.warns(UserWarning, match="Failed to load"):
        assert registry.get(CUSTOM) is _serializer
