# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for OptionalDependencyTester."""

from __future__ import annotations

import sys
import warnings

import pytest

from mqt.core._compat.optional import OptionalDependencyTester  # noqa: PLC2701


def test_available_module() -> None:
    """Test with a module that should be available (sys)."""
    tester = OptionalDependencyTester("sys")
    assert tester
    assert bool(tester)


def test_unavailable_module() -> None:
    """Test with a module that definitely doesn't exist."""
    tester = OptionalDependencyTester("this_module_does_not_exist_xyz123")
    assert not tester
    assert not bool(tester)


def test_caching() -> None:
    """Test that availability check is cached."""
    tester = OptionalDependencyTester("sys")

    # First check
    assert tester

    # Cache should be set now
    assert tester._bool  # noqa: SLF001

    # Second check should use cache
    assert tester


def test_require_now_available() -> None:
    """Test require_now with an available module."""
    tester = OptionalDependencyTester("sys")
    # Should not raise
    tester.require_now("test this")


def test_require_now_unavailable() -> None:
    """Test require_now with an unavailable module."""
    tester = OptionalDependencyTester("this_module_does_not_exist_xyz123")

    with pytest.raises(ImportError, match="required to test this feature"):
        tester.require_now("test this feature")


def test_require_module_available() -> None:
    """Test require_module returns the imported module."""
    tester = OptionalDependencyTester("sys")
    module = tester.require_module("test this")

    assert module is sys


def test_require_module_unavailable() -> None:
    """Test require_module raises for unavailable module."""
    tester = OptionalDependencyTester("this_module_does_not_exist_xyz123")

    with pytest.raises(ImportError, match="required to use feature"):
        tester.require_module("use feature")


def test_custom_install_message() -> None:
    """Test custom installation message appears in errors."""
    tester = OptionalDependencyTester(
        "nonexistent_module",
        install_msg="Install with 'pip install special-package'",
    )

    with pytest.raises(ImportError, match="pip install special-package"):
        tester.require_now("test")


def test_disable_locally() -> None:
    """Test disable_locally context manager."""
    tester = OptionalDependencyTester("sys")

    # Initially available
    assert tester

    # Temporarily disabled
    with tester.disable_locally():
        assert not tester

    # Back to available after context
    assert tester


def test_disable_locally_preserves_state() -> None:
    """Test disable_locally preserves original state including None."""
    tester = OptionalDependencyTester("sys")
    # Don't check yet, so _bool is None
    assert tester._bool is None  # noqa: SLF001

    with tester.disable_locally():
        assert not tester

    # Should be None again (not True)
    assert tester._bool is None  # noqa: SLF001


def test_disable_locally_with_unavailable() -> None:
    """Test disable_locally works with already unavailable modules."""
    tester = OptionalDependencyTester("this_module_does_not_exist_xyz123")

    # Check it's unavailable
    assert not tester

    # Disable locally (should stay False)
    with tester.disable_locally():
        assert not tester

    # Still unavailable after context
    assert not tester


def test_module_name_property() -> None:
    """Test module_name property."""
    tester = OptionalDependencyTester("test_module")
    assert tester.module_name == "test_module"


def test_repr() -> None:
    """Test string representation."""
    tester_available = OptionalDependencyTester("sys")
    tester_unavailable = OptionalDependencyTester("nonexistent")

    repr_available = repr(tester_available)
    repr_unavailable = repr(tester_unavailable)

    assert "sys" in repr_available
    assert "available" in repr_available

    assert "nonexistent" in repr_unavailable
    assert "not available" in repr_unavailable


def test_warn_on_fail() -> None:
    """Test that warn_on_fail emits warnings when enabled."""
    tester = OptionalDependencyTester(
        "this_module_does_not_exist_xyz123",
        warn_on_fail=True,
    )

    with pytest.warns(ImportWarning, match="failed to import"):
        _ = bool(tester)


def test_no_warn_by_default() -> None:
    """Test that warnings are not emitted by default."""
    tester = OptionalDependencyTester("this_module_does_not_exist_xyz123")

    # Should not emit warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = bool(tester)

    # Filter for ImportWarnings
    import_warnings = [warning for warning in w if issubclass(warning.category, ImportWarning)]
    assert len(import_warnings) == 0
