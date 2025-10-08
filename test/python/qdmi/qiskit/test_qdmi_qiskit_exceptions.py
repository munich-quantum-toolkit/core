# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QDMI Qiskit exception types."""

from __future__ import annotations

import pytest

from mqt.core.qdmi.qiskit import (
    CapabilityMismatchError,
    QDMIQiskitError,
    TranslationError,
    UnsupportedOperationError,
)


def test_qdmi_qiskit_error_is_runtime_error() -> None:
    """QDMIQiskitError should be a subclass of RuntimeError."""
    assert issubclass(QDMIQiskitError, RuntimeError)


def test_qdmi_qiskit_error_can_be_raised() -> None:
    """QDMIQiskitError can be raised with a message.

    Raises:
        QDMIQiskitError: When the error is raised.
    """
    msg = "test error"
    with pytest.raises(QDMIQiskitError, match="test error"):
        raise QDMIQiskitError(msg)


def test_unsupported_operation_error_is_qdmi_error() -> None:
    """UnsupportedOperationError should be a subclass of QDMIQiskitError."""
    assert issubclass(UnsupportedOperationError, QDMIQiskitError)


def test_unsupported_operation_error_can_be_raised() -> None:
    """UnsupportedOperationError can be raised with a message.

    Raises:
        UnsupportedOperationError: When the error is raised.
    """
    msg = "operation not supported"
    with pytest.raises(UnsupportedOperationError, match="operation not supported"):
        raise UnsupportedOperationError(msg)


def test_translation_error_is_qdmi_error() -> None:
    """TranslationError should be a subclass of QDMIQiskitError."""
    assert issubclass(TranslationError, QDMIQiskitError)


def test_translation_error_can_be_raised() -> None:
    """TranslationError can be raised with a message.

    Raises:
        TranslationError: When the error is raised.
    """
    msg = "translation failed"
    with pytest.raises(TranslationError, match="translation failed"):
        raise TranslationError(msg)


def test_capability_mismatch_error_is_qdmi_error() -> None:
    """CapabilityMismatchError should be a subclass of QDMIQiskitError."""
    assert issubclass(CapabilityMismatchError, QDMIQiskitError)


def test_capability_mismatch_error_can_be_raised() -> None:
    """CapabilityMismatchError can be raised with a message.

    Raises:
        CapabilityMismatchError: When the error is raised.
    """
    msg = "capability mismatch"
    with pytest.raises(CapabilityMismatchError, match="capability mismatch"):
        raise CapabilityMismatchError(msg)


def test_exception_hierarchy() -> None:
    """All custom exceptions should inherit from QDMIQiskitError."""
    for exc_class in [UnsupportedOperationError, TranslationError, CapabilityMismatchError]:
        assert issubclass(exc_class, QDMIQiskitError)
        assert issubclass(exc_class, RuntimeError)


def test_exception_with_cause() -> None:
    """Exceptions can be raised with a cause.

    Raises:
        TranslationError: When the error is raised.
    """
    original = ValueError("original error")
    msg = "wrapped error"

    try:
        raise original
    except ValueError as e:
        with pytest.raises(TranslationError) as exc_info:
            raise TranslationError(msg) from e

    assert exc_info.value.__cause__ is original


def test_unsupported_operation_error_message() -> None:
    """UnsupportedOperationError should preserve the error message.

    Raises:
        UnsupportedOperationError: When the error is raised.
    """
    msg = "Gate 'custom_gate' is not supported"
    with pytest.raises(UnsupportedOperationError) as exc_info:
        raise UnsupportedOperationError(msg)

    assert str(exc_info.value) == msg


def test_translation_error_message() -> None:
    """TranslationError should preserve the error message.

    Raises:
        TranslationError: When the error is raised.
    """
    msg = "Failed to translate parameter"
    with pytest.raises(TranslationError) as exc_info:
        raise TranslationError(msg)

    assert str(exc_info.value) == msg


def test_capability_mismatch_error_message() -> None:
    """CapabilityMismatchError should preserve the error message.

    Raises:
        CapabilityMismatchError: When the error is raised.
    """
    msg = "Device capabilities do not match"
    with pytest.raises(CapabilityMismatchError) as exc_info:
        raise CapabilityMismatchError(msg)

    assert str(exc_info.value) == msg
