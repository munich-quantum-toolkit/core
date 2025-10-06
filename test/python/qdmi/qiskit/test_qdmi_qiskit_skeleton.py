# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Qiskit backend (dependency guards only).

These tests adapt automatically depending on whether the optional
`qiskit` dependency is present in the environment.
"""

from __future__ import annotations

import importlib.util

import pytest

from mqt.core.qdmi.qiskit import (
    QiskitNotAvailableError,
    is_available,
    require_qiskit,
)


def _qiskit_installed() -> bool:
    return importlib.util.find_spec("qiskit") is not None


def test_qiskit_availability_flag_consistency() -> None:
    """`is_available` must reflect actual importability of the qiskit package."""
    assert is_available() is _qiskit_installed()


def test_require_qiskit_behavior() -> None:
    """`require_qiskit` returns module if installed, else raises custom error."""
    if _qiskit_installed():
        mod = require_qiskit()
        assert mod.__name__ == "qiskit"
    else:
        with pytest.raises(QiskitNotAvailableError):
            require_qiskit()


def test_placeholder_attribute_access() -> None:
    """Accessing future symbols should raise the custom not-available error in Phase Q1.

    Only meaningful when qiskit is not installed to exercise the dynamic attribute path.
    """
    if not _qiskit_installed():
        with pytest.raises(QiskitNotAvailableError):  # noqa: PT012
            # Accessing a placeholder attribute triggers __getattr__ path.
            import importlib

            qk = importlib.import_module("mqt.core.qdmi.qiskit")
            _ = qk.QiskitBackend
