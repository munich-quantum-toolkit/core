# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the IQM plugin entry point."""

from __future__ import annotations

from iqm.qdmi import qiskit as iqm_qiskit

from mqt.core.plugins import iqm as mqt_iqm


def test_iqm_backend_uses_upstream_wrapper() -> None:
    """The MQT Core IQM module should expose the upstream IQM backend class."""
    assert mqt_iqm.IQMBackend is iqm_qiskit.IQMBackend


def test_iqm_module_exports_backend() -> None:
    """The IQM module should only expose the upstream backend entry point."""
    assert mqt_iqm.__all__ == ["IQMBackend"]
